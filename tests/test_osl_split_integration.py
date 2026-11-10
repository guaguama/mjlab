"""Stage 2 end to end on CPU, across every module of ``tasks/osl_split``.

A synthetic stage-1 run (``params/agent.yaml`` from the real stage-1 runner config
with tiny networks, a checkpoint with ``MuscleMimicPPO``'s keys) is converted by
the warm-start command line, exactly as the pipeline converts one. The real
``SplitOnPolicyRunner`` then starts from the file, trains a few iterations on a fake
env, checkpoints, and is resumed in a fresh runner the way ``--agent.resume`` does
it. What the module tests cannot see on their own, this checks across them:

* the converted file is what the algorithm loads: the human actor and the teacher
  head reproduce the stage-1 actor's rows, both critics start as its critic, and
  the lineage sets the iteration axis, the env's step counter and the env-step
  count;
* the freeze curriculum through the real loop: the human actor (weights, std and
  observation statistics) bit-identical while frozen and learning after, its std
  never moving; the OSL actor acting but not learning in the critic-only warm-up
  while its statistics move; the critics learning from the first iteration;
* every module's learning rate at every iteration, against the schedule;
* a checkpoint resumed in a fresh runner: every network, optimizer and statistic
  restored, the stage start and the curriculum kept, and training going on at the
  next iteration; and loaded by play's call into the task's default config, the
  two actors alone;
* the scalars stage 2 adds to rsl_rl's, at the right steps and values;
* a stage-2 task that switches the reward retrains its critics, and one that
  switches the model or the socket is refused, as is a stage-2 config that would
  build another architecture than stage 1 trained.

No mjlab env and no GPU: the env is a fake with what the runner, the algorithm and
the agents' rewards read. Validation and the OSL export have their own tests
(``test_osl_split_runner.py``) and are off here.
"""

from __future__ import annotations

import copy
import itertools
import statistics
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensordict import TensorDict

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_gait_library.config.rl_cfg import osl_gait_library_runner_cfg
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_mimic.rl.muon import MuonAdam
from mjlab.tasks.osl_split.config.rl_cfg import osl_split_runner_cfg
from mjlab.tasks.osl_split.mdp.rewards import CHANNELS, GROUPS, stage1_channel_weights
from mjlab.tasks.osl_split.rl.policy import std_parameter
from mjlab.tasks.osl_split.rl.ppo import RETURN_COUNT_FRESH
from mjlab.tasks.osl_split.rl.runner import SplitOnPolicyRunner
from mjlab.tasks.osl_split.rl.schedules import MODULES, ChannelSchedule, SplitSchedule
from mjlab.tasks.osl_split.rl.warmstart import LOG_STD
from mjlab.tasks.osl_split.scripts.split import main as split_main
from mjlab.utils.os import dump_yaml

NUM_ENVS = 6
NUM_STEPS = 4
ACTOR_DIM = 9
OSL_DIM = 5
HIDDEN = (16, 16)
OSL_HIDDEN = (8, 8)
ACTION_TERMS: dict[Model, tuple[tuple[str, int], ...]] = {
  "msk": (("osl_motors", 2), ("muscles", 5)),
  "robot": (("servos", 27), ("osl_motors", 2)),
}
"""Action-manager order: the OSL rows are 0:2 on the musculoskeletal model and
27:29 on the robot (whose converter cross-check pins its 27 servos)."""
SOCKET: Socket = "compliant"
EPISODE_STEPS = 3
TWIST_WEIGHT = 0.06
TWIST_SPLIT = (1.0, 0.5)

STAGE1_ITER = 40
STAGE1_COUNTER = (STAGE1_ITER + 1) * 80
STAGE1_ENV_STEPS = (STAGE1_ITER + 1) * 80 * 4096
RETURN_MEAN, RETURN_VAR = 11.9552, 97.9229
WANDB_ID = "abcd1234"
STAGE_START = STAGE1_ITER + 1

FREEZE = 2
"""``freeze.human_actor_iterations``."""
WARMUP = 1
"""``freeze.critic_warmup_iterations``."""
RAMP = 2
"""``algorithm.human_actor_ramp_iterations``."""
ANNEAL = 2
"""``channels.anneal_iterations``, starting at the human unfreeze (an
``anneal_start_iteration`` of -1)."""
SCHEDULE = 6
"""``algorithm.schedule_iterations``: longer than the first run, so a resumed run
must keep it."""
ITERATIONS = 4
RESUMED = 2
SAVE_INTERVAL = 2

NORMALIZER = ("obs_normalizer.mean", "obs_normalizer.var", "obs_normalizer.count")
COUNTS = {"human": 84, "osl_sites": 18, "osl_joints": 2}
"""The CLF's coordinates per group on the real models, for the fake shares."""

Weights = dict[str, dict[str, float]]

EXPECTED_TAGS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Train/human/mean_reward",
  "Train/osl/mean_reward",
  "Train/total_env_steps",
  "Train/osl/teacher_rms",
  "Episode_Reward/mimic",
  "Loss/learning_rate",
  "Loss/human/value",
  "Loss/osl/value",
  "Loss/human/surrogate",
  "Loss/human/entropy",
  "Loss/human/clip_fraction",
  "Loss/osl/surrogate",
  "Loss/osl/entropy",
  "Loss/osl/clip_fraction",
  "Policy/mean_std",
  "Policy/human/mean_std",
  "Policy/osl/mean_std",
  "Policy/osl/std_knee",
  "Policy/osl/std_ankle",
  "Policy/std/osl_motors",
  "Policy/noise_gap/osl_motors",
  "Policy/mean_outside/osl_motors",
  "Critic/human/explained_variance",
  "Critic/osl/explained_variance",
  "Stage2/iteration",
  "Stage2/human_frozen",
  "Stage2/osl_actor_frozen",
  "Stage2/blend_beta",
  "Stage2/reward_switch",
  *(f"Stage2/skipped_steps/{module}" for module in MODULES),
  *(f"LR/{module}" for module in MODULES),
  *(f"Channels/{c}/{g}" for c in CHANNELS for g in GROUPS),
  *(f"Channels/{c}/share_{g}" for c in CHANNELS for g in GROUPS),
)


##
# Fakes.
##


def _shares(weights: Mapping[str, float]) -> dict[str, float]:
  total = sum(weights[g] * n for g, n in COUNTS.items())
  return {g: weights[g] * n / total for g, n in COUNTS.items()}


class _SplitTerm:
  """What the runner reads and sets of ``split_mimic_reward``: the channels, the
  components and the group weights, with shares as under the CLF alone."""

  def __init__(self, num_envs: int) -> None:
    self.channel_rewards = torch.zeros(num_envs, 2)
    self.components: dict[str, torch.Tensor] = {}
    self.weights_set: list[Weights] = []
    self.weights_checked: list[Weights] = []
    self._weights = stage1_channel_weights(clf=True)

  @property
  def group_weights(self) -> Weights:
    return copy.deepcopy(self._weights)

  def check_group_weights(self, weights: Mapping[str, Mapping[str, float]]) -> Weights:
    assert set(weights) == set(CHANNELS)
    checked = {c: {g: float(w) for g, w in weights[c].items()} for c in CHANNELS}
    self.weights_checked.append(copy.deepcopy(checked))
    return checked

  def set_group_weights(self, weights: Mapping[str, Mapping[str, float]]) -> None:
    self._weights = self.check_group_weights(weights)
    self.weights_set.append(self.group_weights)

  def effective_shares(self) -> Weights:
    return {channel: _shares(w) for channel, w in self._weights.items()}


class _RewardManager:
  def __init__(self, num_envs: int) -> None:
    self.mimic = _SplitTerm(num_envs)
    twist = SimpleNamespace(components={})
    self._cfgs = {
      "mimic": SimpleNamespace(func=self.mimic, weight=1.0),
      "twist": SimpleNamespace(func=twist, weight=TWIST_WEIGHT),
    }
    self._step_reward = torch.zeros(num_envs, len(self._cfgs))

  @property
  def active_terms(self) -> list[str]:
    return list(self._cfgs)

  def get_term_cfg(self, name: str) -> Any:
    return self._cfgs[name]


class _FakeEnv:
  """The vec env and its unwrapped env in one object.

  Observations and rewards are random. Episodes time out after
  ``EPISODE_STEPS`` steps or end at random. Every step's team reward, the agents'
  rewards it implies (each channel plus the twist times its split) and the dones
  are recorded, to replay the logged episode means from.
  """

  def __init__(self, model: Model, seed: int = 0) -> None:
    terms = {name: SimpleNamespace(action_dim=dim) for name, dim in ACTION_TERMS[model]}
    self.num_envs = NUM_ENVS
    self.num_actions = sum(term.action_dim for term in terms.values())
    self.device = "cpu"
    self.max_episode_length = 1000
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
    self.cfg = SimpleNamespace(scale_rewards_by_dt=False)
    self.action_manager = SimpleNamespace(
      active_terms=list(terms), get_term=terms.__getitem__
    )
    self.reward_manager = _RewardManager(NUM_ENVS)
    self.records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    """Per step: the team reward (N,), the agents' rewards (N, 2), the dones."""
    self._generator = torch.Generator().manual_seed(seed)

  @property
  def unwrapped(self) -> _FakeEnv:
    return self

  def _rand(self, *shape: int) -> torch.Tensor:
    return torch.rand(*shape, generator=self._generator)

  def get_observations(self) -> TensorDict:
    g, n = self._generator, self.num_envs
    return TensorDict(
      {
        "actor": torch.randn(n, ACTOR_DIM, generator=g),
        "osl": 2.0 * torch.randn(n, OSL_DIM, generator=g) + 1.0,
      },
      batch_size=[n],
    )

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    n = self.num_envs
    assert actions.shape == (n, self.num_actions)
    assert bool(torch.isfinite(actions).all())
    channels, mimic = self._rand(n, 2), self._rand(n)
    twist = TWIST_WEIGHT * self._rand(n)
    manager = self.reward_manager
    manager.mimic.channel_rewards = channels
    manager.mimic.components = {"split_reward_osl": channels[:, 1]}
    manager._step_reward = torch.stack((mimic, twist), dim=-1)
    team = mimic + twist
    agents = channels + twist.unsqueeze(-1) * torch.tensor(TWIST_SPLIT)
    self.common_step_counter += 1
    self.episode_length_buf += 1
    time_outs = self.episode_length_buf >= EPISODE_STEPS
    dones = time_outs | (self._rand(n) < 0.2)
    self.episode_length_buf[dones] = 0
    self.records.append((team, agents, dones))
    extras = {"time_outs": time_outs, "log": {"Episode_Reward/mimic": mimic.mean()}}
    return self.get_observations(), team, dones.long(), extras


##
# The runner, recording every iteration.
##


@dataclass(frozen=True)
class _Snapshot:
  """What one iteration left behind."""

  it: int
  k: int
  states: dict[str, dict[str, torch.Tensor]]
  """Every module's state dict, and the teacher head's."""
  optimizer_states: dict[str, int]
  """Per module, the parameters its optimizer keeps state for."""
  lrs: dict[str, float]

  @classmethod
  def of(cls, runner: SplitOnPolicyRunner, it: int, k: int) -> _Snapshot:
    alg = runner.alg
    modules = {**alg.models, "teacher_head": alg.teacher_head}
    return cls(
      it=it,
      k=k,
      states={
        name: {key: v.detach().clone() for key, v in module.state_dict().items()}
        for name, module in modules.items()
      },
      optimizer_states={name: len(opt.state) for name, opt in alg.optimizers.items()},
      lrs=dict(alg.lrs),
    )


class _Runner(SplitOnPolicyRunner):
  """The stage-2 runner, snapshotting each iteration after its update and logging
  (``_log_stage`` runs once per iteration, last)."""

  def __init__(self, env: _FakeEnv, train_cfg: dict, log_dir: str | None) -> None:
    self.history: list[_Snapshot] = []
    super().__init__(cast(Any, env), train_cfg, log_dir, "cpu")

  def _log_stage(self, it: int, k: int) -> None:
    super()._log_stage(it, k)
    self.history.append(_Snapshot.of(self, it, k))


##
# A stage-1 run and the stage-2 config.
##


def _random_state(
  model: MuscleMimicMLPModel, generator: torch.Generator
) -> dict[str, torch.Tensor]:
  """Every tensor random, so an exact copy or slice differs from a mistake.

  The observation count is far below a real run's (2e8), where float32 would
  swallow a 6-env step's increment and hide whether the statistics moved.
  """
  state = {
    key: 0.5 * torch.randn(value.shape, generator=generator)
    for key, value in model.state_dict().items()
  }
  state["obs_normalizer.var"] = 0.5 + torch.rand(ACTOR_DIM, generator=generator)
  state["obs_normalizer.count"] = torch.tensor(1.0e4)
  if LOG_STD in state:
    state[LOG_STD] = 2.0 * torch.rand(state[LOG_STD].shape, generator=generator) - 1.5
  return state


def _unified_actor(
  model: Model,
  state: Mapping[str, torch.Tensor] | None = None,
  activation: str = "swish",
):
  out_dim = sum(dim for _, dim in ACTION_TERMS[model])
  obs = TensorDict({"actor": torch.zeros(1, ACTOR_DIM)}, batch_size=[1])
  actor = MuscleMimicMLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    out_dim,
    hidden_dims=HIDDEN,
    activation=activation,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "log",
    },
  )
  if state is not None:
    actor.load_state_dict(dict(state), strict=True)
  actor.set_norm_mode("frozen")
  return actor


def _stage1_run(root: Path, model: Model, reward: Reward) -> tuple[Path, dict]:
  """A stage-1 run as ``train.py`` leaves one, plus a W&B log naming it; returns
  the run directory and its checkpoint."""
  experiment = f"osl_fullbody_unicycle_{model}_{reward}_{SOCKET}"
  run_dir = root / "logs" / "rsl_rl" / experiment / "2026-11-06_18-40-18"
  agent = osl_gait_library_runner_cfg(
    model, SOCKET, reward, experiment, variant="unicycle"
  )
  agent.actor.hidden_dims = HIDDEN
  agent.critic.hidden_dims = HIDDEN
  dump_yaml(run_dir / "params" / "agent.yaml", asdict(agent))

  generator = torch.Generator().manual_seed(STAGE1_ITER)
  actor = _unified_actor(model)
  obs = TensorDict({"critic": torch.zeros(1, ACTOR_DIM)}, batch_size=[1])
  critic = MuscleMimicMLPModel(
    obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=HIDDEN
  )
  optimizer = MuonAdam(itertools.chain(actor.parameters(), critic.parameters()), 4e-4)
  checkpoint = {
    "actor_state_dict": _random_state(actor, generator),
    "critic_state_dict": _random_state(critic, generator),
    "optimizer_state_dict": optimizer.state_dict(),
    "muscle_mimic_state": {
      "num_updates_done": STAGE1_ITER + 1,
      "return_mean": torch.tensor(RETURN_MEAN),
      "return_var": torch.tensor(RETURN_VAR),
      "return_count": torch.tensor(float(STAGE1_ENV_STEPS)),
    },
    "iter": STAGE1_ITER,
    "infos": {"env_state": {"common_step_counter": STAGE1_COUNTER}},
  }
  torch.save(checkpoint, run_dir / f"model_{STAGE1_ITER}.pt")
  log = root / "wandb" / f"run-20261106_184026-{WANDB_ID}" / "logs" / "debug.log"
  log.parent.mkdir(parents=True)
  log.write_text(f"config: {{'log_dir': '{run_dir.resolve()}'}}\n")
  return run_dir, checkpoint


def _stage2_cfg(
  model: Model,
  reward: Reward,
  warm_start: Path | None,
  *,
  socket: Socket = SOCKET,
  osl_critic_obs: str = "full",
  max_iterations: int = ITERATIONS,
  schedule_iterations: int = SCHEDULE,
) -> dict[str, Any]:
  """The real stage-2 runner config, shrunk, as ``train.py`` passes it."""
  cfg = asdict(osl_split_runner_cfg(model, socket, reward, "integration"))
  for key in ("actor", "critic", "osl_critic"):
    cfg[key]["hidden_dims"] = HIDDEN  # Stage 1's networks.
  cfg["osl_actor"]["hidden_dims"] = OSL_HIDDEN
  cfg["num_steps_per_env"] = NUM_STEPS
  cfg["max_iterations"] = max_iterations
  cfg["save_interval"] = SAVE_INTERVAL
  cfg["logger"] = "tensorboard"
  cfg["upload_model"] = False
  cfg["export_osl_onnx"] = False
  # No validation or stress validation, so no zero-shot pass either.
  cfg["validation"].update(interval=0, stress_interval=0)
  cfg["blend"]["enabled"] = False  # Its default ramp outlasts this freeze.
  cfg["algorithm"].update(
    num_mini_batches=2,
    schedule_iterations=schedule_iterations,
    human_actor_ramp_iterations=RAMP,
    osl_critic_obs=osl_critic_obs,
  )
  cfg["freeze"].update(human_actor_iterations=FREEZE, critic_warmup_iterations=WARMUP)
  cfg["channels"].update(anneal_start_iteration=-1, anneal_iterations=ANNEAL)
  cfg["channels"]["additive_term_weights"] = {"twist": TWIST_SPLIT}
  cfg["warm_start"]["path"] = "" if warm_start is None else str(warm_start)
  return cfg


##
# Checks.
##


def _changed(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]):
  """The keys whose tensors differ in any bit."""
  assert before.keys() == after.keys()
  return {key for key in before if not torch.equal(before[key], after[key])}


def _assert_same(a: Any, b: Any, where: str = "") -> None:
  """Recursive equality, tensors bit for bit."""
  if isinstance(a, torch.Tensor):
    assert isinstance(b, torch.Tensor) and torch.equal(a, b), where
  elif isinstance(a, Mapping):
    assert isinstance(b, Mapping) and set(a) == set(b), where
    for key in a:
      _assert_same(a[key], b[key], f"{where}/{key}")
  elif isinstance(a, (list, tuple)):
    assert isinstance(b, (list, tuple)) and len(a) == len(b), where
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
      _assert_same(x, y, f"{where}[{i}]")
  else:
    assert a == b, where


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
  """Equal up to float32 summation order (the split output layer has fewer rows)."""
  torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def _scalars(log_dir: Path) -> dict[str, dict[int, float]]:
  """``{tag: {step: value}}`` from a run's TensorBoard events."""
  events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  events.Reload()
  return {
    tag: {e.step: e.value for e in events.Scalars(tag)}
    for tag in events.Tags()["scalars"]
  }


def _completed_means(env: _FakeEnv) -> list[dict[str, float]]:
  """Per iteration, the mean return of the episodes completed so far, for the team
  and each agent: float32 sums and the last 100 episodes, as rsl_rl keeps
  ``Train/mean_reward``."""
  team_sum = torch.zeros(NUM_ENVS)
  agent_sum = torch.zeros(NUM_ENVS, 2)
  completed = {name: deque(maxlen=100) for name in ("team", *CHANNELS)}
  means: list[dict[str, float]] = []
  for t, (team, agents, dones) in enumerate(env.records):
    team_sum += team
    agent_sum += agents
    ids = dones.nonzero().squeeze(-1)
    completed["team"].extend(team_sum[ids].tolist())
    for i, agent in enumerate(CHANNELS):
      completed[agent].extend(agent_sum[ids, i].tolist())
    team_sum[ids] = 0.0
    agent_sum[ids] = 0.0
    if (t + 1) % NUM_STEPS == 0:
      means.append({name: statistics.mean(r) for name, r in completed.items()})
  return means


def _check_warm_start(
  runner: SplitOnPolicyRunner, env: _FakeEnv, stage1: dict, osl_critic_obs: str
) -> None:
  """The converted file is what the algorithm loaded."""
  alg = runner.alg
  layout = alg.layout
  unified = _unified_actor(cast(Model, runner.validation_cfg["model"]))
  unified.load_state_dict(stage1["actor_state_dict"], strict=True)
  obs = env.get_observations()
  human_rows, osl_rows = layout.rows("human"), layout.rows("osl")
  with torch.inference_mode():
    actions = runner.get_inference_policy()(obs)  # play.py's policy.
    full = unified(obs)
    features = alg.human_actor.mlp[:-1](alg.human_actor.get_latent(obs))
    teacher = alg.teacher_head(features)
    values = alg.human_critic(obs), alg.osl_critic(obs)
  _assert_close(actions[:, human_rows], full[:, human_rows])
  _assert_close(teacher, full[:, osl_rows])
  log_std = stage1["actor_state_dict"][LOG_STD]
  assert torch.equal(std_parameter(alg.human_actor), log_std[human_rows])

  # Both critics start as the stage-1 critic: the OSL critic's appended OSL inputs
  # (full+osl) have zero weights, so it values every state the same.
  critic = stage1["critic_state_dict"]
  _assert_same(alg.human_critic.state_dict(), critic)
  if osl_critic_obs == "full":
    _assert_same(alg.osl_critic.state_dict(), critic)
  else:
    assert alg.osl_critic.obs_groups == ["actor", "osl"]
    assert alg.osl_critic.obs_dim == ACTOR_DIM + OSL_DIM
  _assert_close(values[1], values[0])

  # Both agents carry the stage-1 return statistics, the count reset to three
  # iterations' samples so they can follow the channels.
  stats = alg.save()["reward_normalization"]
  torch.testing.assert_close(stats["mean"], torch.full((2,), RETURN_MEAN))
  torch.testing.assert_close(stats["var"], torch.full((2,), RETURN_VAR))
  torch.testing.assert_close(
    stats["count"], torch.full((2,), 3.0 * NUM_STEPS * NUM_ENVS)
  )


##
# Tests.
##


@pytest.mark.parametrize(
  ("model", "osl_critic_obs"), [("msk", "full"), ("robot", "full+osl")]
)
def test_stage2_end_to_end(tmp_path: Path, model: Model, osl_critic_obs: str) -> None:
  """A converted stage-1 run, trained, checkpointed and resumed by the real runner;
  both action layouts, and both OSL critic inputs."""
  torch.manual_seed(0)
  run_dir, stage1 = _stage1_run(tmp_path, model, "clf")
  ws_path = split_main([str(run_dir), "--out", str(tmp_path / "ws.pt")])
  env = _FakeEnv(model)
  log_dir = tmp_path / "stage2"
  cfg = _stage2_cfg(model, "clf", ws_path, osl_critic_obs=osl_critic_obs)
  runner = _Runner(env, copy.deepcopy(cfg), str(log_dir))
  alg = runner.alg

  # Stage 2 continues stage 1's iteration axis and step counter.
  assert runner.stage_start == runner.current_learning_iteration == STAGE_START
  assert env.common_step_counter == STAGE1_COUNTER
  assert runner.lineage is not None and runner.lineage["wandb_run_id"] == WANDB_ID
  assert not runner.reward_switch and runner.schedule_iterations == SCHEDULE
  _check_warm_start(runner, env, stage1, osl_critic_obs)
  # The reward term judged both ends of the channels' anneal before any training.
  channels = ChannelSchedule.from_train_cfg(cfg)
  term = env.reward_manager.mimic
  assert term.weights_checked == [channels.start, channels.end]
  assert term.weights_set == []

  start = _Snapshot.of(runner, STAGE_START - 1, -1)
  runner.learn(ITERATIONS)
  history = runner.history
  assert [s.it for s in history] == [STAGE_START + k for k in range(ITERATIONS)]
  assert [s.k for s in history] == list(range(ITERATIONS))
  assert env.common_step_counter == STAGE1_COUNTER + ITERATIONS * NUM_STEPS

  # The freeze curriculum, module by module, bit for bit.
  human_keys = set(start.states["human_actor"])
  for prev, snap in itertools.pairwise([start, *history]):
    k, states = snap.k, snap.states
    if k < FREEZE:  # Weights, std and observation statistics.
      assert _changed(start.states["human_actor"], states["human_actor"]) == set()
      assert snap.optimizer_states["human_actor"] == 0
    else:  # Everything learns but the std, which stays stage 1's.
      changed = _changed(prev.states["human_actor"], states["human_actor"])
      assert changed == human_keys - {LOG_STD}, k
      assert snap.optimizer_states["human_actor"] > 0
    assert torch.equal(
      states["human_actor"][LOG_STD], start.states["human_actor"][LOG_STD]
    )
    osl_changed = _changed(prev.states["osl_actor"], states["osl_actor"])
    if k < WARMUP:  # It acts, its statistics follow, its weights wait.
      assert osl_changed == set(NORMALIZER)
      assert snap.optimizer_states["osl_actor"] == 0
    else:
      assert osl_changed == set(states["osl_actor"]), k
    for critic in ("human_critic", "osl_critic"):  # From k = 0.
      assert _changed(prev.states[critic], states[critic]) == set(states[critic])
      assert snap.optimizer_states[critic] > 0
    assert _changed(start.states["teacher_head"], states["teacher_head"]) == set()
  assert std_parameter(alg.human_actor) not in alg.optimizers["human_actor"].state

  # Learning rates: the schedule at every k, over schedule_iterations (6), not the
  # run's 4 iterations.
  schedule = SplitSchedule.from_train_cfg(
    cfg, total_iterations=SCHEDULE, reward_switch=False
  )
  for snap in history:
    assert snap.lrs == schedule.lrs(snap.k)
  algorithm = cfg["algorithm"]
  osl_lr, peak = algorithm["osl_actor_lr"], algorithm["human_actor_peak_lr"]
  floor = algorithm["lr_final_ratio"]
  for snap in history:
    # The critics ramp over the warm-up and decay to their floor at the end.
    ramp = min((snap.k + 1) / WARMUP, 1.0)
    decay = floor + (1.0 - floor) * (1.0 - snap.k / SCHEDULE)
    for module in ("human_critic", "osl_critic"):
      rate = algorithm[f"{module}_lr"] * ramp * decay
      assert snap.lrs[module] == pytest.approx(rate), module
    expected = 0.0 if snap.k < WARMUP else osl_lr * decay
    assert snap.lrs["osl_actor"] == pytest.approx(expected)
  # The human: frozen, then its ramp to the peak, which it holds.
  human_lrs = [snap.lrs["human_actor"] for snap in history]
  assert human_lrs == pytest.approx([0.0, 0.0, peak / RAMP, peak])

  # Checkpoints at stage-local k % save_interval == 0 but the launch's first
  # iteration, and the last.
  steps = [snap.it for snap in history]
  saved = {s.it for s in history if s.k % SAVE_INTERVAL == 0 and s.k > 0}
  saved.add(steps[-1])
  assert {int(p.stem.split("_")[1]) for p in log_dir.glob("model_*.pt")} == saved
  final = log_dir / f"model_{steps[-1]}.pt"
  checkpoint = torch.load(final, map_location="cpu", weights_only=False)
  assert checkpoint["iter"] == steps[-1]
  assert checkpoint["stage_iteration"] == ITERATIONS - 1
  assert checkpoint["infos"]["split"] == {
    "stage_start": STAGE_START,
    "lineage": runner.lineage,
    "schedule_iterations": SCHEDULE,
    "reward_switch": False,
    # The whole curriculum, which a resumed run keeps.
    "schedule": asdict(schedule),
    "channel_schedule": {
      "start": channels.start,
      "end": channels.end,
      "anneal_start": FREEZE,
      "anneal_iterations": ANNEAL,
    },
    "additive_term_weights": {"twist": list(TWIST_SPLIT)},
  }
  assert checkpoint["infos"]["env_state"] == {
    "common_step_counter": env.common_step_counter
  }

  # The scalars, at the global iterations.
  logged = _scalars(log_dir)
  assert set(EXPECTED_TAGS) <= set(logged), set(EXPECTED_TAGS) - set(logged)
  assert logged["Stage2/iteration"] == {s.it: float(s.k) for s in history}
  frozen = [logged["Stage2/human_frozen"][it] for it in steps]
  assert frozen == [float(k < FREEZE) for k in range(ITERATIONS)]
  warming = [logged["Stage2/osl_actor_frozen"][it] for it in steps]
  assert warming == [float(k < WARMUP) for k in range(ITERATIONS)]
  assert set(logged["Stage2/reward_switch"].values()) == {0.0}
  assert set(logged["Stage2/blend_beta"].values()) == {1.0}
  # Each actor's losses only once it learns; the teacher's distance only while
  # the human trunk it reads is frozen.
  assert set(logged["Loss/human/surrogate"]) == set(steps[FREEZE:])
  assert set(logged["Loss/osl/surrogate"]) == set(steps[WARMUP:])
  assert set(logged["Loss/human/value"]) == set(logged["Loss/osl/value"]) == set(steps)
  assert set(logged["Train/osl/teacher_rms"]) == set(steps[:FREEZE])
  for snap in history:
    for module in MODULES:
      lr = logged[f"LR/{module}"][snap.it]
      assert lr == pytest.approx(snap.lrs[module], rel=1e-6, abs=1e-12)
    assert logged["Loss/learning_rate"][snap.it] == logged["LR/osl_actor"][snap.it]
  # The human samples at stage 1's std all stage long (rsl_rl's Policy/mean_std
  # is the human's); the OSL starts at its own initial std.
  human_rows = alg.layout.rows("human")
  human_std = float(stage1["actor_state_dict"][LOG_STD][human_rows].exp().mean())
  for tag in ("Policy/mean_std", "Policy/human/mean_std"):
    assert set(logged[tag]) == set(steps)
    assert list(logged[tag].values()) == pytest.approx([human_std] * ITERATIONS)
  assert logged["Policy/osl/std_knee"][steps[0]] == pytest.approx(0.2)
  # The channels hold the stage-1 weights through the freeze, then anneal.
  weights = [channels.weights(s.k) for s in history]
  assert term.weights_set == weights
  for snap, w in zip(history, weights, strict=True):
    for channel, group in itertools.product(CHANNELS, GROUPS):
      value = logged[f"Channels/{channel}/{group}"][snap.it]
      assert value == pytest.approx(w[channel][group])
      share = logged[f"Channels/{channel}/share_{group}"][snap.it]
      assert share == pytest.approx(_shares(w[channel])[group], rel=1e-6)
  joints = [logged["Channels/osl/osl_joints"][it] for it in steps]
  assert joints == pytest.approx([0.0, 0.0, 0.0, 5.0])
  # Env steps continue stage 1's count.
  samples = NUM_STEPS * NUM_ENVS
  assert logged["Train/total_env_steps"] == {
    s.it: float(STAGE1_ENV_STEPS + samples * (s.k + 1)) for s in history
  }
  # Train/mean_reward is the team (stage-1) reward; each agent's own beside it.
  means = _completed_means(env)
  for snap, mean in zip(history, means, strict=True):
    for tag, name in (
      ("Train/mean_reward", "team"),
      ("Train/human/mean_reward", "human"),
      ("Train/osl/mean_reward", "osl"),
    ):
      assert logged[tag][snap.it] == pytest.approx(mean[name], rel=1e-5), tag

  # Resume (train.py --agent.resume): a fresh runner without a warm start, whose
  # own config would give a 2-iteration schedule, loads the last checkpoint.
  resumed_dir = tmp_path / "resumed"
  resumed_cfg = _stage2_cfg(
    model,
    "clf",
    None,
    osl_critic_obs=osl_critic_obs,
    max_iterations=RESUMED,
    schedule_iterations=0,
  )
  resumed = _Runner(_FakeEnv(model, seed=1), resumed_cfg, str(resumed_dir))
  assert resumed.schedule_iterations == RESUMED
  infos = resumed.load(str(final))
  assert infos["split"]["stage_start"] == STAGE_START
  assert resumed.current_learning_iteration == steps[-1] + 1  # Not replayed.
  assert resumed.stage_start == STAGE_START
  assert resumed.lineage == runner.lineage
  assert resumed.schedule_iterations == SCHEDULE and not resumed.reward_switch
  assert resumed.env.unwrapped.common_step_counter == env.common_step_counter
  # Networks, optimizers (rates included), return statistics and k.
  _assert_same(resumed.alg.save(), alg.save())

  resumed.learn(RESUMED)
  more = resumed.history
  assert [s.it for s in more] == [steps[-1] + 1 + i for i in range(RESUMED)]
  assert [s.k for s in more] == [ITERATIONS + i for i in range(RESUMED)]
  for snap in more:
    assert snap.lrs == schedule.lrs(snap.k)
  resumed_logged = _scalars(resumed_dir)
  assert resumed_logged["Stage2/iteration"] == {s.it: float(s.k) for s in more}
  assert resumed_logged["Train/total_env_steps"] == {
    s.it: float(STAGE1_ENV_STEPS + samples * (s.k + 1)) for s in more
  }
  # Still learning: the human after its unfreeze, every module but its std.
  last = more[-1].states["human_actor"]
  assert _changed(history[-1].states["human_actor"], last) == human_keys - {LOG_STD}

  # Play (play.py): a runner from the task's config, whose OSL critic reads the
  # default "full" input whatever the run trained with, loads the two actors.
  played = _Runner(_FakeEnv(model, seed=2), _stage2_cfg(model, "clf", None), None)
  played.load(str(final), load_cfg={"actor": True}, strict=True, map_location="cpu")
  assert played.stage_start == STAGE_START and played.lineage == runner.lineage
  obs = env.get_observations()
  with torch.inference_mode():
    actions = played.get_inference_policy()(obs)
    assert torch.equal(actions, alg.get_policy()(obs))


@pytest.mark.parametrize(
  ("model", "socket", "reward"),
  [
    ("msk", SOCKET, "clf"),  # The reward switch: allowed.
    ("msk", "rigid", "mimic"),
    ("robot", SOCKET, "mimic"),
  ],
)
def test_stage2_task_must_match_the_stage1_run(
  tmp_path: Path, model: Model, socket: Socket, reward: Reward
) -> None:
  """From a converted MSK MimicReward run on the compliant socket: only the reward
  may switch. A switched reward starts both critics fresh, on the stage-1 critic's
  observation statistics, with fresh return statistics and a longer warm-up."""
  torch.manual_seed(0)
  run_dir, stage1 = _stage1_run(tmp_path, "msk", "mimic")
  ws_path = split_main([str(run_dir), "--out", str(tmp_path / "ws.pt")])
  cfg = _stage2_cfg(model, reward, ws_path, socket=socket)
  if (model, socket) != ("msk", SOCKET):
    with pytest.raises(ValueError, match="cannot switch between stages"):
      _Runner(_FakeEnv(model), cfg, None)
    return

  runner = _Runner(_FakeEnv(model), copy.deepcopy(cfg), None)
  warmup = cfg["freeze"]["reward_switch_critic_warmup_iterations"]
  assert runner.reward_switch and runner.schedule.critic_warmup == warmup
  alg = runner.alg
  log_std = stage1["actor_state_dict"][LOG_STD]
  assert torch.equal(std_parameter(alg.human_actor), log_std[alg.layout.rows("human")])
  critic = stage1["critic_state_dict"]
  for module in (alg.human_critic, alg.osl_critic):
    state = module.state_dict()
    assert not torch.equal(state["mlp.0.weight"], critic["mlp.0.weight"])
    assert torch.equal(state["obs_normalizer.mean"], critic["obs_normalizer.mean"])
    assert torch.equal(state["obs_normalizer.var"], critic["obs_normalizer.var"])
    assert float(state["obs_normalizer.count"]) == NUM_STEPS * NUM_ENVS
  stats = alg.save()["reward_normalization"]
  assert stats["mean"].tolist() == [0.0, 0.0]
  assert stats["var"].tolist() == [1.0, 1.0]
  torch.testing.assert_close(stats["count"], torch.full((2,), RETURN_COUNT_FRESH))


def test_stage2_builds_the_stage1_architecture(tmp_path: Path) -> None:
  """Stage 2 builds its networks from its own config. A stage-1 run trained with
  another activation (which no state-dict key records) converts and verifies
  against its own architecture, and its weights would load strictly into stage
  2's networks and compute something else: refused, naming the overrides, for
  every network that starts from stage 1's (the critics only without a reward
  switch, which starts them fresh). With the overrides the split is exact."""
  torch.manual_seed(0)
  run_dir, stage1 = _stage1_run(tmp_path, "msk", "clf")
  agent_yaml = run_dir / "params" / "agent.yaml"
  text = agent_yaml.read_text()
  assert text.count("activation: swish") == 2  # The actor and the critic.
  agent_yaml.write_text(text.replace("activation: swish", "activation: elu"))
  ws_path = split_main([str(run_dir), "--out", str(tmp_path / "ws.pt")])

  cfg = _stage2_cfg("msk", "clf", ws_path)
  with pytest.raises(ValueError, match="architecture") as refused:
    _Runner(_FakeEnv("msk"), copy.deepcopy(cfg), None)
  message = str(refused.value)
  for module in ("actor", "critic", "osl-critic"):
    assert f"--agent.{module}.activation elu" in message, message
  switched = _stage2_cfg("msk", "mimic", ws_path)
  with pytest.raises(ValueError, match="architecture") as refused:
    _Runner(_FakeEnv("msk"), copy.deepcopy(switched), None)
  message = str(refused.value)
  assert "--agent.actor.activation elu" in message
  assert "--agent.critic." not in message and "--agent.osl-critic." not in message

  switched["actor"]["activation"] = "elu"
  assert _Runner(_FakeEnv("msk"), switched, None).reward_switch
  for key in ("actor", "critic", "osl_critic"):
    cfg[key]["activation"] = "elu"
  env = _FakeEnv("msk")
  runner = _Runner(env, cfg, None)
  unified = _unified_actor("msk", stage1["actor_state_dict"], activation="elu")
  obs = env.get_observations()
  rows = runner.alg.layout.rows("human")
  with torch.inference_mode():
    _assert_close(runner.get_inference_policy()(obs)[:, rows], unified(obs)[:, rows])
