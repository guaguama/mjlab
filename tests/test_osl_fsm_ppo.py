"""Contracts for stage 1.5's human-only PPO (``tasks/osl_fsm/rl``).

CPU only: tiny musclemimic MLPs on fake observations and a fake env that exposes
just an action manager, as in ``test_osl_split_ppo.py``. The centre is the
equivalence with stage 2: ``HumanOnlyPPO`` must be ``TwoAgentPPO``'s human path, so
that a stage-1.5 run differs from a stage-2 one only in what drives the OSL. Around
it: the held std, the schedule's rates, the warm start, the freeze, checkpoints
(whole, actors only, and the refusal of other stages'), the inference policy, the
refusal of OSL settings that would do nothing, and a clean import.
"""

from __future__ import annotations

import copy
import importlib
import io
import os
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict
from torch import nn

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model
from mjlab.envs.mdp.actions import TendonEffortAction
from mjlab.tasks.osl_fsm.rl.policy import HumanOnlyInferencePolicy
from mjlab.tasks.osl_fsm.rl.ppo import (
  ALGORITHM,
  HUMAN_MODULES,
  OSL_KNOB_DEFAULTS,
  OSL_ONLY_KNOBS,
  HumanOnlyPPO,
)
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel, RunningMeanStd
from mjlab.tasks.osl_split.config.rl_cfg import SplitPpoCfg, osl_split_runner_cfg
from mjlab.tasks.osl_split.rl.expected import action_bounds, expected_clipped_gaussian
from mjlab.tasks.osl_split.rl.policy import distribution_std, std_parameter
from mjlab.tasks.osl_split.rl.ppo import TwoAgentPPO
from mjlab.tasks.osl_split.rl.schedules import SplitSchedule

NUM_ENVS = 6
"""Below torch's float32 vector width (8 lanes), where it sums a column of a pair
and a lone column in the same order (see the equivalence test)."""
NUM_STEPS = 4
NUM_MINI_BATCHES = 4
"""6 x 4 / 4: minibatches of 6, below the vector width too."""
ACTOR_DIM = 12
OSL_OBS_DIM = 6
HUMAN_DIM = 5
HIDDEN = (16, 16)

##
# Fakes and helpers.
##


class _FakeMuscles(TendonEffortAction):
  """A muscle action term without an env: only its type and size matter."""

  def __init__(self, dim: int) -> None:
    self._action_dim = dim


class _FakeActionManager:
  def __init__(self, terms: dict[str, Any]) -> None:
    self._terms = terms

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms)

  def get_term(self, name: str) -> Any:
    return self._terms[name]


def _action_manager(model: Model) -> Any:
  """The stage-1 term order, which stage 1.5 keeps with the FSM in ``osl_motors``:
  MSK ``osl_motors, muscles``; robot ``servos, osl_motors``."""
  if model == "msk":
    terms = {
      "osl_motors": SimpleNamespace(action_dim=2),
      "muscles": _FakeMuscles(HUMAN_DIM),
    }
  else:
    terms = {
      "servos": SimpleNamespace(action_dim=HUMAN_DIM),
      "osl_motors": SimpleNamespace(action_dim=2),
    }
  return _FakeActionManager(terms)


def _env(model: Model = "msk") -> SimpleNamespace:
  return SimpleNamespace(
    num_envs=NUM_ENVS,
    num_actions=HUMAN_DIM + 2,
    unwrapped=SimpleNamespace(action_manager=_action_manager(model)),
  )


def _obs(n: int = NUM_ENVS) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(n, ACTOR_DIM),
      "osl": 3.0 * torch.randn(n, OSL_OBS_DIM) + 1.0,
    },
    batch_size=[n],
  )


def _split_cfg(model: Model = "msk", **algorithm: Any) -> dict[str, Any]:
  """Stage 2's real runner config, shrunk."""
  cfg = asdict(osl_split_runner_cfg(model, "compliant", "clf", "test"))
  for key in ("actor", "critic", "osl_critic"):
    cfg[key]["hidden_dims"] = HIDDEN
  cfg["osl_actor"]["hidden_dims"] = (8, 8)
  cfg["num_steps_per_env"] = NUM_STEPS
  cfg["algorithm"]["num_mini_batches"] = NUM_MINI_BATCHES
  cfg["algorithm"]["num_learning_epochs"] = 2  # Two permutations per update.
  cfg["algorithm"].update(algorithm)
  cfg["multi_gpu"] = None
  return cfg


def _fsm_cfg(model: Model = "msk", **algorithm: Any) -> dict[str, Any]:
  """Stage 1.5's runner config as the spec derives it from stage 2's: the
  human-only algorithm, no freeze, no blend and the human's observation sets."""
  cfg = _split_cfg(model, **algorithm)
  cfg["algorithm"]["class_name"] = "mjlab.tasks.osl_fsm.rl.ppo:HumanOnlyPPO"
  cfg["obs_groups"] = {"actor": ("actor",), "critic": ("actor",)}
  cfg["freeze"]["human_actor_iterations"] = 0
  cfg["blend"]["enabled"] = False
  return cfg


def _alg(model: Model = "msk", **algorithm: Any) -> HumanOnlyPPO:
  return HumanOnlyPPO.construct_algorithm(
    _obs(), _env(model), _fsm_cfg(model, **algorithm), "cpu"
  )


def _schedule(
  human_freeze: int = 0, critic_warmup: int = 2, **overrides: Any
) -> SplitSchedule:
  """Stage 2's schedule shape, with rates large enough that every update visibly
  moves both networks."""
  schedule = SplitSchedule(
    total_iterations=100,
    human_freeze=human_freeze,
    critic_warmup=critic_warmup,
    human_critic_lr=3e-3,
    osl_critic_lr=3e-4,
    critic_ramp=critic_warmup,
    osl_actor_lr=4e-4,
    human_peak_lr=1e-3,
    human_ramp=2,
    lr_final_ratio=0.1,
    human_lr_final_ratio=1.0,
    blend=False,
    blend_ramp=0,
  )
  return replace(schedule, **overrides)


def _step(alg: HumanOnlyPPO, rewards: torch.Tensor | None = None) -> torch.Tensor:
  """One env step with random rewards, dones and time-outs, as the runner does it
  (inside inference mode)."""
  n = alg.storage.num_envs
  with torch.inference_mode():
    obs = _obs(n)
    actions = alg.act(obs)
    alg.process_env_step(
      obs,
      torch.rand(n, 2) if rewards is None else rewards,
      (torch.rand(n) < 0.25).long(),
      {"time_outs": torch.rand(n) < 0.1},
    )
  return actions


def _rollout(alg: HumanOnlyPPO) -> None:
  while not alg.storage.is_full:
    _step(alg)
  with torch.inference_mode():
    alg.compute_returns(_obs(alg.storage.num_envs))


def _normalizer(model: MuscleMimicMLPModel) -> RunningMeanStd:
  normalizer = model.obs_normalizer
  assert isinstance(normalizer, RunningMeanStd)
  return normalizer


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
  return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _assert_same(a: Any, b: Any) -> None:
  """Bit-identical nested state (tensors, dicts, lists, scalars)."""
  if isinstance(a, torch.Tensor):
    assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
  elif isinstance(a, dict):
    assert isinstance(b, dict) and a.keys() == b.keys()
    for key in a:
      _assert_same(a[key], b[key])
  elif isinstance(a, (list, tuple)):
    assert isinstance(b, (list, tuple)) and len(a) == len(b)
    for x, y in zip(a, b, strict=True):
      _assert_same(x, y)
  else:
    assert a == b


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
  return any(not torch.equal(before[k], after[k]) for k in before)


def _warm_start(layout: Any) -> SimpleNamespace:
  """A synthetic warm start in the converter's format, cut from a random unified
  actor and critic (random weights and observation statistics)."""
  obs = _obs()
  unified = MuscleMimicMLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    layout.total_dim,
    hidden_dims=HIDDEN,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.3,
      "std_type": "log",
    },
  )
  critic = MuscleMimicMLPModel(
    obs, {"critic": ["actor"]}, "critic", 1, hidden_dims=HIDDEN
  )
  with torch.no_grad():
    for model in (unified, critic):
      for p in model.parameters():
        p.normal_(0.0, 0.3)
      norm = _normalizer(model)
      norm.mean.normal_()
      norm.var.uniform_(0.5, 2.0)
      norm.count.fill_(2.0e8)
  last = len(unified.mlp) - 1
  sliced = {f"mlp.{last}.weight", f"mlp.{last}.bias", "distribution.log_std_param"}
  human = layout.index("human")
  osl = layout.index("osl")
  state = _state(unified)
  return SimpleNamespace(
    human_actor={
      k: v.index_select(0, human) if k in sliced else v for k, v in state.items()
    },
    human_critic=_state(critic),
    teacher_head={
      "weight": state[f"mlp.{last}.weight"].index_select(0, osl),
      "bias": state[f"mlp.{last}.bias"].index_select(0, osl),
      "log_std": state["distribution.log_std_param"].index_select(0, osl),
    },
    return_stats={
      "mean": torch.tensor(0.3),
      "var": torch.tensor(2.5),
      "count": torch.tensor(1.0e9),
    },
    lineage={},
  )


def _apply(
  alg: HumanOnlyPPO | TwoAgentPPO, ws: SimpleNamespace, reward_switch: bool = False
) -> None:
  alg.apply_warm_start(
    cast(Any, ws),
    reward_switch=reward_switch,
    num_envs=alg.storage.num_envs,
    num_steps=NUM_STEPS,
  )


def _checkpoint(payload: dict) -> dict:
  """``payload`` through ``torch.save`` and back, as a checkpoint file."""
  buffer = io.BytesIO()
  torch.save(payload, buffer)
  buffer.seek(0)
  return torch.load(buffer, weights_only=False)


##
# The equivalence with stage 2's human path.
##


@pytest.mark.parametrize(
  ("human_freeze", "human_learn_std"),
  [
    (0, False),  # Stage 1.5's recipe.
    (1, True),  # A frozen first iteration, then a learned std.
  ],
)
def test_matches_two_agent_ppo_human_path(
  human_freeze: int, human_learn_std: bool
) -> None:
  """From one warm start, with the same seeds, observations and human rewards, and
  stage 2's schedule without the blend, HumanOnlyPPO and TwoAgentPPO's human path
  take the same actions and leave the same actor, critic, optimizer states and
  return statistics, bit for bit, update after update. HumanOnlyPPO also gets a
  different OSL reward column, which it must ignore.

  Bit for bit needs torch to sum a lone column (HumanOnlyPPO's ``(N, 1)``) and a
  column of a pair (stage 2's ``(N, 2)``) in the same order, which it does below
  its float32 vector width of 8: hence 6 envs and minibatches of 6. From 8 on, the
  two dim-0 means over that column (the reward normalization's return mean and the
  minibatch advantage mean) may differ in the last bit, and Muon's bfloat16
  Newton-Schulz carries that into the weights at about the rate times 2^-8 per step
  (measured over these three updates: 5e-8 at 16 envs, 5e-5 at 64), so no
  tolerance would hold across machines; every other operation is identical."""
  torch.manual_seed(0)
  split_cfg = _split_cfg(human_learn_std=human_learn_std)
  two = TwoAgentPPO.construct_algorithm(_obs(), _env(), split_cfg, "cpu")
  one = _alg(human_learn_std=human_learn_std)
  ws = _warm_start(two.layout)
  schedule = _schedule(human_freeze=human_freeze)
  for alg in (two, one):
    _apply(alg, ws)
    alg.set_schedule(schedule)
    alg.train_mode()
  rows_h, rows_o = one.layout.rows("human"), one.layout.rows("osl")
  human_terms = [f"/{term}" for term in one.layout.human_terms()]
  start = _state(one.human_actor)

  for k in range(3):
    two.set_stage_iteration(k)
    one.set_stage_iteration(k)
    assert one.human_actor_frozen == (k < human_freeze) == two.human_actor_frozen
    assert one.lrs == {name: two.lrs[name] for name in HUMAN_MODULES}
    for t in range(NUM_STEPS):
      obs = _obs()
      rewards = torch.rand(NUM_ENVS, 2)
      own = torch.stack((rewards[:, 0], torch.randn(NUM_ENVS)), dim=-1)
      dones = (torch.rand(NUM_ENVS) < 0.25).long()
      extras = {"time_outs": torch.rand(NUM_ENVS) < 0.1}
      with torch.inference_mode():
        torch.manual_seed(100 * k + t)
        theirs = two.act(obs)
        torch.manual_seed(100 * k + t)
        ours = one.act(obs)
        two.process_env_step(obs, rewards, dones, extras)
        one.process_env_step(obs, own, dones, extras)
      assert torch.equal(ours[:, rows_h], theirs[:, rows_h])
      assert torch.equal(ours[:, rows_o], torch.zeros(NUM_ENVS, len(rows_o)))
    with torch.inference_mode():
      last = _obs()
      two.compute_returns(last)
      one.compute_returns(last)
    for name in ("log_probs", "values", "rewards", "returns", "advantages"):
      _assert_same(getattr(one.storage, name), getattr(two.storage, name)[..., :1])
    _assert_same(one.storage.actions["human"], two.storage.actions["human"])

    torch.manual_seed(1000 + k)
    theirs_losses = two.update()
    torch.manual_seed(1000 + k)
    ours_losses = one.update()
    human_losses = {
      key: value for key, value in theirs_losses.items() if key.startswith("human/")
    }
    _assert_same(ours_losses, human_losses)
    assert ("human/surrogate" in ours_losses) == (k >= human_freeze)
    for name in HUMAN_MODULES:
      _assert_same(_state(one.models[name]), _state(two.models[name]))
      _assert_same(one.optimizers[name].state_dict(), two.optimizers[name].state_dict())
    ours_stats, theirs_stats = one.save(), two.save()
    _assert_same(
      ours_stats["reward_normalization"],
      {key: v[:1] for key, v in theirs_stats["reward_normalization"].items()},
    )
    _assert_same(one._return, two._return[:, :1])
    _assert_same(
      one.policy_stats,
      {
        key: value
        for key, value in two.policy_stats.items()
        if key.endswith(tuple(human_terms))
      },
    )
    _assert_same(
      one.critic_stats,
      {"human/explained_variance": two.critic_stats["human/explained_variance"]},
    )
  assert _changed(start, _state(one.human_actor))  # Not a vacuous comparison.


##
# The std, the schedule and the freeze.
##


def test_std_is_held_bit_exact() -> None:
  """With human_learn_std False (the user's decision) the human keeps its warm
  start's std while its mean learns, and the std never gets optimizer state."""
  torch.manual_seed(0)
  alg = _alg()
  ws = _warm_start(alg.layout)
  _apply(alg, ws)
  alg.set_schedule(_schedule())
  std = std_parameter(alg.human_actor)
  assert not std.requires_grad
  before = _state(alg.human_actor)
  for k in range(2):
    alg.set_stage_iteration(k)
    _rollout(alg)
    assert "human/surrogate" in alg.update()
  assert torch.equal(std.detach(), ws.human_actor["distribution.log_std_param"])
  assert std not in alg.optimizers["human_actor"].state
  after = _state(alg.human_actor)
  assert _changed(
    {k: v for k, v in before.items() if k.startswith("mlp.")},
    {k: v for k, v in after.items() if k.startswith("mlp.")},
  )


def test_learning_rates_follow_the_split_schedule() -> None:
  """Stage 1.5's schedule is stage 2's with no freeze: the human actor ramps from
  iteration 0 over 50 iterations to 3e-5 and holds it; the critic ramps over the
  10-iteration warm-up and decays to its floor, the human actor's peak."""
  alg = _alg()
  cfg = _fsm_cfg()
  schedule = SplitSchedule.from_train_cfg(
    cfg, total_iterations=1000, reward_switch=False
  )
  alg.set_schedule(schedule)
  for k in (0, 1, 9, 10, 49, 50, 500, 900, 1000, 5000):
    alg.set_stage_iteration(k)
    assert alg.lrs == {name: schedule.lr(name, k) for name in HUMAN_MODULES}
    assert alg.learning_rate == schedule.lr("human_actor", k)
    assert not alg.human_actor_frozen and alg.osl_actor_frozen
  alg.set_stage_iteration(0)
  assert alg.lrs["human_actor"] == pytest.approx(3e-5 / 50)  # It learns at k = 0.
  assert alg.lrs["human_critic"] == pytest.approx(1e-4 / 10)
  alg.set_stage_iteration(49)
  assert alg.lrs["human_actor"] == pytest.approx(3e-5)
  alg.set_stage_iteration(900)  # 1e-4 x 0.19, floored at the actor's peak.
  assert alg.lrs["human_critic"] == pytest.approx(3e-5)

  # A freeze is supported: frozen, no rate and no gradient, until it ends.
  alg.set_schedule(replace(schedule, human_freeze=5))
  alg.set_stage_iteration(4)
  assert alg.human_actor_frozen and alg.lrs["human_actor"] == 0.0
  assert not any(p.requires_grad for p in alg.human_actor.parameters())
  alg.set_stage_iteration(5)
  assert not alg.human_actor_frozen and alg.lrs["human_actor"] > 0.0
  with pytest.raises(ValueError, match="blend"):
    alg.set_schedule(replace(schedule, blend=True, blend_ramp=1, human_freeze=20))


def test_frozen_actor_takes_no_part() -> None:
  torch.manual_seed(0)
  alg = _alg()
  _apply(alg, _warm_start(alg.layout))
  alg.set_schedule(_schedule())  # Learn once, so the actor has moments to keep.
  _rollout(alg)
  alg.update()
  alg.set_schedule(_schedule(human_freeze=10))
  alg.set_stage_iteration(1)
  actor = _state(alg.human_actor)
  actor_opt = copy.deepcopy(alg.optimizers["human_actor"].state_dict())
  critic = _state(alg.human_critic)
  _rollout(alg)
  losses = alg.update()

  assert set(losses) == {"human/value"}
  # Weights, std and observation statistics, and the optimizer's moments.
  _assert_same(_state(alg.human_actor), actor)
  _assert_same(alg.optimizers["human_actor"].state_dict(), actor_opt)
  assert _changed(critic, _state(alg.human_critic))
  assert alg.lrs["human_actor"] == 0.0
  alg.storage.clear()
  with pytest.raises(RuntimeError, match="full rollout"):
    alg.update()


##
# Construction, the warm start and the rollout.
##


def test_construct_algorithm_contract() -> None:
  alg = _alg()
  assert {name: list(m.obs_groups) for name, m in alg.models.items()} == {
    "human_actor": ["actor"],
    "human_critic": ["actor"],
  }
  # The osl group, which no network reads, is not stored.
  assert alg.storage.groups == ("actor",) and alg.storage.agents == ("human",)
  assert not std_parameter(alg.human_actor).requires_grad
  # Until a schedule is set nothing learns.
  assert alg.human_actor_frozen and set(alg.lrs.values()) == {0.0}
  with pytest.raises(RuntimeError, match="schedule"):
    alg.update()
  with pytest.raises(RuntimeError, match="set_schedule"):
    alg.set_stage_iteration(0)
  # What stage 2's runner reads of an OSL actor.
  assert alg.osl_actor_frozen and alg.blend_beta == 1.0
  assert alg.osl_std.numel() == 0 and alg.last_teacher_rms is None
  assert alg.skipped_steps == {"human_actor": 0, "human_critic": 0}
  # Stage 2's observation sets are checked by rsl_rl and build nothing.
  cfg = _split_cfg()
  cfg["algorithm"]["class_name"] = "mjlab.tasks.osl_fsm.rl.ppo:HumanOnlyPPO"
  wide = HumanOnlyPPO.construct_algorithm(_obs(), _env(), cfg, "cpu")
  assert wide.storage.groups == ("actor",)

  with pytest.raises(ValueError, match="Unknown HumanOnlyPPO settings"):
    _alg(not_a_setting=1)
  with pytest.raises(ValueError, match="std_floor = 0.03"):
    _alg(std_floor=0.03)  # Stage 1's knobs are refused as in stage 2.
  cfg = _fsm_cfg()
  cfg["multi_gpu"] = {"world_size": 2}
  with pytest.raises(NotImplementedError):
    HumanOnlyPPO.construct_algorithm(_obs(), _env(), cfg, "cpu")


def test_osl_knobs_are_refused() -> None:
  """Stage 2's OSL settings reach the stage-1.5 config through SplitPpoCfg; with no
  OSL actor or critic any value but the default would silently do nothing."""
  assert set(OSL_KNOB_DEFAULTS) == set(OSL_ONLY_KNOBS)
  defaults = asdict(SplitPpoCfg())
  assert OSL_KNOB_DEFAULTS == {key: defaults[key] for key in OSL_ONLY_KNOBS}
  _alg()  # The defaults build.
  changed = {
    "osl_critic_lr": 1e-3,
    "osl_actor_lr": 1e-3,
    "osl_clip_param": 0.3,
    "osl_entropy_coef": 0.01,
    "osl_value_loss_coef": 0.5,
    "osl_critic_obs": "full+osl",
    "osl_critic_init": "fresh",
  }
  assert set(changed) == set(OSL_ONLY_KNOBS)
  for key, value in changed.items():
    cfg = _fsm_cfg()
    cfg["algorithm"][key] = value
    with pytest.raises(ValueError, match=f"{key} = ") as refused:
      HumanOnlyPPO.construct_algorithm(_obs(), _env(), cfg, "cpu")
    assert "stage 1.5 does not use" in str(refused.value)


@pytest.mark.parametrize("reward_switch", [False, True])
def test_warm_start(reward_switch: bool) -> None:
  torch.manual_seed(0)
  alg = _alg()
  fresh = _state(alg.human_critic)
  ws = _warm_start(alg.layout)
  _apply(alg, ws, reward_switch)

  _assert_same(alg.human_actor.state_dict(), ws.human_actor)  # Always exact.
  stats = alg.save()["reward_normalization"]
  one_iteration = NUM_ENVS * NUM_STEPS
  if not reward_switch:
    _assert_same(alg.human_critic.state_dict(), ws.human_critic)
    torch.testing.assert_close(stats["mean"], torch.tensor([0.3]))
    torch.testing.assert_close(stats["var"], torch.tensor([2.5]))
    torch.testing.assert_close(stats["count"], torch.tensor([3.0 * one_iteration]))
    return
  # Fresh weights on the stage-1 observation statistics; fresh return statistics.
  after = _state(alg.human_critic)
  for key, value in fresh.items():
    if key.startswith("obs_normalizer."):
      continue
    assert torch.equal(after[key], value), key
  for key in ("mean", "var"):
    assert torch.equal(
      after[f"obs_normalizer.{key}"], ws.human_critic[f"obs_normalizer.{key}"]
    )
  assert after["obs_normalizer.count"].item() == one_iteration
  assert stats["mean"].tolist() == [0.0] and stats["var"].tolist() == [1.0]
  torch.testing.assert_close(stats["count"], torch.tensor([1e-4]))


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_act_and_rewards(model: Model) -> None:
  """The env action carries the human's own sample in its rows and zeros in the OSL
  rows (0:2 on MSK, 5:7 on the robot); the human reward is the agents' column 0 or
  the human's own (B, 1), never the team's (B,)."""
  rewards = torch.rand(NUM_ENVS, 2)
  torch.manual_seed(0)
  alg = _alg(model)
  alg.set_schedule(_schedule())
  rows_h, rows_o = alg.layout.rows("human"), alg.layout.rows("osl")
  assert rows_o == ([0, 1] if model == "msk" else [HUMAN_DIM, HUMAN_DIM + 1])
  actions = _step(alg, rewards)
  assert actions.shape == (NUM_ENVS, HUMAN_DIM + 2)
  assert torch.equal(actions[:, rows_h], alg.storage.actions["human"][0])
  assert torch.equal(actions[:, rows_o], torch.zeros(NUM_ENVS, 2))
  assert alg.storage.values[0].shape == (NUM_ENVS, 1)

  # The human's column alone stores the same as the pair.
  torch.manual_seed(0)
  same = _alg(model)
  same.set_schedule(_schedule())
  _step(same, rewards[:, :1])
  _assert_same(same.storage.rewards[0], alg.storage.rewards[0])
  with torch.inference_mode():
    obs = _obs()
    dones = torch.zeros(NUM_ENVS)
    same.act(obs)
    with pytest.raises(ValueError, match=r"\(6, 1\)"):
      same.process_env_step(obs, torch.rand(NUM_ENVS), dones, {})
    same.process_env_step(obs, rewards, dones, {})  # The step still completes.
    with pytest.raises(RuntimeError, match="act"):
      same.process_env_step(obs, rewards, dones, {})


##
# Checkpoints.
##


def test_save_load_round_trip() -> None:
  torch.manual_seed(0)
  # Trained with other Muon settings than the resuming run's.
  alg = _alg(muon_beta=0.8, muon_ns_steps=3, muon_ns_dtype="float32")
  _apply(alg, _warm_start(alg.layout))
  alg.set_schedule(_schedule())
  alg.set_stage_iteration(3)
  _rollout(alg)
  alg.update()
  loaded = _checkpoint(alg.save())
  assert loaded["algorithm"] == ALGORITHM

  other = _alg()
  other.load(loaded)
  assert other.stage_iteration == 3
  for name in HUMAN_MODULES:
    _assert_same(other.models[name].state_dict(), alg.models[name].state_dict())
    _assert_same(
      other.optimizers[name].state_dict()["state"],
      alg.optimizers[name].state_dict()["state"],
    )
    # The resumed run's Muon settings win over the checkpoint's.
    assert other.optimizers[name].ns_dtype is torch.bfloat16
    for group in other.optimizers[name].param_groups:
      assert (group["beta"], group["nesterov"], group["ns_steps"]) == (0.95, True, 5)
  _assert_same(other.save()["reward_normalization"], alg.save()["reward_normalization"])
  # A resumed algorithm with a schedule re-applies it at the loaded iteration.
  other.set_schedule(_schedule(human_freeze=5))
  assert other.human_actor_frozen and other.lrs["human_actor"] == 0.0

  small_cfg = _fsm_cfg()
  small_cfg["critic"]["hidden_dims"] = (8,)
  small = HumanOnlyPPO.construct_algorithm(_obs(), _env(), small_cfg, "cpu")
  with pytest.raises(ValueError, match="--agent.critic.hidden-dims"):
    small.load(loaded)
  with pytest.raises(TypeError, match="strict"):  # rsl_rl's (loaded, load_cfg, ...).
    other.load(loaded, cast(Any, {"actor": True}))
  incomplete = {k: v for k, v in loaded.items() if k != "optimizer_state_dicts"}
  with pytest.raises(ValueError, match="lacks"):
    other.load(incomplete)


def test_actors_only_load() -> None:
  """play.py loads the actor alone: the critic, optimizers, return statistics and
  stage iteration keep their values."""
  torch.manual_seed(0)
  trained = _alg()
  _apply(trained, _warm_start(trained.layout))
  trained.set_schedule(_schedule())
  trained.set_stage_iteration(3)
  _rollout(trained)
  trained.update()
  loaded = _checkpoint(trained.save())

  played = _alg()
  played.set_schedule(_schedule(human_freeze=5))
  critic = _state(played.human_critic)
  optimizers = copy.deepcopy({k: o.state_dict() for k, o in played.optimizers.items()})
  stats = played.save()["reward_normalization"]
  played.load(
    {k: v for k, v in loaded.items() if k != "human_critic_state_dict"},
    actors_only=True,
  )

  _assert_same(played.human_actor.state_dict(), loaded["human_actor_state_dict"])
  _assert_same(_state(played.human_critic), critic)
  _assert_same({k: o.state_dict() for k, o in played.optimizers.items()}, optimizers)
  _assert_same(played.save()["reward_normalization"], stats)
  assert played.stage_iteration == 0 and played.human_actor_frozen
  obs = _obs()
  with torch.inference_mode():
    torch.testing.assert_close(played.get_policy()(obs), trained.get_policy()(obs))


def test_load_refuses_other_stages_checkpoints() -> None:
  """A stage-2 checkpoint (its human learned beside an OSL actor), a stage-1
  checkpoint and a warm-start file are not stage-1.5 checkpoints, whole or actors
  only."""
  torch.manual_seed(0)
  alg = _alg()
  stage2 = TwoAgentPPO.construct_algorithm(_obs(), _env(), _split_cfg(), "cpu")
  refused = {
    "stage-2": _checkpoint(stage2.save()),
    "stage-1 checkpoint": {
      "actor_state_dict": {},
      "critic_state_dict": {},
      "iter": 610,
    },
    "warm-start file": {"format": "mjlab.osl_split.warm_start/1"},
  }
  for hint, payload in refused.items():
    for actors_only in (False, True):
      with pytest.raises(ValueError, match=hint) as err:
        alg.load(payload, actors_only=actors_only)
      assert "stage-1.5" in str(err.value)
  with pytest.raises(ValueError, match="'two_agent'"):
    alg.load({**alg.save(), "algorithm": "two_agent"})


##
# Inference.
##


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_inference_policy(model: Model) -> None:
  """Play and validation run the human's mean or noise-averaged action with zeros
  in the OSL rows, without moving the training statistics."""
  torch.manual_seed(0)
  alg = _alg(model)
  _apply(alg, _warm_start(alg.layout))
  alg.set_schedule(_schedule())
  _rollout(alg)  # The actor's statistics move away from the warm start's.
  stats = _state(_normalizer(alg.human_actor))
  manager = _action_manager(model)
  rows_h, rows_o = alg.layout.rows("human"), alg.layout.rows("osl")
  obs = _obs()
  with torch.inference_mode():
    mean_policy = alg.get_policy()
    mean = mean_policy(obs)
    expected = alg.inference_policy("expected", manager)(obs)
    mu = alg.human_actor(obs)  # Frozen by the policies.
  _assert_same(_state(_normalizer(alg.human_actor)), stats)  # Never updated.
  assert isinstance(mean_policy, HumanOnlyInferencePolicy)
  assert (mean_policy.human_mode, mean_policy.osl_mode) == ("mean", "mean")
  torch.testing.assert_close(mean[:, rows_h], mu)
  for out in (mean, expected):
    assert torch.equal(out[:, rows_o], torch.zeros(NUM_ENVS, 2))
  lo, hi = action_bounds(manager, alg.layout, "human")
  assert lo.tolist() == [0.0 if model == "msk" else -1.0] * HUMAN_DIM
  torch.testing.assert_close(
    expected[:, rows_h], expected_clipped_gaussian(mu, alg.human_std, lo, hi)
  )
  torch.testing.assert_close(alg.human_std, distribution_std(alg.human_actor))
  for osl_mode in ("sampled", "teacher"):
    with pytest.raises(ValueError, match="no OSL actor"):
      alg.inference_policy("mean", manager, cast(Any, osl_mode))
  with pytest.raises(ValueError, match="action_manager"):
    alg.inference_policy("expected")
  other: Model = "robot" if model == "msk" else "msk"
  with pytest.raises(ValueError, match="layout"):
    alg.inference_policy("mean", _action_manager(other))


##
# Import.
##


_PPO_BEFORE_THE_WALKER = """
import importlib, importlib.util, sys, types
import mjlab
held = types.ModuleType("mjlab.tasks")
held.__path__ = list(importlib.util.find_spec("mjlab.tasks").submodule_search_locations)
sys.modules["mjlab.tasks"] = held  # The walker waits.
import mjlab.tasks.osl_fsm.rl.ppo
del sys.modules["mjlab.tasks"]
importlib.import_module("mjlab.tasks")  # Now the walker, over every task.
"""


def test_modules_import_cleanly() -> None:
  """The tasks walker imports every package and re-raises, so a broken import here
  would end every task launch. Importing any task module runs the walker first,
  but stage 1.5's registration, walked as ``osl_fsm`` before ``osl_gait_library``,
  ``osl_mimic`` and ``osl_split``, will import the algorithm, and with it stage 2's
  config package, ahead of them. A fresh interpreter imports the algorithm before
  any other task package, then lets the walker import every task. The runner finds
  the class by the name the stage-1.5 config gives."""
  for name in ("rl", "rl.policy", "rl.ppo"):
    importlib.import_module(f"mjlab.tasks.osl_fsm.{name}")
  result = subprocess.run(
    [sys.executable, "-c", _PPO_BEFORE_THE_WALKER],
    capture_output=True,
    text=True,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    check=False,
  )
  assert result.returncode == 0, result.stderr[-3000:]
  assert resolve_callable("mjlab.tasks.osl_fsm.rl.ppo:HumanOnlyPPO") is HumanOnlyPPO
