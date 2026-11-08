"""Contracts for stage 2's runner, task registration and OSL export
(``tasks/osl_split/rl/runner.py``, ``rl/export.py``, ``config/__init__.py``).

CPU only and without an mjlab env:

* the registration against the config builders, and the stage-1 tasks untouched;
* the agents' reward composition against a fake reward manager;
* the export's metadata against fake env internals, and its ONNX against the actor
  through onnxruntime;
* the runner end to end on a fake vec env with tiny networks: the warm start, the
  stage-local schedules, the logged scalars, both validation families,
  checkpoints with their exports, resuming (on the checkpoint's curriculum),
  play's load of the actors alone, and the refusals;
* the validation env's config, taken from the training env's where it matters
  (the env itself is never built).

A slow test runs the real stage-2 task for two iterations on the GPU; it needs a
unicycle library and a stage-1 run.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensordict import TensorDict

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  OSL_TORQUE_JOINTS,
  Model,
  Socket,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import TendonEffortAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.osl_gait_library.config.rl_cfg import osl_gait_library_runner_cfg
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_gait_library.rl.runner import GaitLibraryOnPolicyRunner
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.mdp.rewards import mimic_reward
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel, RunningMeanStd
from mjlab.tasks.osl_split.config import (
  SPLIT_TASKS,
  TASK_PREFIX,
  split_experiment_name,
  split_task_id,
)
from mjlab.tasks.osl_split.config.env_cfgs import osl_split_env_cfg
from mjlab.tasks.osl_split.config.rl_cfg import (
  OBS_GROUPS,
  STAGE2_ITERATIONS,
  SplitRunnerCfg,
  WarmStartCfg,
  osl_split_runner_cfg,
)
from mjlab.tasks.osl_split.config.sensors import (
  ACC_RANGE,
  ANKLE_FORCE,
  ANKLE_POS,
  ANKLE_VEL,
  GYRO_RANGE,
  IMU_ACC,
  IMU_GYRO,
  KNEE_POS,
  KNEE_TORQUE,
  KNEE_VEL,
  OSL_SENSORS,
  SOCKET_FORCE,
  SOCKET_TORQUE,
)
from mjlab.tasks.osl_split.mdp import (
  OSL_OBS_TERMS,
  OslObsCfg,
  SplitRewardCfg,
  osl_actpack_dim,
  osl_loadcell_dim,
  osl_obs_dim,
  osl_observation_group,
  split_mimic_reward,
)
from mjlab.tasks.osl_split.mdp.rewards import stage1_channel_weights
from mjlab.tasks.osl_split.rl import runner as split_runner
from mjlab.tasks.osl_split.rl.export import (
  export_osl_policy,
  osl_policy_metadata,
  term_channels,
)
from mjlab.tasks.osl_split.rl.layout import ActionLayout
from mjlab.tasks.osl_split.rl.policy import distribution_std
from mjlab.tasks.osl_split.rl.runner import AgentRewards, SplitOnPolicyRunner
from mjlab.tasks.osl_split.rl.schedules import ChannelSchedule
from mjlab.tasks.osl_split.rl.storage import obs_group
from mjlab.tasks.osl_split.rl.warmstart import (
  LINEAGE_KEYS,
  WARM_START_FORMAT,
  WarmStart,
  load_warm_start,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

NUM_ENVS = 6
NUM_STEPS = 4
ACTOR_DIM = 12
HUMAN_DIM = 5
HIDDEN = (16, 16)
OSL_OBS = OslObsCfg(history_length=2)
OSL_DIM = osl_obs_dim(OSL_OBS)
EPISODE_STEPS = 3
TWIST_WEIGHT = 0.06
TWIST_SPLIT = (1.0, 0.5)
LOAD_TARE = torch.linspace(-3.0, 3.0, 12)
STAGE1_ITER = 40
STAGE1_COUNTER = 3280
STAGE1_ENV_STEPS = 123_456
STAGE_START = STAGE1_ITER + 1
ITERATIONS = 3

MOTORS_XML = """
<mujoco>
  <worldbody>
    <body name="robot/shank">
      <joint name="robot/OSL_KA_L1_osl_knee_angle_r"/>
      <geom size="0.05"/>
      <body name="robot/foot" pos="0 0 -0.4">
        <joint name="robot/OSL_KA_L1_osl_ankle_angle_r"/>
        <geom size="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="robot/OSL_KA_L1_osl_knee_angle_r" gear="49.4" ctrlrange="-2.88 2.88"/>
    <general joint="robot/OSL_KA_L1_osl_ankle_angle_r" gainprm="58.4"
      ctrlrange="-2.88 2.88"/>
  </actuator>
</mujoco>
"""
"""The two OSL motors as the scene names them: the robot model's gear and the
musculoskeletal model's gain, both of which the export must read."""

STAGE1_UNICYCLE_TASKS: dict[str, tuple[Model, Socket, Reward]] = {
  "Mjlab-Gait-Library-Unicycle-OSL-Fullbody"
  + ("-MSK" if model == "msk" else "")
  + ("-CLF" if reward == "clf" else "")
  + ("-Compliant" if socket == "compliant" else ""): (model, socket, reward)
  for model, socket, reward in SPLIT_TASKS.values()
}

##
# Fakes.
##


class _FakeMuscles(TendonEffortAction):
  """A muscle action term without an env: only its type and size matter."""

  def __init__(self, dim: int) -> None:
    self._action_dim = dim


def _action_manager() -> Any:
  """The MSK term order: ``osl_motors`` (rows 0:2), then ``muscles``."""
  terms = {
    "osl_motors": SimpleNamespace(
      action_dim=2, target_names=list(OSL_TORQUE_JOINTS), scale=2.88
    ),
    "muscles": _FakeMuscles(HUMAN_DIM),
  }
  return SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__)


def _observation_manager() -> Any:
  """The real ``osl`` group config (train mode), with a load-cell term instance
  reduced to its tare."""
  cfgs = dict(osl_observation_group("train", OSL_OBS).terms)
  cfgs["loadcell"] = replace(cfgs["loadcell"], func=SimpleNamespace(tare=LOAD_TARE))
  frames = OSL_OBS.history_length
  dims = [
    (osl_actpack_dim(OSL_OBS) * frames,),
    (osl_loadcell_dim(OSL_OBS) * frames,),
    (2 * frames,),
  ]
  return SimpleNamespace(
    active_terms={"osl": list(cfgs)},
    group_obs_term_dim={"osl": dims},
    group_obs_dim={"actor": (ACTOR_DIM,), "osl": (OSL_DIM,)},
    get_term_cfg=lambda group, name: cfgs[name],
  )


class _FakeSplitTerm:
  """What the runner reads and sets of ``split_mimic_reward``."""

  def __init__(self, num_envs: int) -> None:
    self.channel_rewards = torch.zeros(num_envs, 2)
    self.components: dict[str, torch.Tensor] = {}
    self.weights_set: list[dict[str, dict[str, float]]] = []
    self._weights = stage1_channel_weights(clf=True)

  @property
  def group_weights(self) -> dict[str, dict[str, float]]:
    return copy.deepcopy(self._weights)

  def set_group_weights(self, weights: dict[str, dict[str, float]]) -> None:
    self._weights = {c: dict(g) for c, g in weights.items()}
    self.weights_set.append(self.group_weights)

  def check_group_weights(
    self, weights: dict[str, dict[str, float]]
  ) -> dict[str, dict[str, float]]:
    return {c: dict(g) for c, g in weights.items()}

  def effective_shares(self) -> dict[str, dict[str, float]]:
    counts = {"human": 84, "osl_sites": 18, "osl_joints": 2}
    shares = {}
    for channel, w in self._weights.items():
      total = sum(w[g] * n for g, n in counts.items())
      shares[channel] = {g: w[g] * n / total for g, n in counts.items()}
    return shares


class _FakeRewardManager:
  def __init__(self, num_envs: int) -> None:
    self.mimic = _FakeSplitTerm(num_envs)
    self.twist = SimpleNamespace(components={})
    self.cfgs: dict[str, Any] = {
      "mimic": SimpleNamespace(func=self.mimic, weight=1.0),
      "twist": SimpleNamespace(func=self.twist, weight=TWIST_WEIGHT),
    }
    self._step_reward = torch.zeros(num_envs, 2)

  @property
  def active_terms(self) -> list[str]:
    return list(self.cfgs)

  def get_term_cfg(self, name: str) -> Any:
    return self.cfgs[name]


class _FakeEnv:
  """A vec env and its unwrapped env in one object: what the runner, the algorithm
  and the export read. Episodes time out after ``EPISODE_STEPS`` steps or end at
  random; every step's channel and twist rewards are random, and the agents'
  expected rewards are recorded."""

  def __init__(self, num_envs: int = NUM_ENVS, seed: int = 0) -> None:
    self.num_envs = num_envs
    self.num_actions = HUMAN_DIM + 2
    self.device = "cpu"
    self.max_episode_length = 1000
    self.step_dt = 0.01
    self.physics_dt = 0.002
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self.cfg: Any = SimpleNamespace(scale_rewards_by_dt=False, commands={})
    self.action_manager = _action_manager()
    self.reward_manager = _FakeRewardManager(num_envs)
    self.observation_manager = _observation_manager()
    clips = SimpleNamespace(clip_groups=["fwd", "turn"], num_clips=2)
    self.command_manager = SimpleNamespace(get_term=lambda name: clips)
    self.sim = SimpleNamespace(mj_model=mujoco.MjModel.from_xml_string(MOTORS_XML))
    self.expected_agent_rewards: list[torch.Tensor] = []
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

  def reset(self) -> tuple[TensorDict, dict]:
    self.episode_length_buf.zero_()
    return self.get_observations(), {}

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    n = self.num_envs
    assert actions.shape == (n, self.num_actions)
    assert bool(torch.isfinite(actions).all())
    channels = self._rand(n, 2)
    mimic = self._rand(n)
    twist = TWIST_WEIGHT * self._rand(n)
    manager = self.reward_manager
    manager.mimic.channel_rewards = channels
    manager.mimic.components = {
      "split_reward_human": channels[:, 0],
      "split_reward_osl": channels[:, 1],
      "err_site_abs": self._rand(n),
    }
    manager.twist.components = {"twist_reward": twist / TWIST_WEIGHT}
    manager._step_reward = torch.stack((mimic, twist), dim=-1)
    self.expected_agent_rewards.append(
      channels + twist.unsqueeze(-1) * torch.tensor(TWIST_SPLIT)
    )
    self.common_step_counter += 1
    self.episode_length_buf += 1
    time_outs = self.episode_length_buf >= EPISODE_STEPS
    dones = time_outs | (self._rand(n) < 0.1)
    self.episode_length_buf[dones] = 0
    extras = {"time_outs": time_outs, "log": {"Episode_Reward/mimic": mimic.mean()}}
    return self.get_observations(), mimic + twist, dones.long(), extras

  def close(self) -> None:
    pass


class _Runner(SplitOnPolicyRunner):
  """The runner with a fake validation env (the real one is an mjlab env)."""

  def _make_validation_env(self) -> Any:
    return _FakeEnv(num_envs=4, seed=1)


def _layout() -> ActionLayout:
  return ActionLayout.from_action_manager(_action_manager())


def _model_block(key: str) -> dict[str, Any]:
  """The ``key`` network's block of :func:`_train_cfg`, as the converter records
  stage 1's in the lineage (plain types)."""
  block = asdict(osl_split_runner_cfg("msk", "compliant", "clf", "test"))[key]
  block["hidden_dims"] = list(HIDDEN)
  return block


def _lineage(reward: str = "clf") -> dict[str, Any]:
  layout = _layout()
  lineage: dict[str, Any] = {
    "format": WARM_START_FORMAT,
    "stage1_run_dir": "/runs/stage1",
    "stage1_checkpoint": f"/runs/stage1/model_{STAGE1_ITER}.pt",
    "iter": STAGE1_ITER,
    "common_step_counter": STAGE1_COUNTER,
    "env_steps": STAGE1_ENV_STEPS,
    "model": "msk",
    "socket": "compliant",
    "reward": reward,
    "variant": "unicycle",
    "experiment_name": "stage1",
    "obs_dim": ACTOR_DIM,
    "action_terms": layout.to_list(),
    "human_rows": layout.rows("human"),
    "osl_rows": layout.rows("osl"),
    "actor_cfg": _model_block("actor"),
    "critic_cfg": _model_block("critic"),
    "num_updates_done": STAGE1_ITER + 1,
    "human_std": {"min": 3.0, "median": 3.0, "max": 3.0},
    "osl_std": [3.0, 3.0],
    "wandb_run_id": None,
    "created": "2026-11-07T00:00:00",
    "verified_max_abs_diff": 0.0,
  }
  assert set(LINEAGE_KEYS) <= set(lineage)
  return lineage


def _randomized(model: MuscleMimicMLPModel) -> MuscleMimicMLPModel:
  with torch.no_grad():
    for p in model.parameters():
      p.normal_(0.0, 0.3)
    normalizer = model.obs_normalizer
    assert isinstance(normalizer, RunningMeanStd)
    normalizer.mean.normal_()
    normalizer.var.uniform_(0.5, 2.0)
    normalizer.count.fill_(2.0e8)
  return model


def _write_warm_start(path: Path, reward: str = "clf") -> WarmStart:
  """A warm-start file cut from a random unified actor, as the converter cuts one."""
  torch.manual_seed(0)
  obs = _FakeEnv().get_observations()
  layout = _layout()
  unified = _randomized(
    MuscleMimicMLPModel(
      obs,
      {"actor": ["actor"]},
      "actor",
      layout.total_dim,
      hidden_dims=HIDDEN,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 3.0,
        "std_type": "log",
      },
    )
  )
  critic = _randomized(
    MuscleMimicMLPModel(obs, {"critic": ["actor"]}, "critic", 1, hidden_dims=HIDDEN)
  )
  state = {k: v.detach().clone() for k, v in unified.state_dict().items()}
  last = len(unified.mlp) - 1
  rows = {f"mlp.{last}.weight", f"mlp.{last}.bias", "distribution.log_std_param"}
  human, osl = layout.index("human"), layout.index("osl")
  ws = WarmStart(
    human_actor={
      k: v.index_select(0, human) if k in rows else v for k, v in state.items()
    },
    human_critic={k: v.detach().clone() for k, v in critic.state_dict().items()},
    teacher_head={
      "weight": state[f"mlp.{last}.weight"].index_select(0, osl),
      "bias": state[f"mlp.{last}.bias"].index_select(0, osl),
      "log_std": state["distribution.log_std_param"].index_select(0, osl),
    },
    return_stats={
      "mean": torch.tensor(0.5),
      "var": torch.tensor(4.0),
      "count": torch.tensor(2.0e8),
    },
    lineage=_lineage(reward),
  )
  ws.save(path)
  return ws


def _train_cfg(warm_start: Path | None) -> dict[str, Any]:
  """The real MSK-CLF-compliant runner config, shrunk."""
  cfg = asdict(osl_split_runner_cfg("msk", "compliant", "clf", "test"))
  for key in ("actor", "critic", "osl_critic"):
    cfg[key]["hidden_dims"] = HIDDEN
  cfg["osl_actor"]["hidden_dims"] = (8, 8)
  cfg["num_steps_per_env"] = NUM_STEPS
  cfg["algorithm"]["num_mini_batches"] = 2
  cfg["max_iterations"] = ITERATIONS
  cfg["save_interval"] = 2
  cfg["logger"] = "tensorboard"
  cfg["upload_model"] = False
  cfg["validation"].update(interval=2, num_steps=5, num_envs=4)
  cfg["freeze"].update(human_actor_iterations=2, critic_warmup_iterations=1)
  cfg["channels"].update(anneal_start_iteration=0, anneal_iterations=2)
  cfg["channels"]["additive_term_weights"] = {"twist": TWIST_SPLIT}
  cfg["warm_start"]["path"] = str(warm_start) if warm_start is not None else ""
  return cfg


def _scalars(log_dir: Path, *tags: str) -> dict[str, dict[int, float]]:
  """``{tag: {step: value}}`` from the run's TensorBoard events."""
  events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  events.Reload()
  return {tag: {e.step: e.value for e in events.Scalars(tag)} for tag in tags}


def _states(runner: SplitOnPolicyRunner) -> dict[str, dict[str, torch.Tensor]]:
  return {name: model.state_dict() for name, model in runner.alg.models.items()}


##
# Registration.
##


def test_split_tasks_are_registered() -> None:
  ids = sorted(t for t in list_tasks() if t.startswith("Mjlab-Split-"))
  tags = [
    m + r + s for m in ("", "-MSK") for r in ("", "-CLF") for s in ("", "-Compliant")
  ]
  assert ids == sorted(SPLIT_TASKS) == sorted(TASK_PREFIX + t for t in tags)
  for task_id, (model, socket, reward) in SPLIT_TASKS.items():
    assert split_task_id(model, socket, reward) == task_id
    assert load_runner_cls(task_id) is SplitOnPolicyRunner
    # The experiment name follows the id's tags: -MSK-CLF -> _msk_clf.
    suffix = task_id.removeprefix(TASK_PREFIX).lower().replace("-", "_")
    experiment = "osl_fullbody_split_unicycle" + suffix
    assert split_experiment_name(model, socket, reward) == experiment

    rl_cfg = load_rl_cfg(task_id)
    assert isinstance(rl_cfg, SplitRunnerCfg)
    assert rl_cfg == osl_split_runner_cfg(model, socket, reward, experiment)
    assert rl_cfg.experiment_name == experiment
    assert dict(rl_cfg.obs_groups) == OBS_GROUPS
    validation = rl_cfg.validation
    assert (validation.model, validation.socket, validation.reward) == (
      model,
      socket,
      reward,
    )
    assert validation.variant == "unicycle"
    assert rl_cfg.algorithm.class_name == "mjlab.tasks.osl_split.rl.ppo:TwoAgentPPO"
    assert rl_cfg.warm_start == WarmStartCfg()
    assert rl_cfg.max_iterations == STAGE2_ITERATIONS
    assert rl_cfg.export_osl_onnx
    stage1 = stage1_channel_weights(reward == "clf")
    assert asdict(rl_cfg.channels.human_start) == stage1["human"]
    assert asdict(rl_cfg.channels.osl_start) == stage1["osl"]

    for play in (False, True):
      env_cfg = load_env_cfg(task_id, play=play)
      assert set(env_cfg.observations) == {"actor", "osl"}
      assert env_cfg.observations["osl"].enable_corruption is not play
      assert env_cfg.rewards["mimic"].func is split_mimic_reward
      assert env_cfg.scale_rewards_by_dt is False


def test_stage1_tasks_are_unchanged() -> None:
  """Stage 2 registers its own tasks and leaves the ones it starts from alone."""
  for task_id, (model, socket, reward) in STAGE1_UNICYCLE_TASKS.items():
    suffix = task_id.removeprefix("Mjlab-Gait-Library-Unicycle-OSL-Fullbody")
    experiment = "osl_fullbody_unicycle" + suffix.lower().replace("-", "_")
    assert load_runner_cls(task_id) is GaitLibraryOnPolicyRunner
    assert load_rl_cfg(task_id) == osl_gait_library_runner_cfg(
      model, socket, reward, experiment, "unicycle"
    )
    env_cfg = load_env_cfg(task_id)
    assert set(env_cfg.observations) == {"actor", "critic"}
    assert env_cfg.rewards["mimic"].func is mimic_reward


##
# Agent rewards.
##


def test_agent_rewards_compose_channels_and_additive_terms() -> None:
  env = _FakeEnv()
  manager = env.reward_manager
  manager.cfgs["alive"] = SimpleNamespace(func=None, weight=0.5)
  channels = torch.rand(NUM_ENVS, 2)
  step = torch.rand(NUM_ENVS, 3)
  manager.mimic.channel_rewards = channels
  manager._step_reward = step

  rewards = AgentRewards(cast(Any, env), {"twist": (1.0, 0.25)})
  assert rewards.additive == {"twist": (1.0, 0.25), "alive": (1.0, 1.0)}
  assert rewards.term is manager.mimic
  expected = channels + step[:, 1:2] * torch.tensor([1.0, 0.25]) + step[:, 2:3]
  torch.testing.assert_close(rewards(), expected, rtol=0.0, atol=1e-6)

  # The channels carry the mimic term's weight, as the team reward does.
  manager.cfgs["mimic"].weight = 2.0
  torch.testing.assert_close(
    AgentRewards(cast(Any, env), {})(),
    2.0 * channels + step[:, 1:2] + step[:, 2:3],
    rtol=0.0,
    atol=1e-6,
  )

  # Both channels at the team's imitation value: both agents get the team reward,
  # bit for bit.
  manager.cfgs = {k: manager.cfgs[k] for k in ("mimic", "twist")}
  manager.cfgs["mimic"].weight = 1.0
  mimic, twist = torch.rand(NUM_ENVS), 0.06 * torch.rand(NUM_ENVS)
  manager._step_reward = torch.stack((mimic, twist), dim=-1)
  manager.mimic.channel_rewards = torch.stack((mimic, mimic), dim=-1)
  team = torch.zeros(NUM_ENVS) + mimic + twist  # The reward manager's sum.
  assert torch.equal(AgentRewards(cast(Any, env), {})(), torch.stack((team, team), -1))


def test_agent_rewards_refusals() -> None:
  env = _FakeEnv()
  manager = env.reward_manager
  for weights in ({"nope": (1.0, 1.0)}, {"mimic": (1.0, 1.0)}):
    with pytest.raises(ValueError, match="additive reward terms are \\['twist'\\]"):
      AgentRewards(cast(Any, env), weights)
  for pair in ((1.0,), (1.0, math.inf)):
    with pytest.raises(ValueError, match="finite \\(human, osl\\) pair"):
      AgentRewards(cast(Any, env), {"twist": pair})
  manager.cfgs["mimic"].weight = 0.0
  with pytest.raises(ValueError, match="non-zero weight"):
    AgentRewards(cast(Any, env), {})
  manager.cfgs["mimic"] = SimpleNamespace(func=SimpleNamespace(), weight=1.0)
  with pytest.raises(TypeError, match="split_mimic_reward"):
    AgentRewards(cast(Any, env), {})
  env.cfg.scale_rewards_by_dt = True
  with pytest.raises(ValueError, match="scale_rewards_by_dt"):
    AgentRewards(cast(Any, env), {})


##
# Export.
##


def test_osl_policy_metadata() -> None:
  env = _FakeEnv()
  actor = SimpleNamespace(obs_dim=OSL_DIM, obs_groups=["osl"])
  alg = SimpleNamespace(
    osl_actor=actor, osl_std=torch.tensor([0.2, 0.3]), layout=_layout()
  )
  lineage = _lineage()
  meta = osl_policy_metadata(cast(Any, env), cast(Any, alg), lineage)
  assert json.loads(json.dumps(meta, allow_nan=False)) == meta  # Plain JSON.

  obs = meta["obs"]
  frames = OSL_OBS.history_length
  assert obs["group"] == "osl" and obs["dim"] == OSL_DIM == 24 * frames
  assert obs["layout"] == "term-major, oldest to newest"
  terms = {t["name"]: t for t in obs["terms"]}
  assert list(terms) == list(OSL_OBS_TERMS)
  assert [t["offset"] for t in obs["terms"]] == [0, 10 * frames, 22 * frames]
  assert [t["frame_dim"] for t in obs["terms"]] == [10, 12, 2]
  assert all(t["history_length"] == frames for t in obs["terms"])
  actpack = terms["actpack"]
  assert actpack["channels"][:4] == [KNEE_POS, ANKLE_POS, KNEE_VEL, ANKLE_VEL]
  assert actpack["channels"][4:7] == [f"{IMU_GYRO}_{a}" for a in "xyz"]
  assert actpack["units"] == ["rad"] * 2 + ["rad/s"] * 5 + ["m/s^2"] * 3
  loadcell = terms["loadcell"]
  assert loadcell["channels"][0] == f"{ANKLE_FORCE}_x"
  assert loadcell["sensors"][-2:] == [SOCKET_FORCE, SOCKET_TORQUE]
  assert terms["last_action"]["sensors"] == []

  sensors = {s["name"]: s for s in meta["sensors"]["sensors"]}
  np.testing.assert_allclose(sensors[ANKLE_FORCE]["tare"], LOAD_TARE[:3])
  np.testing.assert_allclose(sensors[SOCKET_TORQUE]["tare"], LOAD_TARE[9:])
  assert sensors[IMU_GYRO]["observed"] and sensors[IMU_GYRO]["object"] == "osl_imu"
  assert sensors[IMU_GYRO]["scene_name"] == "robot/osl_imu_gyro"
  assert not sensors[KNEE_TORQUE]["observed"]  # Applied torque is off by default.
  assert "tare" not in sensors[KNEE_POS]
  # The IMU saturates at the ActPack's ranges, in the observation (the MuJoCo
  # sensors have no cutoff to read it from); nothing else saturates.
  saturation = {name: sensors[name]["saturation"] for name in OSL_SENSORS}
  assert saturation.pop(IMU_GYRO) == GYRO_RANGE
  assert saturation.pop(IMU_ACC) == ACC_RANGE
  assert set(saturation.values()) == {None}

  action = meta["action"]
  assert action["joints"] == list(OSL_TORQUE_JOINTS)
  assert action["env_rows"] == [0, 1]
  np.testing.assert_allclose(action["gains"], [49.4, 58.4])
  np.testing.assert_allclose(action["torque_limit"], [49.4 * 2.88, 58.4 * 2.88])
  assert action["scale"] == [2.88, 2.88]
  assert action["ctrl_range"] == [[-2.88, 2.88]] * 2
  assert action["normalized_clip"] == [-1.0, 1.0]
  assert action["dt"] == 0.01 and action["control_hz"] == pytest.approx(100.0)
  assert meta["io"] == {
    "input": "obs",
    "input_shape": [1, OSL_DIM],
    "output": "actions",
    "output_shape": [1, 2],
  }

  observation = meta["randomization"]["observation"]
  for bus in ("actpack", "loadcell"):
    assert observation[bus]["delay_steps"] == list(OSL_OBS.delay_steps)
    assert observation[bus]["delay_held_per_episode"]
  assert observation["actpack"]["noise_std"][:4] == [0.002, 0.002, 0.05, 0.05]
  assert observation["actpack"]["per_episode_bias"]["imu_mount_deg"] == 3.0
  assert observation["loadcell"]["per_episode_bias"] == {
    "force_bias": 2.0,
    "torque_bias": 0.2,
  }
  last = observation["last_action"]
  assert last["delay_steps"] == [0, 0] and last["noise_std"] is None
  assert meta["std"] == pytest.approx([0.2, 0.3])
  assert meta["lineage"] == lineage
  assert "osl_actuator_delay_s" not in meta["randomization"]

  # With the real training config, the OSL motors' delay and strength ranges.
  env.cfg = osl_split_env_cfg("msk", "compliant", "clf", "train")
  randomization = osl_policy_metadata(cast(Any, env), cast(Any, alg), None)[
    "randomization"
  ]
  assert randomization["osl_actuator_delay_s"] == pytest.approx([0.0, 0.02])
  assert randomization["osl_motor_strength"] == pytest.approx([0.9, 1.1])

  actor.obs_dim = OSL_DIM - 1
  with pytest.raises(ValueError, match="OSL actor reads"):
    osl_policy_metadata(cast(Any, env), cast(Any, alg), None)


def test_term_channels_follow_the_options() -> None:
  extras = term_channels("actpack", {"projected_gravity": True, "applied_torque": True})
  assert len(extras) == osl_actpack_dim(
    OslObsCfg(projected_gravity=True, applied_torque=True)
  )
  assert [c for c, _, _ in extras[-5:]] == [
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    KNEE_TORQUE,
    "osl_ankle_torque",
  ]
  ankle_only = term_channels("loadcell", {"socket": False})
  assert len(ankle_only) == osl_loadcell_dim(OslObsCfg(socket_load_cell=False))
  with pytest.raises(ValueError, match="Unknown OSL observation term"):
    term_channels("phase", {})


def test_export_matches_the_actor(tmp_path: Path) -> None:
  onnxruntime = pytest.importorskip("onnxruntime")
  onnx = pytest.importorskip("onnx")
  torch.manual_seed(1)
  env = _FakeEnv()
  obs = env.get_observations()
  actor = _randomized(
    MuscleMimicMLPModel(
      obs,
      {"osl_actor": ["osl"]},
      "osl_actor",
      2,
      hidden_dims=(8, 8),
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.2,
        "std_type": "log",
      },
    )
  )
  actor.set_norm_mode("update")
  before = {k: v.clone() for k, v in actor.state_dict().items()}
  alg = SimpleNamespace(
    osl_actor=actor, osl_std=distribution_std(actor).detach(), layout=_layout()
  )

  path = export_osl_policy(
    cast(Any, alg), cast(Any, env), tmp_path / "exported", "osl_policy_7", {"iter": 6}
  )
  assert path == tmp_path / "exported" / "osl_policy_7.onnx"
  meta = json.loads(path.with_suffix(".json").read_text())
  assert meta["lineage"] == {"iter": 6}
  embedded = {p.key: json.loads(p.value) for p in onnx.load(str(path)).metadata_props}
  assert embedded == meta

  x = obs["osl"][:1]
  session = onnxruntime.InferenceSession(str(path))
  (exported,) = session.run(["actions"], {"obs": x.numpy()})
  actor.set_norm_mode("frozen")
  with torch.no_grad():
    expected = actor(TensorDict({"osl": x}, batch_size=[1]))
  np.testing.assert_allclose(exported, expected.numpy(), rtol=1e-5, atol=1e-6)
  for key, value in actor.state_dict().items():  # The export worked on a copy.
    assert torch.equal(value, before[key])


##
# The runner, end to end on a fake env.
##


def test_runner_trains_validates_saves_and_resumes(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)
  log_dir = tmp_path / "run"
  cfg = _train_cfg(ws_path)
  env = _FakeEnv()
  runner = _Runner(cast(Any, env), cfg, str(log_dir), "cpu")
  assert runner.stage_start == runner.current_learning_iteration == STAGE_START
  assert env.common_step_counter == STAGE1_COUNTER
  assert runner.lineage is not None and runner.lineage["iter"] == STAGE1_ITER
  assert not runner.reward_switch and runner.schedule_iterations == ITERATIONS

  recorded: list[torch.Tensor] = []
  process_env_step = runner.alg.process_env_step

  def spy(obs: TensorDict, rewards: torch.Tensor, *args: Any) -> None:
    recorded.append(rewards.clone())
    process_env_step(obs, rewards, *args)

  monkeypatch.setattr(runner.alg, "process_env_step", spy)
  runner.learn(ITERATIONS)

  # The agents learned from their own rewards: channels plus the weighted twist.
  assert len(recorded) == ITERATIONS * NUM_STEPS
  for got, expected in zip(recorded, env.expected_agent_rewards, strict=True):
    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-6)

  # Schedules run on the stage-local iteration: reward weights every iteration...
  channels = ChannelSchedule.from_train_cfg(cfg)
  weights = [channels.weights(k) for k in range(ITERATIONS)]
  assert env.reward_manager.mimic.weights_set == weights
  assert weights[1]["osl"]["osl_joints"] == 5.0  # Half-way to osl_end.
  # ...and the validation env's before each validation: zero-shot, then k = 1.
  validation_env = cast(_FakeEnv, runner._validation_env)
  assert validation_env.reward_manager.mimic.weights_set == weights[:2]

  steps = [STAGE_START + k for k in range(ITERATIONS)]
  logged = _scalars(
    log_dir,
    "Stage2/iteration",
    "Stage2/human_frozen",
    "Stage2/osl_actor_frozen",
    "LR/osl_actor",
    "LR/human_actor",
    "Loss/learning_rate",
    "Channels/osl/osl_joints",
    "Train/total_env_steps",
    "Policy/osl/std_knee",
    "Train/mean_reward",
    "Train/human/mean_reward",
    "Train/osl/mean_reward",
    "Validation/mean_episode_return_osl",
    "ValidationExpected/mean_episode_return_human",
    "Validation/split_reward_osl/turn",
  )
  assert logged["Stage2/iteration"] == {s: float(k) for k, s in enumerate(steps)}
  assert list(logged["Stage2/human_frozen"].values()) == [1.0, 1.0, 0.0]
  assert list(logged["Stage2/osl_actor_frozen"].values()) == [1.0, 0.0, 0.0]
  assert logged["LR/osl_actor"][steps[0]] == 0.0 < logged["LR/osl_actor"][steps[1]]
  assert logged["Loss/learning_rate"] == logged["LR/osl_actor"]
  assert logged["LR/human_actor"][steps[1]] == 0.0 < logged["LR/human_actor"][steps[2]]
  assert list(logged["Channels/osl/osl_joints"].values()) == [0.0, 5.0, 10.0]
  samples = NUM_ENVS * NUM_STEPS
  assert list(logged["Train/total_env_steps"].values()) == [
    STAGE1_ENV_STEPS + samples * (k + 1) for k in range(ITERATIONS)
  ]
  assert logged["Policy/osl/std_knee"][steps[0]] == pytest.approx(0.2)
  for tag in ("Train/mean_reward", "Train/human/mean_reward", "Train/osl/mean_reward"):
    assert set(logged[tag]) == set(steps), tag
  for tag in (
    "Validation/mean_episode_return_osl",
    "ValidationExpected/mean_episode_return_human",
    "Validation/split_reward_osl/turn",
  ):
    assert set(logged[tag]) == {steps[0], steps[1]}, tag  # Zero-shot, then k = 1.

  # Saves at k = 0 and 2, each with the OSL actor's export.
  for it in (steps[0], steps[2]):
    assert (log_dir / f"model_{it}.pt").is_file()
    assert (log_dir / "exported" / f"osl_policy_{it}.onnx").is_file()
    assert (log_dir / "exported" / f"osl_policy_{it}.json").is_file()
  assert not (log_dir / f"model_{steps[1]}.pt").exists()
  meta = json.loads((log_dir / "exported" / f"osl_policy_{steps[2]}.json").read_text())
  assert meta["lineage"] == runner.lineage

  # A resumed run (train.py: construct without a warm start, then load) continues
  # after the checkpoint, with the stage's start, lineage and schedule.
  checkpoint = str(log_dir / f"model_{steps[2]}.pt")
  resumed = _Runner(cast(Any, _FakeEnv()), _train_cfg(None), None, "cpu")
  with pytest.raises(RuntimeError, match="--agent.warm-start.path"):
    resumed.learn(1)
  infos = resumed.load(checkpoint)
  assert resumed.current_learning_iteration == steps[2] + 1
  assert resumed.stage_start == STAGE_START
  assert resumed.lineage == runner.lineage
  assert resumed.schedule_iterations == ITERATIONS and not resumed.reward_switch
  assert resumed.env.unwrapped.common_step_counter == env.common_step_counter
  assert infos["env_state"]["common_step_counter"] == env.common_step_counter
  assert resumed.alg.stage_iteration == ITERATIONS - 1
  ours, theirs = _states(runner), _states(resumed)
  for name, state in ours.items():
    for key, value in state.items():
      assert torch.equal(value, theirs[name][key]), (name, key)

  # Play: load with play.py's arguments, then both actors' mean actions.
  played = _Runner(cast(Any, _FakeEnv()), _train_cfg(None), None, "cpu")
  played.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
  policy = played.get_inference_policy(device="cpu")
  obs = env.get_observations()
  with torch.inference_mode():
    actions = policy(obs)
    assert actions.shape == (NUM_ENVS, HUMAN_DIM + 2)
    assert torch.equal(actions, runner.alg.get_policy()(obs))

  # Stage-1 checkpoints and warm-start files are refused, with the way forward.
  stage1 = tmp_path / "model_610.pt"
  torch.save({"actor_state_dict": {}, "critic_state_dict": {}, "iter": 610}, stage1)
  for path in (stage1, ws_path):
    with pytest.raises(ValueError, match="--agent.warm-start.path"):
      played.load(str(path))


def test_warm_start_refusals_and_reward_switch(tmp_path: Path) -> None:
  ws = _write_warm_start(tmp_path / "ws.pt")
  runner = _Runner(cast(Any, _FakeEnv()), _train_cfg(None), None, "cpu")
  assert load_warm_start(tmp_path / "ws.pt").lineage == ws.lineage

  for key, value in (("socket", "rigid"), ("model", "robot")):
    other = replace(ws, lineage={**ws.lineage, key: value})
    with pytest.raises(ValueError, match="cannot switch between stages"):
      runner.warm_start(other)
  wider = replace(ws, lineage={**ws.lineage, "obs_dim": ACTOR_DIM + 1})
  with pytest.raises(ValueError, match="observations"):
    runner.warm_start(wider)

  # Stage 1 trained on MimicReward, stage 2 on the CLF: the critics start fresh
  # and the critic-only warm-up lengthens, also after a resume.
  runner.warm_start(replace(ws, lineage={**ws.lineage, "reward": "mimic"}))
  warmup = _train_cfg(None)["freeze"]["reward_switch_critic_warmup_iterations"]
  assert runner.reward_switch and runner.schedule.critic_warmup == warmup
  checkpoint = tmp_path / f"model_{STAGE_START}.pt"
  runner.save(str(checkpoint))
  resumed = _Runner(cast(Any, _FakeEnv()), _train_cfg(None), None, "cpu")
  assert not resumed.reward_switch
  resumed.load(str(checkpoint))
  assert resumed.reward_switch and resumed.schedule.critic_warmup == warmup


def _quick_cfg(warm_start: Path | None) -> dict[str, Any]:
  """:func:`_train_cfg` without validation or ONNX exports."""
  cfg = _train_cfg(warm_start)
  cfg["validation"]["interval"] = 0
  cfg["export_osl_onnx"] = False
  return cfg


def test_play_loads_only_the_actors(tmp_path: Path) -> None:
  """``play.py`` builds the runner from the task's registered config and loads
  with ``load_cfg={"actor": True}``. A run trained with ``osl_critic_obs=
  "full+osl"`` has a wider OSL critic than that config builds, so only the actors
  (and the teacher head) may load; the stage's place and the env's step counter
  come along, but a runner without the run's critics cannot train."""
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)
  cfg = _quick_cfg(ws_path)
  cfg["algorithm"]["osl_critic_obs"] = "full+osl"
  env = _FakeEnv()
  runner = _Runner(cast(Any, env), cfg, str(tmp_path / "run"), "cpu")
  assert runner.alg.osl_critic.obs_dim == ACTOR_DIM + OSL_DIM
  runner.learn(1)
  checkpoint = str(tmp_path / "run" / f"model_{STAGE_START}.pt")

  played = _Runner(cast(Any, _FakeEnv(seed=2)), _train_cfg(None), None, "cpu")
  assert played.alg.osl_critic.obs_dim == ACTOR_DIM  # The registered default.
  critics = {
    name: copy.deepcopy(played.alg.models[name].state_dict())
    for name in ("human_critic", "osl_critic")
  }
  played.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
  assert played.stage_start == STAGE_START
  assert played.lineage == runner.lineage
  assert played.env.unwrapped.common_step_counter == env.common_step_counter
  ours, theirs = _states(runner), _states(played)
  for name in ("human_actor", "osl_actor"):
    for key, value in ours[name].items():
      assert torch.equal(value, theirs[name][key]), (name, key)
  for key, value in runner.alg.teacher_head.state_dict().items():
    assert torch.equal(value, played.alg.teacher_head.state_dict()[key]), key
  for name, state in critics.items():  # Left as built.
    for key, value in state.items():
      assert torch.equal(value, theirs[name][key]), (name, key)
  obs = env.get_observations()
  with torch.inference_mode():
    actions = played.get_inference_policy(device="cpu")(obs)
    assert torch.equal(actions, runner.alg.get_policy()(obs))
  with pytest.raises(RuntimeError, match="Nothing to train from"):
    played.learn(1)


def test_resume_keeps_the_checkpoint_curriculum(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  """A resumed run continues on the curriculum its checkpoint was trained with
  (freeze, warm-up, learning rates, blend, the channels' anneal and the additive
  term weights), not on what the resuming command line happens to say: a resume
  with the task's defaults must not freeze an unfrozen human again or snap the OSL
  channel back to its start weights. Every overridden value is reported."""
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)
  cfg = _quick_cfg(ws_path)
  cfg["algorithm"].update(
    schedule_iterations=10,
    critic_lr=3e-4,
    human_actor_peak_lr=2e-4,
    human_actor_ramp_iterations=1,
  )
  runner = _Runner(cast(Any, _FakeEnv()), cfg, str(tmp_path / "run"), "cpu")
  runner.learn(ITERATIONS)
  checkpoint = tmp_path / "run" / f"model_{STAGE_START + ITERATIONS - 1}.pt"
  split = torch.load(checkpoint, weights_only=False)["infos"]["split"]
  assert split["schedule"] == asdict(runner.schedule)
  channels = runner.channel_schedule
  assert split["channel_schedule"] == {
    "start": channels.start,
    "end": channels.end,
    "anneal_start": 0,
    "anneal_iterations": 2,
  }
  assert split["additive_term_weights"] == {"twist": list(TWIST_SPLIT)}

  # Resumed with the task's own curriculum: a 200-iteration freeze, the channels'
  # anneal from there, default rates and the twist counted fully in both agents.
  stock = asdict(osl_split_runner_cfg("msk", "compliant", "clf", "test"))
  resumed_cfg = _quick_cfg(None)
  for key in ("freeze", "channels"):
    resumed_cfg[key] = stock[key]
  env = _FakeEnv(seed=1)
  resumed = _Runner(cast(Any, env), resumed_cfg, str(tmp_path / "resumed"), "cpu")
  assert resumed.schedule.human_freeze == 200
  capsys.readouterr()
  resumed.load(str(checkpoint))
  out = capsys.readouterr().out
  assert resumed.schedule == runner.schedule
  assert resumed.channel_schedule == runner.channel_schedule
  assert "[WARN]" in out and "checkpoint's" in out
  for name in (
    "schedule.human_freeze: checkpoint 2, config 200",
    "schedule.critic_warmup: checkpoint 1, config 10",
    "schedule.critic_lr",
    "schedule.human_peak_lr",
    "schedule.human_ramp",
    "channel_schedule.anneal_start: checkpoint 0, config 200",
    "channel_schedule.anneal_iterations: checkpoint 2, config 200",
    "additive_term_weights.twist",
  ):
    assert name in out, name
  assert "total_iterations" not in out  # The resumed run sets no stage length.

  recorded: list[torch.Tensor] = []
  process_env_step = resumed.alg.process_env_step

  def spy(obs: TensorDict, rewards: torch.Tensor, *args: Any) -> None:
    recorded.append(rewards.clone())
    process_env_step(obs, rewards, *args)

  monkeypatch.setattr(resumed.alg, "process_env_step", spy)
  resumed.learn(1)
  k = ITERATIONS  # The human learns, the OSL channel stays annealed...
  assert resumed.alg.lrs["human_actor"] == runner.schedule.lr("human_actor", k) > 0.0
  assert env.reward_manager.mimic.weights_set == [channels.weights(k)]
  assert channels.weights(k) == channels.end
  # ...and both agents keep the run's twist split.
  for got, expected in zip(recorded, env.expected_agent_rewards, strict=True):
    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-6)

  # A resume that repeats the run's settings warns about nothing.
  same = _Runner(cast(Any, _FakeEnv()), _quick_cfg(None), None, "cpu")
  same.cfg["algorithm"].update(cfg["algorithm"])
  capsys.readouterr()
  same.load(str(checkpoint))
  assert "curriculum" not in capsys.readouterr().out
  # A stage length set explicitly is compared too, and loses as well.
  same.cfg["algorithm"]["schedule_iterations"] = 5
  same.load(str(checkpoint))
  out = capsys.readouterr().out
  assert "schedule.total_iterations: checkpoint 10, config 5" in out
  assert same.schedule_iterations == 10

  # A checkpoint from before the curriculum was kept: its stage length and reward
  # switch, the rest from the config, and a warning.
  old = torch.load(checkpoint, weights_only=False)
  for key in ("schedule", "channel_schedule", "additive_term_weights"):
    del old["infos"]["split"][key]
  torch.save(old, tmp_path / "old.pt")
  older = _Runner(cast(Any, _FakeEnv()), copy.deepcopy(resumed_cfg), None, "cpu")
  older.load(str(tmp_path / "old.pt"))
  assert "predates" in capsys.readouterr().out
  assert older.schedule.total_iterations == runner.schedule.total_iterations
  assert older.schedule.human_freeze == 200
  assert older.channel_schedule == ChannelSchedule.from_train_cfg(resumed_cfg)


def test_validation_env_follows_the_training_env(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The validation env scores the reward being trained and feeds the OSL actor
  the observation it trains on: the imitation term's params (the channels'
  ``owner_penalties``, say) and the ``osl`` group (history, switches) come from
  the training env, the group cleaned of the training corruption. Without
  overrides that is exactly the task's validation config. The env itself is not
  built: its constructor records the config."""
  built: list[Any] = []

  def build(cfg: Any, device: str) -> _FakeEnv:  # Stands in for ManagerBasedRlEnv.
    del device
    built.append(cfg)
    return _FakeEnv(num_envs=cfg.scene.num_envs, seed=1)

  monkeypatch.setattr(split_runner, "ManagerBasedRlEnv", build)
  monkeypatch.setattr(split_runner, "RslRlVecEnvWrapper", lambda env: env)
  env = _FakeEnv()
  runner = SplitOnPolicyRunner(cast(Any, env), _quick_cfg(None), None, "cpu")

  def validation_cfg(train: Any) -> Any:
    env.cfg = train
    runner._make_validation_env()
    return built[-1]

  defaults = osl_split_env_cfg("msk", "compliant", "clf", "validation")
  cfg = validation_cfg(osl_split_env_cfg("msk", "compliant", "clf", "train"))
  assert cfg.rewards["mimic"].params == defaults.rewards["mimic"].params
  assert cfg.observations["osl"] == defaults.observations["osl"]
  assert cfg.scene.num_envs == 4  # validation.num_envs.

  osl_obs = OslObsCfg(history_length=4, applied_torque=True, socket_load_cell=False)
  train = osl_split_env_cfg("msk", "compliant", "clf", "train", osl_obs=osl_obs)
  mimic = train.rewards["mimic"]
  channels = SplitRewardCfg(owner_penalties=False, osl_joint_scale=0.3)
  mimic.params["channels"] = channels
  motion = train.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = str(tmp_path)
  cfg = validation_cfg(train)
  validation_motion = cfg.commands["motion"]
  assert isinstance(validation_motion, GaitLibraryCommandCfg)
  assert validation_motion.library_dir == str(tmp_path)
  params = cfg.rewards["mimic"].params
  assert params == mimic.params and params["channels"] == channels
  assert params is not mimic.params and params["channels"] is not channels
  group = cfg.observations["osl"]
  assert group == osl_observation_group("validation", osl_obs)
  assert not group.enable_corruption
  lag = osl_obs.validation_delay_steps
  for name in ("actpack", "loadcell"):
    term = group.terms[name]
    assert term.history_length == 4 and term.noise is None
    assert not term.params["randomize"]
    assert (term.delay_min_lag, term.delay_max_lag) == (lag, lag)
  assert group.terms["actpack"].params["applied_torque"]
  assert not group.terms["loadcell"].params["socket"]
  # The training env's config is left as it was.
  assert train.observations["osl"] == osl_observation_group("train", osl_obs)


##
# GPU smoke test (slow; needs a library and a stage-1 run).
##

LIBRARY_ENV_VAR = "MJLAB_OSL_UNICYCLE_LIBRARY"
STAGE1_RUN_ENV_VAR = "MJLAB_OSL_SPLIT_STAGE1_RUN"


@pytest.mark.slow
@pytest.mark.skipif(
  not (os.environ.get(LIBRARY_ENV_VAR) and os.environ.get(STAGE1_RUN_ENV_VAR)),
  reason=f"Set {LIBRARY_ENV_VAR} and {STAGE1_RUN_ENV_VAR} (a stage-1 run, "
  "checkpoint or converted warm start) to run the stage-2 smoke test.",
)
def test_stage2_smoke_on_gpu(tmp_path: Path) -> None:
  """Two stage-2 iterations of the real task: warm start, the zero-shot and a
  regular validation, checkpoints with their exports, then play's load."""
  if not torch.cuda.is_available():
    pytest.skip("Needs a GPU.")
  device = "cuda:0"
  ws_path = tmp_path / "ws.pt"
  ws = load_warm_start(os.environ[STAGE1_RUN_ENV_VAR])
  ws.save(ws_path)
  lineage = ws.lineage
  task = split_task_id(lineage["model"], lineage["socket"], lineage["reward"])
  library = os.environ[LIBRARY_ENV_VAR]

  def make_env(play: bool, num_envs: int) -> RslRlVecEnvWrapper:
    env_cfg = load_env_cfg(task, play=play)
    env_cfg.scene.num_envs = num_envs
    motion = env_cfg.commands["motion"]
    assert isinstance(motion, GaitLibraryCommandCfg)
    motion.library_dir = library
    return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  agent = load_rl_cfg(task)
  assert isinstance(agent, SplitRunnerCfg)
  agent.num_steps_per_env = 8
  agent.algorithm.num_mini_batches = 2
  agent.max_iterations = 2
  agent.save_interval = 1
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.freeze.human_actor_iterations = 1
  agent.freeze.critic_warmup_iterations = 0
  agent.validation.interval = 1
  agent.validation.num_envs = 4
  agent.validation.num_steps = 20
  agent.warm_start.path = str(ws_path)

  env = make_env(play=False, num_envs=8)
  log_dir = tmp_path / "run"
  runner = SplitOnPolicyRunner(env, asdict(agent), str(log_dir), device)
  runner.learn(agent.max_iterations)
  runner.close()
  start = int(lineage["iter"]) + 1
  for it in (start, start + 1):
    assert (log_dir / f"model_{it}.pt").is_file()
    assert (log_dir / "exported" / f"osl_policy_{it}.onnx").is_file()
  # Logged at every validation (the episode length only once one finishes).
  tags = (
    "Validation/early_termination_rate",
    "ValidationExpected/early_termination_rate",
  )
  for steps in _scalars(log_dir, *tags).values():
    assert set(steps) == {start, start + 1}  # Zero-shot and k = 0, then k = 1.
  env.close()

  play_env = make_env(play=True, num_envs=2)
  played = SplitOnPolicyRunner(play_env, asdict(load_rl_cfg(task)), device=device)
  played.load(
    str(log_dir / f"model_{start + 1}.pt"),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = played.get_inference_policy(device=device)
  obs = play_env.get_observations()
  with torch.inference_mode():
    for _ in range(5):
      obs, _, _, _ = play_env.step(policy(obs))
  for group in ("actor", "osl"):
    assert bool(torch.isfinite(obs_group(obs, group)).all())
  play_env.close()
