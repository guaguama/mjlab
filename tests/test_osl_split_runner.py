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
* the OSL foot touch sites: recorded in every checkpoint, and required to match
  the env's by the warm start, a resume and play;
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
import re
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
  TOUCH_SITES,
  Model,
  OslEntityCfg,
  Socket,
  TouchSites,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import TendonEffortAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.osl_gait_library.config.rl_cfg import osl_gait_library_runner_cfg
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_gait_library.rl.runner import GaitLibraryOnPolicyRunner
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.mdp.rewards import mimic_reward
from mjlab.tasks.osl_mimic.rl import runner as mimic_runner
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel, RunningMeanStd
from mjlab.tasks.osl_mimic.touch_sites import set_touch_sites
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
from mjlab.tasks.osl_split.rl.export import (
  export_osl_policy,
  osl_policy_metadata,
  term_channels,
)
from mjlab.tasks.osl_split.rl.layout import ActionLayout
from mjlab.tasks.osl_split.rl.policy import (
  SplitInferencePolicy,
  distribution_std,
  std_parameter,
)
from mjlab.tasks.osl_split.rl.runner import AgentRewards, SplitOnPolicyRunner
from mjlab.tasks.osl_split.rl.schedules import ChannelSchedule
from mjlab.tasks.osl_split.rl.storage import obs_group
from mjlab.tasks.osl_split.rl.warmstart import (
  LINEAGE_KEYS,
  WARM_START_FORMAT,
  WarmStart,
  library_release,
  lineage_touch_sites,
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
  + ("-MSK" if model == "msk" else "-ROBOT")
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
  expected rewards are recorded. Its config builds the foot ``touch_sites`` (a
  freshly configured run's ``feet_extended`` by default)."""

  def __init__(
    self,
    num_envs: int = NUM_ENVS,
    seed: int = 0,
    touch_sites: TouchSites = "feet_extended",
  ) -> None:
    self.num_envs = num_envs
    self.num_actions = HUMAN_DIM + 2
    self.device = "cpu"
    self.max_episode_length = 1000
    self.step_dt = 0.01
    self.physics_dt = 0.002
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self.cfg: Any = SimpleNamespace(
      scale_rewards_by_dt=False, commands={}, osl_touch_sites=touch_sites
    )
    self.action_manager = _action_manager()
    self.reward_manager = _FakeRewardManager(num_envs)
    self.observation_manager = _observation_manager()
    # A tight left arc and a right pivot (wz > 0 turns left).
    clips = SimpleNamespace(
      clip_groups=["fwd_tight", "turn"],
      num_clips=2,
      twists=torch.tensor([[0.5, 0.0, 0.6], [0.0, 0.0, -0.5]]),
    )
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
      "split_leg_contact_force": 100.0 * self._rand(n),
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


class _GlobalRngEnv(_FakeEnv):
  """The fake env drawing from torch's global generators, as mjlab's commands and
  events do; it records a draw made while it is built."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    self.built_with = float(torch.rand(1))
    super().__init__(*args, **kwargs)

  def _rand(self, *shape: int) -> torch.Tensor:
    return torch.rand(*shape)

  def get_observations(self) -> TensorDict:
    n = self.num_envs
    return TensorDict(
      {
        "actor": torch.randn(n, ACTOR_DIM),
        "osl": 2.0 * torch.randn(n, OSL_DIM) + 1.0,
      },
      batch_size=[n],
    )


class _Runner(SplitOnPolicyRunner):
  """The runner with fake evaluation envs (the real ones are mjlab envs)."""

  def _make_validation_env(self) -> Any:
    return _FakeEnv(num_envs=4, seed=1)

  def _make_stress_env(self) -> Any:
    return _FakeEnv(num_envs=5, seed=2)


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
    "osl_touch_sites": "feet_extended",  # As the fake env builds by default.
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
  cfg["validation"].update(interval=2, num_steps=5, num_envs=4, stress_num_steps=4)
  cfg["freeze"].update(human_actor_iterations=2, critic_warmup_iterations=1)
  # The blend is on by default, with a ramp longer than this two-iteration freeze.
  cfg["blend"]["enabled"] = False
  cfg["channels"].update(anneal_start_iteration=0, anneal_iterations=2)
  cfg["channels"]["additive_term_weights"] = {"twist": TWIST_SPLIT}
  cfg["warm_start"]["path"] = str(warm_start) if warm_start is not None else ""
  return cfg


def _scalars(log_dir: Path, *tags: str) -> dict[str, dict[int, float]]:
  """``{tag: {step: value}}`` from the run's TensorBoard events."""
  events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  events.Reload()
  return {tag: {e.step: e.value for e in events.Scalars(tag)} for tag in tags}


def _tags(log_dir: Path) -> set[str]:
  """Every scalar tag the run logged."""
  events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  events.Reload()
  return set(events.Tags()["scalars"])


def _states(runner: SplitOnPolicyRunner) -> dict[str, dict[str, torch.Tensor]]:
  return {name: model.state_dict() for name, model in runner.alg.models.items()}


##
# Registration.
##


def test_split_tasks_are_registered() -> None:
  ids = sorted(t for t in list_tasks() if t.startswith("Mjlab-Split-"))
  tags = [
    m + r + s
    for m in ("-ROBOT", "-MSK")
    for r in ("", "-CLF")
    for s in ("", "-Compliant")
  ]
  assert ids == sorted(SPLIT_TASKS) == sorted(TASK_PREFIX + t for t in tags)
  for task_id, (model, socket, reward) in SPLIT_TASKS.items():
    assert split_task_id(model, socket, reward) == task_id
    assert load_runner_cls(task_id) is SplitOnPolicyRunner
    # The experiment name follows the id's tags: -MSK-CLF -> _msk_clf. The robot's
    # tag is in the id only (its log directories keep the untagged name).
    suffix = task_id.removeprefix(TASK_PREFIX).removeprefix("-ROBOT")
    suffix = suffix.lower().replace("-", "_")
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
    # The model tag names the robot in the task id only: its runs' log
    # directories keep the untagged experiment name.
    suffix = suffix.removeprefix("-ROBOT")
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
  # How to run it: the mean, the reset conventions, CPU MuJoCo's solver caps.
  assert meta["default_inference"] == "mean"
  assert "first frame" in meta["reset"]["history"]
  assert meta["reset"]["last_action"].startswith("0")
  deployment = meta["deployment"]
  assert deployment["solver"] == {"iterations": 10, "ls_iterations": 20}
  assert deployment["training_solver"] is None  # The fake env has no sim config.

  # With the real training config, the OSL motors' delay and strength ranges, and
  # the training solver caps (mujoco_warp's 4/8) raised for CPU MuJoCo.
  env.cfg = osl_split_env_cfg("msk", "compliant", "clf", "train")
  real = osl_policy_metadata(cast(Any, env), cast(Any, alg), None)
  randomization = real["randomization"]
  assert randomization["osl_actuator_delay_s"] == pytest.approx([0.0, 0.02])
  assert randomization["osl_motor_strength"] == pytest.approx([0.9, 1.1])
  assert real["deployment"]["training_solver"] == {"iterations": 4, "ls_iterations": 8}
  assert real["deployment"]["solver"] == {"iterations": 10, "ls_iterations": 20}
  env.cfg.sim.mujoco.iterations = 12  # A higher training cap is kept.
  solver = osl_policy_metadata(cast(Any, env), cast(Any, alg), None)["deployment"]
  assert solver["solver"] == {"iterations": 12, "ls_iterations": 20}

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
  socket_only = term_channels("loadcell", {"ankle": False})
  assert len(socket_only) == osl_loadcell_dim(OslObsCfg(ankle_load_cell=False)) == 6
  assert [c for c, _, _ in socket_only] == [
    c for c, _, _ in term_channels("loadcell", {})[6:]
  ]
  with pytest.raises(ValueError, match="At least one load cell"):
    OslObsCfg(ankle_load_cell=False, socket_load_cell=False)
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
  cfg["blend"].update(enabled=True, ramp_iterations=1)  # Warm-up 1 + 1 <= freeze 2.
  cfg["validation"]["stress_interval"] = 2
  env = _FakeEnv()
  runner = _Runner(cast(Any, env), cfg, str(log_dir), "cpu")
  assert runner.stage_start == runner.current_learning_iteration == STAGE_START
  assert runner.schedule.blend
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
  # ...and the evaluation envs' before each evaluation: validation and stress at
  # zero-shot, at k = 1, and at the final k = 2 (off the interval-2 grid).
  validation_env = cast(_FakeEnv, runner._validation_env)
  assert validation_env.reward_manager.mimic.weights_set == weights
  stress_env = cast(_FakeEnv, runner._stress_env)
  assert stress_env.reward_manager.mimic.weights_set == weights

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
    "ValidationTeacher/mean_episode_return_osl",
    "ValidationTeacherExpected/mean_episode_return_human",
    "Validation/split_reward_osl/turn",
    "Validation/early_termination_rate/fwd_tight_left",
    "Validation/early_termination_rate/turn_right",
    "Validation/split_leg_contact_force_peak",
    "Stress/survival_rate",
    "Stress/mean_first_episode_return_osl",
    "StressOslSampled/survival_rate",
    "StressTeacher/survival_rate",
    "Perf/zero_shot_validation_time",
    "Perf/zero_shot_stress_time",
    "Perf/validation_time",
    "Perf/stress_time",
    "Policy/std/muscles",
    "Policy/std/osl_motors",
    "Policy/mean_outside/osl_motors",
    "Policy/noise_gap/muscles",
    "Critic/human/explained_variance",
    "Critic/osl/explained_variance",
    "Stage2/blend_beta",
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
    "Validation/split_reward_osl/turn",
    "Validation/early_termination_rate/fwd_tight_left",
    "Validation/early_termination_rate/turn_right",
    "Validation/split_leg_contact_force_peak",
  ):
    # Zero-shot, k = 1, then the final checkpoint.
    assert set(logged[tag]) == set(steps), tag
  # The teacher's families at zero-shot only: the stage-1 policy, raw and
  # noise-averaged (validation), raw (stress)...
  for tag in (
    "ValidationTeacher/mean_episode_return_osl",
    "ValidationTeacherExpected/mean_episode_return_human",
    "StressTeacher/survival_rate",
    "Perf/zero_shot_validation_time",
    "Perf/zero_shot_stress_time",
  ):
    assert set(logged[tag]) == {steps[0]}, tag
  # ...and the stress families of the OSL actor at its interval (k = 1) and the
  # final checkpoint (k = 2) only: the fresh actor at zero-shot would fail at the
  # floor.
  for tag in (
    "Stress/survival_rate",
    "Stress/mean_first_episode_return_osl",
    "StressOslSampled/survival_rate",
    "Perf/validation_time",
    "Perf/stress_time",
  ):
    assert set(logged[tag]) == {steps[1], steps[2]}, tag
  tags = _tags(log_dir)
  assert not any(tag.startswith("ValidationExpected/") for tag in tags)
  assert not any(tag.startswith("ValidationOslSampled/") for tag in tags)
  for tag in ("Policy/std/muscles", "Critic/osl/explained_variance"):
    assert set(logged[tag]) == set(steps), tag
  assert logged["Policy/std/osl_motors"][steps[0]] == pytest.approx(0.2)
  # The blend: the teacher drives the OSL through k = 1, the actor from k = 2.
  assert list(logged["Stage2/blend_beta"].values()) == [0.0, 0.0, 1.0]

  # A save at k = 2 only (k = 0, the launch's first iteration, is skipped), with
  # the OSL actor's export.
  it = steps[2]
  assert (log_dir / f"model_{it}.pt").is_file()
  assert (log_dir / "exported" / f"osl_policy_{it}.onnx").is_file()
  assert (log_dir / "exported" / f"osl_policy_{it}.json").is_file()
  for it in steps[:2]:
    assert not (log_dir / f"model_{it}.pt").exists()
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
  """:func:`_train_cfg` without validation, stress validation or ONNX exports."""
  cfg = _train_cfg(warm_start)
  cfg["validation"].update(interval=0, stress_interval=0)
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


def _touch_runner(touch_sites: TouchSites) -> _Runner:
  """A runner without a warm start, on an env that builds ``touch_sites``."""
  env = _FakeEnv(touch_sites=touch_sites)
  return _Runner(cast(Any, env), _quick_cfg(None), None, "cpu")


def test_checkpoints_record_and_require_the_touch_sites(tmp_path: Path) -> None:
  """Every checkpoint records the OSL foot touch sites its env simulates, under the
  top-level ``osl_touch_sites``. One loaded into an env that builds the other
  zones is refused, by a resume and by play's load of the actors alone, naming the
  flag that builds the trained zones: the human actor's touch inputs would read
  what they never saw. A checkpoint that records none was saved before the setting
  and trained with ``original``. Play builds the checkpoint's zones itself, and a
  matching checkpoint loads and records its value again."""
  ws = _write_warm_start(tmp_path / "ws.pt")
  checkpoints: dict[Path, TouchSites] = {}
  for touch_sites in TOUCH_SITES:
    runner = _touch_runner(touch_sites)
    runner.warm_start(
      replace(ws, lineage={**ws.lineage, "osl_touch_sites": touch_sites})
    )
    path = tmp_path / f"model_{touch_sites}.pt"
    runner.save(str(path))
    assert torch.load(path, weights_only=False)["osl_touch_sites"] == touch_sites
    checkpoints[path] = touch_sites
  # The env's zones, also from a runner that holds no lineage to copy them from.
  _touch_runner("feet_extended").save(str(tmp_path / "bare.pt"))
  bare = torch.load(tmp_path / "bare.pt", weights_only=False)
  assert bare["osl_touch_sites"] == "feet_extended"
  legacy = torch.load(tmp_path / "model_original.pt", weights_only=False)
  del legacy["osl_touch_sites"]
  torch.save(legacy, tmp_path / "model_legacy.pt")
  checkpoints[tmp_path / "model_legacy.pt"] = "original"

  for path, trained in checkpoints.items():
    flag = re.escape(f"--env.osl-touch-sites {trained}")
    for load_cfg in (None, {"actor": True}):  # A resume, then play's load.
      for other in TOUCH_SITES:
        if other != trained:
          with pytest.raises(ValueError, match=flag):
            _touch_runner(other).load(str(path), load_cfg=load_cfg)

    # play.py makes the task's config build the checkpoint's zones, then loads.
    cfg = osl_split_env_cfg("msk", "compliant", "clf", "play")
    SplitOnPolicyRunner.configure_env_for_checkpoint(cfg, path)
    assert cfg.osl_touch_sites == trained
    played = _touch_runner(cfg.osl_touch_sites)
    played.load(str(path), load_cfg={"actor": True}, strict=True, map_location="cpu")
    assert played.stage_start == STAGE_START

    resumed = _touch_runner(trained)
    resumed.load(str(path))
    assert resumed.stage_start == STAGE_START
    resumed.save(str(tmp_path / "again.pt"))
    again = torch.load(tmp_path / "again.pt", weights_only=False)
    assert again["osl_touch_sites"] == trained


def test_warm_start_needs_the_stage1_touch_sites(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Stage 2 builds the OSL foot touch sites stage 1 trained with: the human actor
  reads the touch sensors, and its normalizer's single count would never adapt to
  other touch readings. A lineage that records none (converted before the
  setting) trained with ``original``: refused by the default ``feet_extended``
  env, naming the flag, and accepted by an ``original`` one. A heel_extended
  stage 1 is refused by a feet_extended stage 2 and vice versa. The runner reports
  the zones stage 1 trained with."""
  ws = _write_warm_start(tmp_path / "ws.pt")
  assert ws.lineage["osl_touch_sites"] == "feet_extended"
  legacy = replace(
    ws, lineage={k: v for k, v in ws.lineage.items() if k != "osl_touch_sites"}
  )
  heel = replace(ws, lineage={**ws.lineage, "osl_touch_sites": "heel_extended"})
  # (the warm start, stage 2's zones, stage 1's zones)
  mismatches: tuple[tuple[WarmStart, TouchSites, TouchSites], ...] = (
    (legacy, "feet_extended", "original"),
    (ws, "original", "feet_extended"),
    (heel, "feet_extended", "heel_extended"),
    (ws, "heel_extended", "feet_extended"),
  )
  for stage1, stage2, trained in mismatches:
    refused = _touch_runner(stage2)
    with pytest.raises(ValueError, match=re.escape(f"--env.osl-touch-sites {trained}")):
      refused.warm_start(stage1)
    assert not refused._ready

  capsys.readouterr()
  started = _touch_runner("original")
  started.warm_start(legacy)
  assert started._ready and started.stage_start == STAGE_START
  out = capsys.readouterr().out
  assert "[INFO] OSL touch sites: original (not recorded" in out
  started = _touch_runner("feet_extended")
  started.warm_start(ws)
  assert started._ready and started.stage_start == STAGE_START
  assert "[INFO] OSL touch sites: feet_extended\n" in capsys.readouterr().out


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
    human_critic_lr=2e-4,
    osl_critic_lr=5e-4,
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
  # anneal from iteration 10, default rates and the twist counted fully in both.
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
    "schedule.critic_ramp: checkpoint 1, config 10",
    "schedule.human_critic_lr",
    "schedule.osl_critic_lr",
    "schedule.human_peak_lr",
    "schedule.human_ramp",
    "channel_schedule.anneal_start: checkpoint 0, config 10",
    "channel_schedule.anneal_iterations: checkpoint 2, config 150",
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

  def build(cfg: Any, device: str, **kwargs: Any) -> _FakeEnv:  # ManagerBasedRlEnv.
    del device, kwargs
    built.append(cfg)
    return _FakeEnv(num_envs=cfg.scene.num_envs, seed=1)

  # The runners build their evaluation envs in stage 1's base runner.
  monkeypatch.setattr(mimic_runner, "ManagerBasedRlEnv", build)
  monkeypatch.setattr(mimic_runner, "RslRlVecEnvWrapper", lambda env: env)
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

  # Every command-line setting of the run reaches validation: the OSL foot's touch
  # sites (built by the robot entity), the simulation, every shared reward term's
  # weight and params, and the OSL lag (the middle of the training range).
  train = osl_split_env_cfg("msk", "compliant", "clf", "train")
  set_touch_sites(train, "original")  # --env.osl-touch-sites original
  train.sim.mujoco.iterations = 7
  train.rewards["twist"].weight = 0.35
  train.rewards["twist"].params["std"] = 0.2
  for name in ("actpack", "loadcell"):
    train.observations["osl"].terms[name].delay_max_lag = 4
  cfg = validation_cfg(train)
  assert cfg.osl_touch_sites == "original"
  robot = cfg.scene.entities["robot"]
  assert isinstance(robot, OslEntityCfg) and robot.touch_sites == "original"
  assert cfg.sim.mujoco.iterations == 7
  assert cfg.rewards["twist"].weight == 0.35
  assert cfg.rewards["twist"].params["std"] == 0.2
  for name in ("actpack", "loadcell"):
    term = cfg.observations["osl"].terms[name]
    assert (term.delay_min_lag, term.delay_max_lag) == (2, 2)
    assert not term.params["randomize"]
  # Validation keeps its own mode: terminations and commands.
  assert "root_deviation" not in cfg.terminations
  assert not cfg.observations["actor"].enable_corruption

  # Stress validation plays a copy of the training config: the OSL sensor noise
  # and every randomization stay. Only the run's seed goes: an env built with one
  # re-seeds every generator, so the stress seed would not set its draws.
  train.seed = 7
  stress = runner._eval_env_cfg("stress")
  assert stress.seed is None and train.seed == 7
  robot = stress.scene.entities["robot"]
  assert isinstance(robot, OslEntityCfg) and robot.touch_sites == "original"
  assert replace(stress, seed=7) == train and stress is not train
  assert stress.observations["osl"].terms["actpack"].params["randomize"]
  assert stress.observations["osl"].terms["actpack"].delay_max_lag == 4


def test_evaluations_are_guarded_and_timed(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """A failing evaluation (here every validation and stress pass, as an
  out-of-memory error would) prints a warning and training goes on; each
  evaluation's wall time is logged."""
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)
  cfg = _train_cfg(ws_path)
  cfg["validation"].update(stress_interval=2)
  log_dir = tmp_path / "run"
  runner = _Runner(cast(Any, _FakeEnv()), cfg, str(log_dir), "cpu")

  def fail(it: int, zero_shot: bool = False) -> dict[str, float]:
    raise RuntimeError("CUDA out of memory")

  runner.validate = fail  # type: ignore[method-assign]
  runner.stress_validate = fail  # type: ignore[method-assign]

  # As on a GPU: the memory is logged every iteration, and once more after the
  # last save and video.
  def memory() -> dict[str, float]:
    return {"Perf/gpu_max_allocated_gib": 1.5}

  runner._gpu_memory = memory  # type: ignore[method-assign]
  capsys.readouterr()
  runner.learn(ITERATIONS)
  out = capsys.readouterr().out
  steps = [STAGE_START + k for k in range(ITERATIONS)]
  for name, it in (
    ("zero_shot_validation", steps[0]),
    ("zero_shot_stress", steps[0]),
    ("validation", steps[1]),
    ("stress", steps[1]),
    # The final checkpoint's (k = 2, off the interval-2 grid), guarded alike.
    ("validation", steps[2]),
    ("stress", steps[2]),
  ):
    assert f"[WARN] {name} at it={it} failed (training continues)" in out, name
  assert (log_dir / f"model_{steps[-1]}.pt").is_file()  # Training finished.
  logged = _scalars(
    log_dir,
    "Perf/zero_shot_validation_time",
    "Perf/zero_shot_stress_time",
    "Perf/validation_time",
    "Perf/stress_time",
    "Perf/video_time",
  )
  assert set(logged["Perf/zero_shot_validation_time"]) == {steps[0]}
  assert set(logged["Perf/validation_time"]) == set(logged["Perf/stress_time"])
  assert set(logged["Perf/stress_time"]) == {steps[1], steps[2]}
  assert set(logged["Perf/video_time"]) == {steps[2]}  # The k = 2 save.
  events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  events.Reload()
  memory_steps = [e.step for e in events.Scalars("Perf/gpu_max_allocated_gib")]
  assert memory_steps == [*steps, steps[-1]]
  # CPU: no GPU memory to log.
  assert SplitOnPolicyRunner._gpu_memory(runner) == {}


@pytest.mark.parametrize("iterations, evaluated", [(0, []), (2, [1]), (3, [1, 2])])
def test_the_final_checkpoint_is_evaluated_once(
  tmp_path: Path, iterations: int, evaluated: list[int]
) -> None:
  """With an interval of 2, a launch ending off the grid (k = 2) is validated and
  stress-validated there once more, before its final save; one ending on it (k = 1)
  is not evaluated twice, and a launch of no iteration is not evaluated at all."""
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)
  cfg = _train_cfg(ws_path)
  cfg["validation"].update(stress_interval=2, zero_shot=False)
  cfg["save_interval"] = 100  # No periodic save at k = 2 before the final one.
  log_dir = tmp_path / "run"
  runner = _Runner(cast(Any, _FakeEnv()), cfg, str(log_dir), "cpu")
  calls: list[tuple[str, int, bool]] = []

  def spy(kind: str) -> Any:
    def record(it: int, zero_shot: bool = False) -> dict[str, float]:
      saved = (log_dir / f"model_{it}.pt").exists()
      calls.append((kind, it - STAGE_START, saved))
      return {}

    return record

  runner.validate = spy("validation")
  runner.stress_validate = spy("stress")
  runner.learn(iterations)
  assert calls == [
    (kind, k, False) for k in evaluated for kind in ("validation", "stress")
  ]
  if iterations:
    assert (log_dir / f"model_{STAGE_START + iterations - 1}.pt").is_file()


def test_validation_is_seeded_and_leaves_training_rng(tmp_path: Path) -> None:
  """Every validation family, and the validation env's build, draws from
  ``validation.seed``: the same weights score the same, and training's random
  state is untouched."""
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)

  class Runner(_Runner):
    def _make_validation_env(self) -> Any:
      return _GlobalRngEnv(num_envs=4, seed=1)

  cfg = _train_cfg(ws_path)
  cfg["validation"]["inference"] = ("mean", "osl_sampled")
  runner = Runner(cast(Any, _FakeEnv()), cfg, None, "cpu")
  torch.manual_seed(123)
  state = torch.get_rng_state()
  first = runner.validate(STAGE_START + 1)
  assert torch.equal(torch.get_rng_state(), state)
  assert first["Validation/early_termination_rate"] > 0.0  # The env drew dones.
  assert runner.validate(STAGE_START + 1) == first
  assert torch.equal(torch.get_rng_state(), state)
  env = cast(_GlobalRngEnv, runner._validation_env)
  torch.manual_seed(0)
  assert env.built_with == float(torch.rand(1))
  runner.validation_cfg["seed"] = 1
  assert runner.validate(STAGE_START + 1) != first


def test_video_plays_the_human_action_by_its_std(tmp_path: Path) -> None:
  """``video.inference`` "auto" is stage 1's rule on the human's muscle std: the
  noise-averaged action above a median of 0.3, the raw mean at the recipe's 0.07;
  the OSL actor plays its mean. Other modes are refused at construction."""
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)  # A random human std around 1.
  runner = _Runner(cast(Any, _FakeEnv()), _quick_cfg(ws_path), None, "cpu")
  env = cast(Any, _FakeEnv())
  assert runner._video_inference(env) == "expected"
  with torch.no_grad():
    std_parameter(runner.alg.human_actor).fill_(math.log(0.07))
  assert runner._video_inference(env) == "mean"
  with runner._video_policy(env) as policy:
    assert isinstance(policy, SplitInferencePolicy)
    assert (policy.human_mode, policy.osl_mode) == ("mean", "mean")
  runner.video_cfg["inference"] = "expected"
  assert runner._video_inference(env) == "expected"
  cfg = _quick_cfg(ws_path)
  cfg["video"]["inference"] = "sampled"
  with pytest.raises(ValueError, match="video.inference"):
    _Runner(cast(Any, _FakeEnv()), cfg, None, "cpu")


def test_stress_pass_counts_each_env_first_episode() -> None:
  """Env 1 terminates at step 2 and env 2 is truncated at step 3 (both then
  auto-reset and keep running); envs 0 and 3 survive the whole window. Each agent's
  first-episode return sums its reward over the env's first episode only."""
  n, num_steps = 4, 5
  runner = SplitOnPolicyRunner.__new__(SplitOnPolicyRunner)
  runner.validation_cfg = {"stress_num_steps": num_steps}
  runner.device = "cpu"
  step_count = 0
  term = SimpleNamespace(components={"err_site_abs": torch.zeros(n)})

  class Rewards:
    def __init__(self) -> None:
      self.term = term

    def __call__(self) -> torch.Tensor:  # Human 1, OSL 2 per step.
      return torch.tensor([[1.0, 2.0]]).expand(n, -1)

  def reset() -> tuple[TensorDict, dict]:
    return TensorDict({"actor": torch.zeros(n, 1)}, batch_size=[n]), {}

  def step(actions: torch.Tensor):
    nonlocal step_count
    step_count += 1
    dones, time_outs = torch.zeros(n), torch.zeros(n)
    if step_count == 2:
      dones[1] = 1.0
    if step_count == 3:
      dones[2] = time_outs[2] = 1.0
    if step_count == 4:  # Second episodes count for nothing.
      dones[1] = dones[2] = 1.0
    term.components["err_site_abs"] = torch.arange(n, dtype=torch.float32)
    # The hardest strike of a first episode is env 3's; env 1's is in its second.
    force = torch.zeros(n)
    force[1] = 500.0 if step_count == 4 else 0.0
    force[3] = 80.0 if step_count == 5 else 0.0
    term.components["split_leg_contact_force"] = force
    obs = TensorDict({"actor": torch.zeros(n, 1)}, batch_size=[n])
    return obs, torch.full((n,), 0.5), dones, {"time_outs": time_outs}

  mimic = SimpleNamespace(func=term)
  manager = SimpleNamespace(active_terms=["mimic"], get_term_cfg=lambda name: mimic)
  env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.01, reward_manager=manager)
  vec = SimpleNamespace(unwrapped=env, device="cpu", reset=reset, step=step)

  def policy(obs: TensorDict) -> torch.Tensor:
    return torch.zeros(n, HUMAN_DIM + 2)

  metrics = runner._split_stress_pass(
    cast(Any, vec), cast(Any, policy), cast(Any, Rewards())
  )
  assert metrics["survival_rate"] == pytest.approx(3 / 4)
  lengths = torch.tensor([5.0, 2.0, 3.0, 5.0])  # First-episode lengths.
  assert metrics["mean_first_episode_s"] == pytest.approx(float(lengths.mean()) * 0.01)
  assert metrics["mean_first_episode_return"] == pytest.approx(0.5 * 15 / 4)
  assert metrics["mean_first_episode_return_human"] == pytest.approx(15 / 4)
  assert metrics["mean_first_episode_return_osl"] == pytest.approx(2.0 * 15 / 4)
  assert metrics["err_site_abs"] == pytest.approx((1.5 * 2 + 5 / 3 + 1.5 * 2) / 5)
  assert metrics["split_leg_contact_force_peak"] == 80.0


def test_validation_pass_group_means_and_peaks() -> None:
  """The per-group diagnostics are each group's mean over its envs at every step
  (weighted sums, no boolean indexing), and the leg-contact force's peak is the
  largest over every env and step."""
  n, num_steps = 6, 3
  runner = SplitOnPolicyRunner.__new__(SplitOnPolicyRunner)
  runner.validation_cfg = {"num_steps": num_steps}
  runner.device = "cpu"
  masks = {
    "left": torch.tensor([True, True, False, False, False, False]),
    "right": torch.tensor([False, False, True, False, False, True]),
  }
  runner._validation_groups = lambda env: masks
  values = torch.arange(num_steps * n, dtype=torch.float32).reshape(num_steps, n)
  forces = torch.zeros(num_steps, n)
  forces[1, 4] = 250.0
  term = SimpleNamespace(components={})
  step_count = 0

  class Rewards:
    def __init__(self) -> None:
      self.term = term

    def __call__(self) -> torch.Tensor:
      return torch.ones(n, 2)

  def reset() -> tuple[TensorDict, dict]:
    return TensorDict({"actor": torch.zeros(n, 1)}, batch_size=[n]), {}

  def step(actions: torch.Tensor):
    nonlocal step_count
    term.components = {
      "err_site_abs": values[step_count],
      "split_leg_contact_force": forces[step_count],
    }
    step_count += 1
    obs = TensorDict({"actor": torch.zeros(n, 1)}, batch_size=[n])
    return obs, torch.ones(n), torch.zeros(n), {"time_outs": torch.zeros(n)}

  mimic = SimpleNamespace(func=term)
  manager = SimpleNamespace(active_terms=["mimic"], get_term_cfg=lambda name: mimic)
  env = SimpleNamespace(num_envs=n, device="cpu", reward_manager=manager)
  vec = SimpleNamespace(unwrapped=env, device="cpu", reset=reset, step=step)

  def policy(obs: TensorDict) -> torch.Tensor:
    return torch.zeros(n, HUMAN_DIM + 2)

  metrics = runner._validation_pass(
    cast(Any, vec), cast(Any, policy), cast(Any, Rewards())
  )
  for group, mask in masks.items():
    expected = float(values[:, mask].mean(dim=1).mean())
    assert metrics[f"err_site_abs/{group}"] == pytest.approx(expected), group
  assert metrics["err_site_abs"] == pytest.approx(float(values.mean()))
  assert metrics["split_leg_contact_force_peak"] == 250.0
  assert metrics["split_leg_contact_force"] == pytest.approx(250.0 / (n * num_steps))


def test_zero_shot_families() -> None:
  """The families each evaluation plays: at zero-shot the fresh OSL actor's raw
  mean and the teacher's (raw and noise-averaged in validation, raw in stress);
  then the configured ones. The teacher's are refused anywhere else."""
  runner = SplitOnPolicyRunner.__new__(SplitOnPolicyRunner)
  defaults = asdict(osl_split_runner_cfg("msk", "compliant", "clf", "test"))
  runner.validation_cfg = defaults["validation"]
  assert runner._families("validation", zero_shot=True) == [
    "mean",
    "teacher",
    "teacher_expected",
  ]
  assert runner._families("stress", zero_shot=True) == ["teacher"]
  assert runner._families("validation", zero_shot=False) == ["mean"]
  assert runner._families("stress", zero_shot=False) == ["mean", "osl_sampled"]
  runner.validation_cfg = {**defaults["validation"], "teacher": False}
  assert runner._families("stress", zero_shot=True) == []
  assert runner._families("validation", zero_shot=True) == ["mean"]
  for setting in ("zero_shot_stress_inference", "inference"):
    cfg = _quick_cfg(None)
    cfg["validation"][setting] = ("teacher_expected",)
    with pytest.raises(ValueError, match="zero-shot only"):
      _Runner(cast(Any, _FakeEnv()), cfg, None, "cpu")


def test_validation_groups_by_turn_direction() -> None:
  """Stage 1's families plus the tight arcs and pivots by direction; env ``i``
  plays clip ``i % num_clips``."""
  twists = torch.tensor(
    [
      [0.0, 0.0, 0.0],  # idle
      [0.5, 0.0, 0.8],  # tight left arc
      [0.5, 0.0, -0.8],  # tight right arc
      [0.5, 0.0, 0.3],  # gentle arc
      [0.0, 0.0, 0.5],  # left pivot
      [0.0, 0.0, -0.5],  # right pivot
      [-0.5, 0.0, 0.8],  # backward arc
    ]
  )
  from mjlab.tasks.osl_gait_library.mdp import clip_group

  cmd = SimpleNamespace(
    clip_groups=[clip_group(t.numpy()) for t in twists],
    num_clips=len(twists),
    twists=twists,
  )
  env = SimpleNamespace(
    num_envs=9, device="cpu", command_manager=SimpleNamespace(get_term=lambda n: cmd)
  )
  runner = SplitOnPolicyRunner.__new__(SplitOnPolicyRunner)
  groups = {
    name: mask.nonzero().squeeze(-1).tolist()
    for name, mask in runner._validation_groups(cast(Any, env)).items()
  }
  assert groups["fwd_tight"] == [1, 2, 8] and groups["turn"] == [4, 5]
  assert groups["fwd_tight_left"] == [1, 8] and groups["fwd_tight_right"] == [2]
  assert groups["turn_left"] == [4] and groups["turn_right"] == [5]
  assert groups["fwd"] == [3] and groups["bck"] == [6]


def test_resume_takes_the_config_curriculum(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """--agent.resume-curriculum config: the resumed run trains on the command line's
  curriculum, here a longer schedule and other rates, and says what changed."""
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)
  cfg = _quick_cfg(ws_path)
  cfg["algorithm"]["schedule_iterations"] = ITERATIONS
  runner = _Runner(cast(Any, _FakeEnv()), cfg, str(tmp_path / "run"), "cpu")
  runner.learn(ITERATIONS)
  checkpoint = tmp_path / "run" / f"model_{STAGE_START + ITERATIONS - 1}.pt"
  k = ITERATIONS  # Past the first run's schedule: its rates sit at their floors.
  assert runner.schedule.lr("osl_actor", k) == pytest.approx(0.1 * 4e-4)

  resumed_cfg = _quick_cfg(None)
  resumed_cfg["resume_curriculum"] = "config"
  resumed_cfg["algorithm"].update(schedule_iterations=20, osl_actor_lr=2e-4)
  resumed_cfg["channels"]["anneal_iterations"] = 5
  resumed = _Runner(cast(Any, _FakeEnv(seed=1)), resumed_cfg, None, "cpu")
  capsys.readouterr()
  resumed.load(str(checkpoint))
  out = capsys.readouterr().out
  assert "takes the config's curriculum" in out
  assert "schedule.total_iterations: checkpoint 3, config 20" in out
  assert "schedule.osl_actor_lr: checkpoint 0.0004, config 0.0002" in out
  assert "channel_schedule.anneal_iterations: checkpoint 2, config 5" in out
  assert f"osl_actor rate at k={k}: 4e-05 -> " in out
  assert resumed.schedule_iterations == 20
  assert resumed.channel_schedule.anneal_iterations == 5
  assert resumed.schedule.lr("osl_actor", k) == pytest.approx(
    2e-4 * (0.1 + 0.9 * (1.0 - k / 20))
  )
  resumed.learn(1)
  assert resumed.alg.lrs["osl_actor"] == resumed.schedule.lr("osl_actor", k)

  # A config that freezes the human again is applied, with a warning.
  refrozen_cfg = _quick_cfg(None)
  refrozen_cfg["resume_curriculum"] = "config"
  refrozen_cfg["freeze"]["human_actor_iterations"] = 50
  refrozen = _Runner(cast(Any, _FakeEnv(seed=2)), refrozen_cfg, None, "cpu")
  capsys.readouterr()
  refrozen.load(str(checkpoint))
  assert "freezes the human actor again at k=3" in capsys.readouterr().out
  assert refrozen.alg.human_actor_frozen


def _provenance_lineage(**changes: Any) -> dict[str, Any]:
  lineage = {
    **_lineage(),
    "human_std": {"min": 0.07, "median": 0.07, "max": 0.07},
    "stage1_lr": 3e-5,
    "stage1_std_cap": 0.07,
    "stage1_env": {
      "clf_pen_w": 0.0,
      "twist_weight": 0.2,
      "root_deviation": 2.0,
      "env_stages_in_force": {},
    },
    "stage1_planned_last_iteration": STAGE1_ITER,
    "library_release": {"release": "unicycle_v001", "sha256": "0" * 64},
  }
  lineage.update(changes)
  return lineage


def test_warm_start_provenance(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Stage 1's recorded end state against stage 2's settings: another CLF penalty
  weight is refused unless allowed; other env settings, another library, a large
  human std, an unfinished run and a human rate above stage 1's last are
  reported."""
  library = tmp_path / "unicycle_v001"
  library.mkdir()
  (library / "library.json").write_text('{"release": "unicycle_v001"}')
  digest = library_release(library)
  assert digest is not None and digest["sha256"] is not None
  ws = _write_warm_start(tmp_path / "ws.pt")
  train = osl_split_env_cfg("msk", "compliant", "clf", "train")
  motion = train.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = str(library)

  def runner(allow: bool = False) -> _Runner:
    env = _FakeEnv()
    env.cfg = train  # The real training config, for its env settings.
    cfg = _quick_cfg(None)
    cfg["warm_start"]["allow_pen_w_mismatch"] = allow
    return _Runner(cast(Any, env), cfg, None, "cpu")

  # A matching warm start reports its end state and the same library release.
  capsys.readouterr()
  runner().warm_start(replace(ws, lineage=_provenance_lineage(library_release=digest)))
  out = capsys.readouterr().out
  assert "stage 1 at its end: learning rate 3e-05, std cap 0.07" in out
  assert "same library release unicycle_v001" in out
  assert "[WARN]" not in out

  lineage = _provenance_lineage(
    stage1_env={
      "clf_pen_w": 0.25,
      "twist_weight": 0.06,
      "root_deviation": 1.0,
      "env_stages_in_force": {"rewards.twist.weight": 0.06},
    },
    stage1_lr=1e-5,
    human_std={"min": 0.5, "median": 0.8, "max": 1.0},
    stage1_planned_last_iteration=STAGE1_ITER + 100,
  )
  with pytest.raises(ValueError, match="pen_w 0.25; stage 2's is 0"):
    runner().warm_start(replace(ws, lineage=lineage))
  capsys.readouterr()
  allowed = runner(allow=True)
  allowed.warm_start(replace(ws, lineage=lineage))
  out = capsys.readouterr().out
  assert allowed._ready
  for text in (
    "Allowed (warm_start.allow_pen_w_mismatch)",
    "twist weight 0.06; stage 2 uses 0.2",
    "root-deviation limit 1; stage 2 uses 2",
    "unicycle_v001, but their library.json differ",  # Same name, other bytes.
    "The human's peak rate 3e-05 exceeds stage 1's last rate 1e-05",
    "median std is 0.8",
    f"set to end at {STAGE1_ITER + 100}",
  ):
    assert text in out, text
  # A reward switch starts the critics fresh: another pen_w is only reported.
  capsys.readouterr()
  runner().warm_start(replace(ws, lineage={**lineage, "reward": "mimic"}))
  assert "The reward switches" in capsys.readouterr().out


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
  """Two stage-2 iterations of the real task: warm start, the zero-shot and regular
  validations and stress validations (the teacher included), checkpoints with
  their exports, then play's load."""
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
    # Stage 1's touch sites, as the pipeline passes --env.osl-touch-sites.
    set_touch_sites(env_cfg, lineage_touch_sites(lineage))
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
  agent.blend.ramp_iterations = 1
  agent.validation.interval = 1
  agent.validation.num_envs = 4
  agent.validation.num_steps = 20
  agent.validation.stress_interval = 1
  agent.validation.stress_num_envs = 4
  agent.validation.stress_num_steps = 20
  agent.warm_start.path = str(ws_path)
  # Any stage-1 run will do for plumbing, whatever its CLF penalty weight.
  agent.warm_start.allow_pen_w_mismatch = True

  env = make_env(play=False, num_envs=8)
  log_dir = tmp_path / "run"
  runner = SplitOnPolicyRunner(env, asdict(agent), str(log_dir), device)
  runner.learn(agent.max_iterations)
  runner.close()
  start = int(lineage["iter"]) + 1
  # No checkpoint at the launch's first iteration; the second is the last.
  assert not (log_dir / f"model_{start}.pt").exists()
  assert (log_dir / f"model_{start + 1}.pt").is_file()
  assert (log_dir / "exported" / f"osl_policy_{start + 1}.onnx").is_file()
  # Logged at every validation (the episode length only once one finishes).
  logged = _scalars(
    log_dir,
    "Validation/early_termination_rate",
    "Stress/survival_rate",
    "StressOslSampled/survival_rate",
    "ValidationTeacher/early_termination_rate",
    "ValidationTeacherExpected/early_termination_rate",
    "StressTeacher/survival_rate",
  )
  for tag in (
    "Validation/early_termination_rate",  # Zero-shot and k = 0, then k = 1.
    "Stress/survival_rate",  # k = 0, then k = 1.
    "StressOslSampled/survival_rate",
  ):
    assert set(logged[tag]) == {start, start + 1}, tag
  for tag in (
    "ValidationTeacher/early_termination_rate",
    "ValidationTeacherExpected/early_termination_rate",
    "StressTeacher/survival_rate",
  ):
    assert set(logged[tag]) == {start}, tag  # Zero-shot only.
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
