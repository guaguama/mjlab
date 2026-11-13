"""Contracts for stage 1.5's runner, runner settings and task registration
(``tasks/osl_fsm/rl/runner.py``, ``config/rl_cfg.py``, ``config/__init__.py``).

CPU only and without an mjlab env. The runner runs on stage 2's runner fakes
(``test_osl_split_runner``) with the controller's real action term in the
``osl_motors`` slot, on fake sensors over the real compiled model
(``test_osl_fsm_env``), so its settings and diagnostics are the real ones:

* the registration: eight tasks, the runner class, and runner settings that are
  stage 2's but for the listed changes; the launch's command line parses, and
  settings that need an OSL actor fail there, before an env is built;
* the runner end to end: the warm start, the human learning from the first
  iteration on stage 2's rates with its std held, zeros in the env's OSL rows, the
  human's scalars only, the controller's diagnostics in validation and stress
  validation (zero-shot included), checkpoints that record the controller, a resume
  and play's load of the actor alone;
* the controller a checkpoint records: refused when the env runs another, built by
  play's configure step; other stages' checkpoints refused with the way forward;
* the evaluation envs' configs: the training controller, no OSL motor delay, the
  clean ``osl`` group;
* a clean import in the walker's order.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import pytest
import torch
import tyro
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict
from test_osl_fsm_env import _delays, _fake_env, _scene_model
from test_osl_split_runner import (
  HIDDEN,
  HUMAN_DIM,
  ITERATIONS,
  NUM_ENVS,
  NUM_STEPS,
  STAGE1_COUNTER,
  STAGE1_ENV_STEPS,
  STAGE_START,
  TWIST_SPLIT,
  _FakeEnv,
  _FakeMuscles,
  _scalars,
  _tags,
  _write_warm_start,
)

import mjlab
from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OslEntityCfg, TouchSites
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_fsm.config import (
  FSM_TASKS,
  fsm_experiment_name,
  fsm_task_id,
)
from mjlab.tasks.osl_fsm.config.env_cfgs import (
  OslFsmCfg,
  OslFsmEnvCfg,
  clean_osl_group,
  fsm_settings,
  osl_fsm_env_cfg,
)
from mjlab.tasks.osl_fsm.config.rl_cfg import (
  FAMILIES,
  FsmRunnerCfg,
  check_fsm_runner_cfg,
  osl_fsm_runner_cfg,
)
from mjlab.tasks.osl_fsm.mdp import FSM_DIAGNOSTICS, OslFsmActionCfg
from mjlab.tasks.osl_fsm.rl.policy import HumanOnlyInferencePolicy
from mjlab.tasks.osl_fsm.rl.ppo import ALGORITHM, HumanOnlyPPO
from mjlab.tasks.osl_fsm.rl.runner import (
  FSM_INFO_KEY,
  FsmSplitRunner,
  fsm_differences,
  recorded_fsm_settings,
)
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.touch_sites import set_touch_sites
from mjlab.tasks.osl_split.config import (
  SPLIT_TASKS,
  split_experiment_name,
  split_task_id,
)
from mjlab.tasks.osl_split.config.rl_cfg import osl_split_runner_cfg
from mjlab.tasks.osl_split.config.sensors import (
  ANKLE_FORCE,
  OSL_SENSOR_SPECS,
  scene_sensor_name,
)
from mjlab.tasks.osl_split.mdp.rewards import stage1_channel_weights
from mjlab.tasks.osl_split.rl.schedules import SplitSchedule
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

RUN_TASK = "Mjlab-FSM-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
"""The task of the first stage-1.5 run."""
SDK = OslFsmCfg(
  rules="sdk", swing_load_exit=False, body_mass=80.0, initial_state="e_stance"
)
"""A controller other than the task's default in every setting a run must repeat."""
STATE_SHARES = [
  f"fsm_{state}" for state in ("e_stance", "l_stance", "e_swing", "l_swing")
]
LAUNCH = (
  "--env.osl-touch-sites",
  "original",
  "--env.osl-fsm.rules",
  "myosuite",
  "--env.commands.motion.library-dir",
  "/libraries/unicycle_v001",
  "--env.commands.motion.sampling",
  "continuous",
  "--env.commands.motion.fwd-arc-regions",
  "(((0.5,1.5),(0.0,0.5),0.5),((0.6,1.2),(0.5,1.5),0.5))",
  "--agent.warm-start.path",
  "/runs/2026-11-09_14-58-51_tune2_C4b_tight",
  "--agent.warm-start.checkpoint",
  "model_5998.pt",
  "--agent.max-iterations",
  "1000",
)
"""The first run's command line (its paths are only parsed here)."""

##
# Fakes.
##


@functools.cache
def _model() -> mujoco.MjModel:
  """The musculoskeletal compliant robot with the OSL sensors, compiled once: the
  controller takes its tare, motor gains and body mass from it."""
  return _scene_model("msk", "compliant")


class _FsmEnv(_FakeEnv):
  """Stage 2's fake env with the controller in its ``osl_motors`` slot: the task's
  own ``OslFsmAction`` on fake sensors, which every step fills with random readings
  (loads up to 1.2 body weights) before the controller runs, so it changes state. A
  reset resets the term, as the action manager does. The OSL rows of every action
  the env receives are recorded."""

  def __init__(
    self,
    num_envs: int = NUM_ENVS,
    seed: int = 0,
    touch_sites: TouchSites = "heel_extended",
    fsm: OslFsmCfg | None = None,
    controller: str = "fsm",
  ) -> None:
    super().__init__(num_envs, seed, touch_sites)
    term_cfg = osl_fsm_env_cfg(fsm=fsm).actions["osl_motors"]
    assert isinstance(term_cfg, OslFsmActionCfg)
    self._sensor_env = _fake_env(_model(), num_envs)
    self.osl_term = replace(term_cfg, controller=controller).build(self._sensor_env)
    self.osl_term.reset(None)
    terms = {"osl_motors": self.osl_term, "muscles": _FakeMuscles(HUMAN_DIM)}
    self.action_manager = SimpleNamespace(
      active_terms=list(terms), get_term=terms.__getitem__
    )
    self.osl_rows: list[torch.Tensor] = []

  def _read_sensors(self) -> None:
    n, g, term = self.num_envs, self._generator, self.osl_term
    scene = self._sensor_env.scene
    for spec in OSL_SENSOR_SPECS:
      scene[scene_sensor_name(spec.name)].data.copy_(
        torch.randn(n, spec.dim, generator=g)
      )
    load = 1.2 * term.body_weight_n * torch.rand(n, generator=g)
    scene[scene_sensor_name(ANKLE_FORCE)].data[:, 1] = term.tare_n - load

  def reset(self) -> tuple[TensorDict, dict]:
    self.osl_term.reset(None)
    return super().reset()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    self.osl_rows.append(actions[:, :2].clone())
    self._read_sensors()
    self.osl_term.process_actions(actions[:, :2])
    obs, reward, dones, extras = super().step(actions)
    ended = dones.nonzero().squeeze(-1)
    if ended.numel():
      self.osl_term.reset(ended)
    return obs, reward, dones, extras


class _FsmRunner(FsmSplitRunner):
  """The runner with fake evaluation envs (the real ones are mjlab envs)."""

  def _make_validation_env(self) -> Any:
    return _FsmEnv(num_envs=4, seed=1)

  def _make_stress_env(self) -> Any:
    return _FsmEnv(num_envs=5, seed=2)


def _train_cfg(warm_start: Path | None) -> dict[str, Any]:
  """The real MSK-CLF-compliant stage-1.5 runner config, shrunk as stage 2's tests
  shrink theirs. The OSL critic's block keeps its default: the runner gives it the
  human critic's."""
  cfg = asdict(osl_fsm_runner_cfg("msk", "compliant", "clf", "test"))
  for key in ("actor", "critic"):
    cfg[key]["hidden_dims"] = HIDDEN
  cfg["num_steps_per_env"] = NUM_STEPS
  cfg["algorithm"]["num_mini_batches"] = 2
  cfg["max_iterations"] = ITERATIONS
  cfg["save_interval"] = 2
  cfg["logger"] = "tensorboard"
  cfg["upload_model"] = False
  cfg["validation"].update(interval=2, num_steps=5, num_envs=4, stress_num_steps=4)
  cfg["channels"]["additive_term_weights"] = {"twist": TWIST_SPLIT}
  cfg["warm_start"]["path"] = str(warm_start) if warm_start is not None else ""
  return cfg


def _quick_cfg(warm_start: Path | None) -> dict[str, Any]:
  """:func:`_train_cfg` without validation or stress validation."""
  cfg = _train_cfg(warm_start)
  cfg["validation"].update(interval=0, stress_interval=0)
  return cfg


def _bare_runner(
  fsm: OslFsmCfg | None = None,
  controller: str = "fsm",
  touch_sites: TouchSites = "heel_extended",
) -> _FsmRunner:
  """A runner without a warm start (it saves, loads and plays, but cannot train
  from scratch) on an env whose controller has ``fsm``'s settings."""
  env = _FsmEnv(fsm=fsm, controller=controller, touch_sites=touch_sites)
  return _FsmRunner(cast(Any, env), _quick_cfg(None), None, "cpu")


def _cli(task: str, *args: str) -> TrainConfig:
  """The configs train.py's command line builds for ``task``."""
  return tyro.cli(
    TrainConfig,
    args=list(args),
    default=TrainConfig.from_task(task),
    config=mjlab.TYRO_FLAGS,
  )


def _flat(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
  """``tree``'s leaves under dotted keys."""
  flat: dict[str, Any] = {}
  for key, value in tree.items():
    if isinstance(value, Mapping):
      flat.update(_flat(value, f"{prefix}{key}."))
    else:
      flat[f"{prefix}{key}"] = value
  return flat


##
# Registration and settings.
##


def test_fsm_tasks_are_registered() -> None:
  """Stage 2's eight combinations under their own prefix and experiment names, each
  with the stage-1.5 runner, its runner settings and the controller's env."""
  ids = sorted(task for task in list_tasks() if task.startswith("Mjlab-FSM-"))
  assert ids == sorted(FSM_TASKS) and len(ids) == 8
  assert sorted(FSM_TASKS.values()) == sorted(SPLIT_TASKS.values())
  for task_id, (model, socket, reward) in FSM_TASKS.items():
    assert fsm_task_id(model, socket, reward) == task_id
    stage2 = split_task_id(model, socket, reward)
    assert task_id == stage2.replace("Mjlab-Split-", "Mjlab-FSM-")
    experiment = fsm_experiment_name(model, socket, reward)
    stage2_experiment = split_experiment_name(model, socket, reward)
    assert experiment == stage2_experiment.replace("_split_", "_fsm_")
    assert load_runner_cls(task_id) is FsmSplitRunner

    rl_cfg = load_rl_cfg(task_id)
    assert isinstance(rl_cfg, FsmRunnerCfg)
    assert rl_cfg == osl_fsm_runner_cfg(model, socket, reward, experiment)
    assert rl_cfg.experiment_name == experiment
    assert resolve_callable(rl_cfg.algorithm.class_name) is HumanOnlyPPO
    validation = rl_cfg.validation
    assert (validation.model, validation.socket, validation.reward) == (
      model,
      socket,
      reward,
    )

    for play in (False, True):
      env_cfg = load_env_cfg(task_id, play=play)
      assert isinstance(env_cfg, OslFsmEnvCfg)
      assert env_cfg.osl_fsm == OslFsmCfg()
      term = env_cfg.actions["osl_motors"]
      assert isinstance(term, OslFsmActionCfg) and term.controller == "fsm"
      assert env_cfg.observations["osl"] == clean_osl_group()


@pytest.mark.parametrize("reward", ["clf", "mimic"])
def test_runner_settings_are_stage2s_for_the_human_alone(reward: Reward) -> None:
  """Every setting is stage 2's (the warm start, the human's 3e-5 rate on its
  50-iteration ramp with its std held, the critic's 1e-4 floored at 3e-5 and its
  10-iteration ramp, the PPO constants, the evaluations' cadence) except: the
  human-only algorithm, no freeze, no blend, no OSL export, both channels held at
  the stage-1 weights, the human's observation sets, the human's raw mean in every
  evaluation without a teacher, and the tags."""
  fsm = asdict(osl_fsm_runner_cfg("msk", "compliant", reward, "test"))
  split = asdict(osl_split_runner_cfg("msk", "compliant", reward, "test"))
  ours, theirs = _flat(fsm), _flat(split)
  unset = object()
  differ = {
    key for key in {*ours, *theirs} if ours.get(key, unset) != theirs.get(key, unset)
  }
  stage1 = stage1_channel_weights(reward == "clf")["osl"]
  annealed = split["channels"]["osl_end"]
  assert differ == {
    "algorithm.class_name",
    "validation.teacher",
    "validation.stress_inference",
    "validation.zero_shot_stress_inference",
    "freeze.human_actor_iterations",
    "blend.enabled",
    "export_osl_onnx",
    "wandb_tags",
    "obs_groups.osl_actor",
    "obs_groups.osl_critic",
    *(f"channels.osl_end.{g}" for g, w in stage1.items() if w != annealed[g]),
  }
  channels = fsm["channels"]
  for end in ("human_start", "human_end", "osl_start", "osl_end"):
    assert channels[end] == stage1, end
  assert fsm["obs_groups"] == {"actor": ("actor",), "critic": ("actor",)}
  assert fsm["wandb_tags"] == (
    "osl_fullbody",
    "msk",
    reward,
    "unicycle",
    "fsm",
    "stage1.5",
  )
  assert (fsm["max_iterations"], fsm["save_interval"]) == (1000, 250)
  validation = fsm["validation"]
  for setting in ("inference", "zero_shot_inference", "stress_inference"):
    assert validation[setting] == ("mean",), setting
  assert validation["zero_shot_stress_inference"] == ("mean",)
  assert not validation["teacher"] and not fsm["export_osl_onnx"]

  # The schedule the runner builds: the human learns from k = 0, ramping to its
  # held peak; the critic ramps over the warm-up and decays to its floor.
  schedule = SplitSchedule.from_train_cfg(
    fsm, total_iterations=1000, reward_switch=False
  )
  assert (schedule.human_freeze, schedule.critic_warmup, schedule.blend) == (
    0,
    10,
    False,
  )
  assert schedule.lr("human_actor", 0) == pytest.approx(3e-5 / 50)
  assert schedule.lr("human_actor", 49) == schedule.lr("human_actor", 999) == 3e-5
  assert schedule.lr("human_critic", 0) == pytest.approx(1e-5)
  assert schedule.lr("human_critic", 9) == pytest.approx(1e-4 * schedule.decay(9))
  assert schedule.lr("human_critic", 999) == pytest.approx(3e-5)
  assert not fsm["algorithm"]["human_learn_std"]

  # A bare config holds stage 1.5's own defaults, so it passes its own checks.
  check_fsm_runner_cfg(asdict(FsmRunnerCfg()))
  assert FsmRunnerCfg().freeze.human_actor_iterations == 0


def test_the_launch_command_line_parses() -> None:
  """The first run's command line builds the stage-1.5 configs it means: the
  controller with the myosuite rules, the original touch sites, the library and its
  sampling, the warm start and the run length."""
  args = _cli(RUN_TASK, *LAUNCH)
  env, agent = args.env, args.agent
  assert isinstance(env, OslFsmEnvCfg) and isinstance(agent, FsmRunnerCfg)
  assert env.osl_fsm == OslFsmCfg(rules="myosuite")
  term = env.actions["osl_motors"]
  assert isinstance(term, OslFsmActionCfg)
  assert (term.rules, term.swing_load_exit, term.body_mass, term.initial_state) == (
    "myosuite",
    None,
    None,
    "pose",
  )
  robot = env.scene.entities["robot"]
  assert isinstance(robot, OslEntityCfg)
  assert env.osl_touch_sites == robot.touch_sites == "original"
  assert _delays(env) == {"osl": (0, 0), "human": (10, 20)}
  motion = env.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  assert motion.library_dir == "/libraries/unicycle_v001"
  assert motion.sampling == "continuous"
  assert motion.fwd_arc_regions == (
    ((0.5, 1.5), (0.0, 0.5), 0.5),
    ((0.6, 1.2), (0.5, 1.5), 0.5),
  )
  assert agent.warm_start.checkpoint == "model_5998.pt"
  assert agent.warm_start.path.endswith("C4b_tight")
  assert agent.max_iterations == 1000
  assert agent == replace(
    load_rl_cfg(RUN_TASK), warm_start=agent.warm_start, max_iterations=1000
  )


@pytest.mark.parametrize(
  ("args", "flag"),
  [
    (("--agent.blend.enabled", "True"), "--agent.blend.enabled True"),
    (("--agent.validation.teacher", "True"), "--agent.validation.teacher True"),
    (("--agent.export-osl-onnx", "True"), "--agent.export-osl-onnx True"),
    (
      ("--agent.validation.stress-inference", "('mean','osl_sampled')"),
      "--agent.validation.stress-inference names ['osl_sampled']",
    ),
    (("--agent.algorithm.osl-actor-lr", "0.001"), "osl_actor_lr = 0.001"),
  ],
)
def test_settings_that_need_an_osl_actor_fail_on_the_command_line(
  args: tuple[str, str], flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
  """Refused while tyro builds the configs, before any env exists."""
  with pytest.raises(SystemExit) as error:
    _cli(RUN_TASK, *args)
  assert error.value.code == 2
  err = " ".join(capsys.readouterr().err.replace("│", " ").split())
  assert flag in err


REFUSED: list[tuple[str, str, Any, str]] = [
  ("validation", "inference", ("mean", "osl_sampled"), "inference names"),
  ("validation", "zero_shot_inference", ("teacher",), "zero-shot-inference names"),
  ("validation", "stress_inference", "osl_sampled", "stress-inference names"),
  ("validation", "zero_shot_stress_inference", ("teacher_expected",), "stress-inf"),
  ("validation", "teacher", True, "--agent.validation.teacher True"),
  ("blend", "enabled", True, "--agent.blend.enabled True"),
  ("", "export_osl_onnx", True, "--agent.export-osl-onnx True"),
]
"""``(section, key, value, part of the message)``: a setting that needs an OSL
actor (``""``: a top-level one)."""


@pytest.mark.parametrize(("section", "key", "value", "flag"), REFUSED)
def test_settings_that_need_an_osl_actor_are_refused(
  section: str, key: str, value: Any, flag: str
) -> None:
  """By the config's construction and by the runner on a dict config, before it
  builds anything, each naming the flag. Only the human's own families pass."""
  default = osl_fsm_runner_cfg("msk", "compliant", "clf", "test")
  with pytest.raises(ValueError, match=re.escape(flag)):
    if section:
      changed = replace(getattr(default, section), **{key: value})
      replace(default, **{section: changed})
    else:
      replace(default, **{key: value})
  cfg = _quick_cfg(None)
  (cfg[section] if section else cfg)[key] = value
  with pytest.raises(ValueError, match=re.escape(flag)):
    _FsmRunner(cast(Any, _FsmEnv()), cfg, None, "cpu")
  if section == "validation" and key != "teacher":
    check_fsm_runner_cfg({"validation": {key: FAMILIES}})


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
  cfg["validation"]["stress_interval"] = 2
  env = _FsmEnv()
  runner = _FsmRunner(cast(Any, env), cfg, str(log_dir), "cpu")
  assert isinstance(runner.alg, HumanOnlyPPO)
  assert runner.stage_start == runner.current_learning_iteration == STAGE_START
  assert env.common_step_counter == STAGE1_COUNTER
  assert runner.lineage is not None and not runner.reward_switch
  # Stage 2's warm start checked the OSL critic's block: it is the human critic's.
  assert runner.cfg["osl_critic"] == runner.cfg["critic"]
  schedule = runner.schedule
  assert (schedule.human_freeze, schedule.blend) == (0, False)

  recorded: list[torch.Tensor] = []
  process_env_step = runner.alg.process_env_step

  def spy(obs: TensorDict, rewards: torch.Tensor, *args: Any) -> None:
    recorded.append(rewards.clone())
    process_env_step(obs, rewards, *args)

  monkeypatch.setattr(runner.alg, "process_env_step", spy)
  runner.learn(ITERATIONS)

  # The human learned from its own reward, its channel plus the twist term (of the
  # agents' pair HumanOnlyPPO takes the human's column)...
  assert len(recorded) == ITERATIONS * NUM_STEPS
  for got, expected in zip(recorded, env.expected_agent_rewards, strict=True):
    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-6)
  # ...and wrote zeros in the OSL rows, which the controller ignores anyway.
  rows = torch.stack(env.osl_rows)
  assert rows.shape == (ITERATIONS * NUM_STEPS, NUM_ENVS, 2) and not rows.any()

  steps = [STAGE_START + k for k in range(ITERATIONS)]
  diagnostics = [
    f"{prefix}/{name}"
    for prefix in ("Validation", "Stress")
    for name in FSM_DIAGNOSTICS
  ]
  logged = _scalars(
    log_dir,
    "Stage2/iteration",
    "Stage2/human_frozen",
    "Stage2/reward_switch",
    "LR/human_actor",
    "LR/human_critic",
    "Loss/learning_rate",
    "Policy/human/mean_std",
    "Policy/std/muscles",
    "Critic/human/explained_variance",
    "Channels/osl/osl_joints",
    "Train/total_env_steps",
    "Train/mean_reward",
    "Train/human/mean_reward",
    "Stage2/skipped_steps/human_actor",
    "Validation/mean_episode_return_human",
    "Validation/early_termination_rate/fwd_tight_left",
    "Stress/survival_rate",
    "Perf/zero_shot_validation_time",
    "Perf/zero_shot_stress_time",
    *diagnostics,
  )
  assert logged["Stage2/iteration"] == {s: float(k) for k, s in enumerate(steps)}
  # The human learns from the first iteration, on stage 2's ramp.
  assert list(logged["Stage2/human_frozen"].values()) == [0.0] * ITERATIONS
  for k, step in enumerate(steps):
    for module in ("human_actor", "human_critic"):
      lr = logged[f"LR/{module}"][step]
      assert lr == pytest.approx(schedule.lr(module, k)), (module, k)
  assert logged["LR/human_actor"][steps[0]] == pytest.approx(3e-5 / 50)
  assert logged["Loss/learning_rate"] == logged["LR/human_actor"]
  # The std stays the warm start's; the OSL channel never anneals.
  assert len(set(logged["Policy/human/mean_std"].values())) == 1
  assert len(set(logged["Policy/std/muscles"].values())) == 1
  assert list(logged["Channels/osl/osl_joints"].values()) == [0.0] * ITERATIONS
  samples = NUM_ENVS * NUM_STEPS
  assert list(logged["Train/total_env_steps"].values()) == [
    STAGE1_ENV_STEPS + samples * (k + 1) for k in range(ITERATIONS)
  ]
  for tag in ("Train/mean_reward", "Train/human/mean_reward", "Policy/std/muscles"):
    assert set(logged[tag]) == set(steps), tag
  # Validation and stress validation at zero-shot (the stage-1 human beside the
  # controller) and at k = 1, the controller's diagnostics among them.
  for tag in (
    "Validation/mean_episode_return_human",
    "Validation/early_termination_rate/fwd_tight_left",
    "Stress/survival_rate",
    *diagnostics,
  ):
    assert set(logged[tag]) == {steps[0], steps[1]}, tag
  for prefix in ("Validation", "Stress"):
    for step in (steps[0], steps[1]):
      shares = [logged[f"{prefix}/{name}"][step] for name in STATE_SHARES]
      assert all(0.0 <= share <= 1.0 for share in shares), (prefix, shares)
      assert sum(shares) == pytest.approx(1.0, abs=1e-5)  # One state per step.
      assert 0.0 < logged[f"{prefix}/fsm_load_bw"][step] < 1.2
  for tag in ("Perf/zero_shot_validation_time", "Perf/zero_shot_stress_time"):
    assert set(logged[tag]) == {steps[0]}, tag
  # The human's scalars only: nothing of an OSL actor, its critic, the blend or the
  # teacher, and no family that needs them.
  tags = _tags(log_dir)
  for tag in (
    "Train/osl/mean_reward",
    "Train/osl/teacher_rms",
    "LR/osl_actor",
    "LR/osl_critic",
    "Policy/osl/mean_std",
    "Policy/osl/std_knee",
    "Policy/std/osl_motors",
    "Stage2/osl_actor_frozen",
    "Stage2/blend_beta",
    "Critic/osl/explained_variance",
  ):
    assert tag not in tags, tag
  for prefix in ("Teacher", "OslSampled", "Expected"):
    assert not any(
      tag.startswith((f"Validation{prefix}/", f"Stress{prefix}/")) for tag in tags
    ), prefix

  # A save at k = 2 only, recording the controller, with nothing exported.
  checkpoint = log_dir / f"model_{steps[2]}.pt"
  assert checkpoint.is_file() and not (log_dir / "exported").exists()
  for it in steps[:2]:
    assert not (log_dir / f"model_{it}.pt").exists()
  saved = torch.load(checkpoint, weights_only=False)
  assert saved["algorithm"] == ALGORITHM
  assert (
    saved["infos"][FSM_INFO_KEY] == fsm_settings(env) == recorded_fsm_settings(saved)
  )
  assert saved["osl_touch_sites"] == "heel_extended"

  # A resumed run (train.py: construct without a warm start, then load) continues
  # after the checkpoint on its schedule.
  resumed_env = _FsmEnv()
  resumed = _FsmRunner(
    cast(Any, resumed_env), _train_cfg(None), str(tmp_path / "resumed"), "cpu"
  )
  with pytest.raises(RuntimeError, match="--agent.warm-start.path"):
    resumed.learn(1)
  infos = resumed.load(str(checkpoint))
  assert infos[FSM_INFO_KEY] == fsm_settings(env)
  assert resumed.current_learning_iteration == steps[2] + 1
  assert resumed.stage_start == STAGE_START and resumed.lineage == runner.lineage
  assert resumed.alg.stage_iteration == ITERATIONS - 1
  assert resumed_env.common_step_counter == env.common_step_counter
  for name, state in runner.alg.models.items():
    theirs = resumed.alg.models[name].state_dict()
    for key, value in state.state_dict().items():
      assert torch.equal(value, theirs[key]), (name, key)
  resumed.learn(1)
  k = ITERATIONS
  assert resumed.alg.lrs["human_actor"] == schedule.lr("human_actor", k)
  assert not torch.stack(resumed_env.osl_rows).any()

  # Play: the actor alone, with play.py's arguments, then the human's mean with
  # zeros in the OSL rows.
  played = _FsmRunner(cast(Any, _FsmEnv()), _train_cfg(None), None, "cpu")
  played.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu"
  )
  policy = played.get_inference_policy(device="cpu")
  assert isinstance(policy, HumanOnlyInferencePolicy)
  obs = env.get_observations()
  with torch.inference_mode():
    actions = policy(obs)
    assert actions.shape == (NUM_ENVS, HUMAN_DIM + 2) and not actions[:, :2].any()
    assert torch.equal(actions, runner.alg.get_policy()(obs))
  with pytest.raises(RuntimeError, match="Nothing to train from"):
    played.learn(1)


##
# The controller a checkpoint records.
##


def test_checkpoints_record_and_require_their_controller(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """A checkpoint records the controller settings its env ran. A resume or play
  under others is refused, naming the flags that build the checkpoint's; play's
  configure step builds them itself (it takes no --env overrides). The sdk rules'
  default swing exit and an explicit True are one controller."""
  path = tmp_path / "model_sdk.pt"
  saver = _bare_runner(SDK)
  saver.save(str(path))
  stored = torch.load(path, weights_only=False)["infos"][FSM_INFO_KEY]
  assert stored == fsm_settings(saver.env)
  assert [stored[key] for key in ("rules", "swing_load_exit", "body_mass")] == [
    "sdk",
    False,
    80.0,
  ]
  assert stored["initial_state"] == "e_stance" and stored["controller"] == "fsm"
  assert stored["body_weight_n"] == pytest.approx(80.0 * 9.81)

  for load_cfg in (None, {"actor": True}):  # A resume, then play's load.
    with pytest.raises(ValueError, match="another OSL controller") as error:
      _bare_runner().load(str(path), load_cfg=load_cfg)
    message = str(error.value)
    for flag in (
      "--env.osl-fsm.rules sdk",
      "--env.osl-fsm.swing-load-exit False",
      "--env.osl-fsm.body-mass 80.0",
      "--env.osl-fsm.initial-state e_stance",
    ):
      assert flag in message, flag
    assert "--env.actions.osl-motors.controller" not in message  # It agrees.
    # The relaunch flags build the checkpoint's controller: all four, since with the
    # sdk rules an unset swing exit would resolve to on.
    relaunch = message.split("Relaunch with ")[1].removesuffix(".").split()
    rebuilt = _cli(RUN_TASK, *relaunch).env
    assert isinstance(rebuilt, OslFsmEnvCfg) and rebuilt.osl_fsm == SDK

  # play.py makes the task's config build the checkpoint's controller, then loads.
  cfg = osl_fsm_env_cfg("msk", "compliant", "clf", "play")
  capsys.readouterr()
  FsmSplitRunner.configure_env_for_checkpoint(cfg, path)
  assert (
    "[INFO] OSL controller: sdk rules, swing load exit off" in capsys.readouterr().out
  )
  assert cfg.osl_fsm == SDK
  term = cfg.actions["osl_motors"]
  assert isinstance(term, OslFsmActionCfg)
  assert (term.rules, term.swing_load_exit, term.body_mass, term.initial_state) == (
    "sdk",
    False,
    80.0,
    "e_stance",
  )
  assert cfg.osl_touch_sites == "heel_extended"  # Stage 2's step ran too.
  played = _bare_runner(cfg.osl_fsm)
  played.load(str(path), load_cfg={"actor": True}, strict=True, map_location="cpu")
  resumed = _bare_runner(SDK)
  resumed.load(str(path))
  assert resumed._ready

  # One controller under two spellings: the sdk rules' swing exit is on by default.
  default_exit = tmp_path / "default_exit.pt"
  _bare_runner(OslFsmCfg(rules="sdk")).save(str(default_exit))
  explicit = _bare_runner(OslFsmCfg(rules="sdk", swing_load_exit=True))
  recorded = recorded_fsm_settings(torch.load(default_exit, weights_only=False))
  assert recorded is not None and recorded["swing_load_exit"] is True
  assert fsm_differences(recorded, fsm_settings(explicit.env)) == {}
  explicit.load(str(default_exit))
  assert fsm_differences(stored, fsm_settings(explicit.env)) == {
    "swing_load_exit": (False, True),
    "body_mass": (80.0, None),
    "initial_state": ("e_stance", "pose"),
  }

  # The term's controller choice is part of it.
  policy_path = tmp_path / "model_policy.pt"
  _bare_runner(controller="policy").save(str(policy_path))
  flag = re.escape("--env.actions.osl-motors.controller policy")
  with pytest.raises(ValueError, match=flag) as error:
    _bare_runner().load(str(policy_path))
  relaunch = str(error.value).split("Relaunch with ")[1].removesuffix(".").split()
  rebuilt = _cli(RUN_TASK, *relaunch).env.actions["osl_motors"]
  assert isinstance(rebuilt, OslFsmActionCfg) and rebuilt.controller == "policy"
  cfg = osl_fsm_env_cfg("msk", "compliant", "clf", "play")
  FsmSplitRunner.configure_env_for_checkpoint(cfg, policy_path)
  term = cfg.actions["osl_motors"]
  assert isinstance(term, OslFsmActionCfg) and term.controller == "policy"
  _bare_runner(controller="policy").load(str(policy_path), load_cfg={"actor": True})


def test_other_checkpoints_are_refused_with_the_way_forward(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Stage 2's checkpoints, stage 1's and warm-start files are not stage 1.5's,
  and neither is a human-only checkpoint without the controller settings. Play's
  configure step leaves the controller of a checkpoint that records none, and the
  load then says what the file is."""
  runner = _bare_runner()
  others = {
    "stage-2 (two-agent) checkpoint": {
      "human_actor_state_dict": {},
      "osl_actor_state_dict": {},
      "iter": 5,
      "infos": {},
    },
    "stage-1 checkpoint": {"actor_state_dict": {}, "iter": 610, "infos": None},
    "warm-start file": {"format": "mjlab.osl_split.warm_start/1"},
  }
  for what, payload in others.items():
    path = tmp_path / f"{what.split()[0]}.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="not a stage-1.5 checkpoint") as error:
      runner.load(str(path))
    assert what in str(error.value)
    assert "--agent.warm-start.path" in str(error.value)

  good = tmp_path / "good.pt"
  runner.save(str(good))
  unrecorded = torch.load(good, weights_only=False)
  del unrecorded["infos"][FSM_INFO_KEY]
  torch.save(unrecorded, tmp_path / "unrecorded.pt")
  with pytest.raises(ValueError, match="records no OSL controller settings"):
    runner.load(str(tmp_path / "unrecorded.pt"))

  cfg = osl_fsm_env_cfg("msk", "compliant", "clf", "play", fsm=SDK)
  capsys.readouterr()
  FsmSplitRunner.configure_env_for_checkpoint(cfg, tmp_path / "stage-1.pt")
  assert "[WARN]" in capsys.readouterr().out
  assert cfg.osl_fsm == SDK
  assert cfg.osl_touch_sites == "original"  # Stage 1's record: none, so original.


def test_a_reward_switch_holds_the_human_for_the_fresh_critic(tmp_path: Path) -> None:
  """Under a reward switch the critic starts fresh, so a human learning from k=0
  would follow a random value function's advantages: refused, naming the freeze
  that holds it for the critic's warm-up, which then starts."""
  ws_path = tmp_path / "ws_mimic.pt"
  _write_warm_start(ws_path, reward="mimic")
  with pytest.raises(ValueError, match="--agent.freeze.human-actor-iterations 50"):
    _FsmRunner(cast(Any, _FsmEnv()), _quick_cfg(ws_path), None, "cpu")
  cfg = _quick_cfg(ws_path)
  cfg["freeze"]["human_actor_iterations"] = 50
  runner = _FsmRunner(cast(Any, _FsmEnv()), cfg, None, "cpu")
  assert runner.reward_switch and runner.schedule.human_freeze == 50


def test_the_initial_state_is_no_part_of_the_controller(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """A checkpoint that differs from the env in its initial state alone loads (with
  a note), and play starts its episodes in early stance, as validation and
  deployment do, whatever the checkpoint trained with."""
  path = tmp_path / "model_pose.pt"
  _bare_runner(replace(SDK, initial_state="pose")).save(str(path))
  capsys.readouterr()
  resumed = _bare_runner(SDK)
  resumed.load(str(path))
  assert resumed._ready
  assert "trained with the initial state 'pose'" in capsys.readouterr().out
  cfg = osl_fsm_env_cfg("msk", "compliant", "clf", "play")
  FsmSplitRunner.configure_env_for_checkpoint(cfg, path)
  assert cfg.osl_fsm == SDK
  term = cfg.actions["osl_motors"]
  assert isinstance(term, OslFsmActionCfg) and term.initial_state == "e_stance"
  assert "episodes start in e_stance" in capsys.readouterr().out
  _bare_runner(cfg.osl_fsm).load(str(path), load_cfg={"actor": True})


def test_exports_point_to_the_bundle_exporter(tmp_path: Path) -> None:
  runner = _bare_runner()
  with pytest.raises(NotImplementedError, match="mjlab.tasks.osl_fsm.export"):
    runner.export_policy_to_onnx(str(tmp_path))
  with pytest.raises(NotImplementedError, match="mjlab.tasks.osl_fsm.export"):
    runner.export_policy_to_jit(str(tmp_path))


##
# Evaluation envs.
##


def test_evaluation_envs_run_the_training_controller(tmp_path: Path) -> None:
  """Validation and video configs are rebuilt from the task with the training
  env's controller (its osl_fsm and its whole osl_motors term), library, settings
  and touch sites, their episodes starting in early stance as osl_deploy's do;
  stress copies the training config, initial state included. All three keep the OSL
  motors undelayed and the clean osl group, and each its mode's human delay. The
  envs themselves are not built."""
  pose = replace(SDK, initial_state="pose")
  train = osl_fsm_env_cfg("msk", "compliant", "clf", "train", fsm=pose)
  term = train.actions["osl_motors"]
  assert isinstance(term, OslFsmActionCfg)
  train.actions = {**train.actions, "osl_motors": replace(term, controller="policy")}
  set_touch_sites(train, "original")
  motion = train.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = str(tmp_path)
  train.sim.mujoco.iterations = 7
  train.rewards["twist"].weight = 0.35

  runner: Any = object.__new__(FsmSplitRunner)
  runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=train))
  runner.validation_cfg = dict(asdict(load_rl_cfg(RUN_TASK))["validation"])
  human_delays = {"validation": (15, 15), "video": (15, 15), "stress": (10, 20)}
  for purpose, human in human_delays.items():
    cfg = runner._eval_env_cfg(purpose)
    assert isinstance(cfg, OslFsmEnvCfg), purpose
    stress = purpose == "stress"
    assert cfg.osl_fsm == (pose if stress else SDK), purpose
    trained = train.actions["osl_motors"]
    assert isinstance(trained, OslFsmActionCfg)
    expected = trained if stress else replace(trained, initial_state="e_stance")
    assert cfg.actions["osl_motors"] == expected, purpose
    assert _delays(cfg) == {"osl": (0, 0), "human": human}, purpose
    assert cfg.observations["osl"] == clean_osl_group(), purpose
    robot = cfg.scene.entities["robot"]
    assert isinstance(robot, OslEntityCfg)
    assert cfg.osl_touch_sites == robot.touch_sites == "original", purpose
    eval_motion = cfg.commands["motion"]
    assert isinstance(eval_motion, GaitLibraryCommandCfg)
    assert eval_motion.library_dir == str(tmp_path), purpose
    assert cfg.sim.mujoco.iterations == 7, purpose
    assert cfg.rewards["twist"].weight == 0.35, purpose
  # Validation keeps its own mode (no pushes, no drift termination); the video
  # plays the nominal plant; stress is the training config without the seed.
  validation = runner._eval_env_cfg("validation")
  assert "push_robot" not in validation.events
  assert "root_deviation" not in validation.terminations
  assert "osl_motor_strength" not in runner._eval_env_cfg("video").events
  assert "osl_motor_strength" in train.events
  train.seed = 7
  stress = runner._eval_env_cfg("stress")
  assert stress.seed is None and train.seed == 7
  assert replace(stress, seed=7) == train and stress is not train

  # A stage-2 training config has no controller to give.
  runner.env.unwrapped.cfg = load_env_cfg("Mjlab-Split-Unicycle-OSL-Fullbody-MSK")
  with pytest.raises(TypeError, match="stage-1.5"):
    runner._eval_env_cfg("validation")


##
# Import.
##


_REGISTRATION_FIRST = """
import importlib, importlib.util, sys, types
import mjlab
held = types.ModuleType("mjlab.tasks")
held.__path__ = list(importlib.util.find_spec("mjlab.tasks").submodule_search_locations)
sys.modules["mjlab.tasks"] = held  # The walker waits.
import mjlab.tasks.osl_fsm.config
del sys.modules["mjlab.tasks"]
importlib.import_module("mjlab.tasks")  # Now the walker, over every task.
from mjlab.tasks.registry import list_tasks
print(sum(task.startswith("Mjlab-FSM-") for task in list_tasks()), "stage-1.5 tasks")
"""


def test_modules_import_cleanly() -> None:
  """The tasks walker imports every package and re-raises, so a broken import here
  would end every task launch. Importing any task module runs the walker first (as
  this test module's imports just did), and the walker reaches this registration
  before stage 2's packages, which the runner builds on: the registration completes
  stage 2's first. A fresh interpreter imports the registration ahead of every other
  task package, then lets the walker import them all."""
  result = subprocess.run(
    [sys.executable, "-c", _REGISTRATION_FIRST],
    capture_output=True,
    text=True,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    check=False,
    timeout=600,
  )
  assert result.returncode == 0, result.stderr[-3000:]
  assert "8 stage-1.5 tasks" in result.stdout
