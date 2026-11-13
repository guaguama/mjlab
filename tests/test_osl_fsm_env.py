"""Tests for the stage-1.5 environment (``mjlab.tasks.osl_fsm``): the OSL controller's
action term and the env config.

CPU only. The action term runs on a fake env (sensors as ``.data`` tensors, an
entity that records its effort targets) over a real compiled model, which gives the
tare, the motor gains and the body mass; the configs are compared term for term with
stage 2's. Only the slow test builds the real env (``MJLAB_OSL_UNICYCLE_LIBRARY``).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  OSL_TORQUE_JOINTS,
  Model,
  Socket,
  get_spec,
)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.scene import Scene
from mjlab.tasks.osl_fsm import fsm as core
from mjlab.tasks.osl_fsm import mdp
from mjlab.tasks.osl_fsm.config.env_cfgs import (
  FSM_FLAGS,
  OslFsmCfg,
  OslFsmEnvCfg,
  check_osl_fsm,
  clean_osl_group,
  configure_fsm,
  fsm_settings,
  osl_fsm_env_cfg,
  sync_osl_fsm,
)
from mjlab.tasks.osl_gait_library.config.dr import OslDrCfg
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic.config.env_cfgs import OSL_MOTOR_CTRL, Mode, Reward
from mjlab.tasks.osl_split import mdp as split_mdp
from mjlab.tasks.osl_split.config import sensors
from mjlab.tasks.osl_split.config.env_cfgs import osl_split_env_cfg

MODELS: tuple[Model, ...] = ("robot", "msk")
SOCKETS: tuple[Socket, ...] = ("rigid", "compliant")
REWARDS: tuple[Reward, ...] = ("mimic", "clf")
MODES: tuple[Mode, ...] = ("train", "play", "validation")
KNEE, ANKLE = OSL_TORQUE_JOINTS
WEIGHT = 800.0  # N, osl_deploy's test body weight.
GAINS = (49.4, 58.4)  # N m per ctrl unit, knee then ankle.
NM_PER_CTRL = {"knee": GAINS[0], "ankle": GAINS[1]}
rad = math.radians

##
# The action term on a fake env.
##


def _scene_model(model: Model, socket: Socket) -> mujoco.MjModel:
  """The robot with the OSL sensors attached under ``robot/`` over a floor, as
  mjlab's scene attaches entities."""
  scene = mujoco.MjSpec()
  scene.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(5.0, 5.0, 0.1))
  entity = sensors.with_osl_sensors(lambda: get_spec(model, socket))()
  scene.attach(entity, prefix="robot/", frame=scene.worldbody.add_frame())
  return scene.compile()


def _motor_ids(model: mujoco.MjModel, joints: tuple[str, ...] | list[str]) -> list[int]:
  """Each joint's actuator id in a scene model, found by its transmission."""
  ids = []
  for joint in joints:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint}")
    (motor,) = [
      a
      for a in range(model.nu)
      if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
      and int(model.actuator_trnid[a, 0]) == joint_id
    ]
    ids.append(motor)
  return ids


@pytest.fixture(scope="module")
def msk_model() -> mujoco.MjModel:
  return _scene_model("msk", "compliant")


class _Entity:
  """The entity calls the action term makes: its joints and effort targets."""

  def __init__(self, targets: tuple[str, ...], num_envs: int) -> None:
    self.targets = targets
    self.effort = torch.full((num_envs, len(targets)), math.nan)

  def find_joints_by_actuator_names(self, keys: Any) -> tuple[list[int], list[str]]:
    del keys
    return list(range(len(self.targets))), list(self.targets)

  def set_joint_effort_target(
    self, effort: torch.Tensor, joint_ids: torch.Tensor | None = None
  ) -> None:
    assert joint_ids is not None
    self.effort[:, joint_ids] = effort


def _fake_env(
  model: mujoco.MjModel,
  num_envs: int = 4,
  targets: tuple[str, ...] = OSL_TORQUE_JOINTS,
  sensor_names: tuple[str, ...] = sensors.OSL_SENSORS,
) -> Any:
  scene: dict[str, Any] = {
    sensors.scene_sensor_name(spec.name): SimpleNamespace(
      data=torch.zeros(num_envs, spec.dim)
    )
    for spec in sensors.OSL_SENSOR_SPECS
    if spec.name in sensor_names
  }
  scene["robot"] = _Entity(targets, num_envs)
  return SimpleNamespace(
    num_envs=num_envs, device="cpu", scene=scene, sim=SimpleNamespace(mj_model=model)
  )


def _cfg(**overrides: Any) -> mdp.OslFsmActionCfg:
  """The stage-1.5 task's own term config, with ``overrides``."""
  term = osl_fsm_env_cfg().actions["osl_motors"]
  assert isinstance(term, mdp.OslFsmActionCfg)
  return replace(term, **overrides)


def _term(env: Any, **overrides: Any) -> mdp.OslFsmAction:
  term = _cfg(**overrides).build(env)
  term.reset(None)  # As the env's first reset.
  return term


def _read(
  env: Any,
  knee: float = 0.0,
  knee_velocity: float = 0.0,
  ankle: float = 0.0,
  ankle_velocity: float = 0.0,
  load: float = 0.0,
  tare: float = 0.0,
  rows: list[int] | None = None,
) -> None:
  """Set the sensors: ``load`` N of compression, which the ankle cell reads as
  ``tare - load`` along y (raw, untared)."""
  values = {
    sensors.KNEE_POS: [knee],
    sensors.KNEE_VEL: [knee_velocity],
    sensors.ANKLE_POS: [ankle],
    sensors.ANKLE_VEL: [ankle_velocity],
    sensors.ANKLE_FORCE: [0.3, tare - load, -0.2],
  }
  for name, value in values.items():
    data = env.scene[sensors.scene_sensor_name(name)].data
    data[slice(None) if rows is None else rows] = torch.tensor(value)


def _states(term: mdp.OslFsmAction) -> list[str]:
  return [core.STATES[code] for code in term.state.tolist()]


def test_term_commands_the_new_states_torque_through_the_nominal_motors(
  msk_model: mujoco.MjModel,
) -> None:
  """osl_deploy's policy test on the action term: 400 N of compression and a
  dorsiflexed ankle leave early stance, the late-stance spring-damper is commanded
  as ctrl = torque / (nominal N m per ctrl), and the raw action is ctrl / 2.88. The
  policy's rows play no part."""
  env = _fake_env(msk_model)
  term = _term(env, body_mass=WEIGHT / 9.81, initial_state="e_stance")
  assert term.body_weight_n == pytest.approx(WEIGHT)
  assert term.nm_per_ctrl.tolist() == pytest.approx(GAINS)
  _read(env, knee=0.2, knee_velocity=1.0, ankle=0.15, load=400.0, tare=term.tare_n)
  term.process_actions(torch.randn(4, 2) * 10.0)
  assert _states(term) == ["l_stance"] * 4
  knee = 99.372 * (rad(8) - 0.2) - 1.272 * 1.0
  ankle = 79.498 * (rad(-20) - 0.15)
  expected = torch.tensor([knee, ankle]).expand(4, -1)
  torch.testing.assert_close(term.torque, expected)
  ctrl = expected / torch.tensor(GAINS)
  torch.testing.assert_close(term.ctrl, ctrl)
  torch.testing.assert_close(term.raw_action, ctrl / OSL_MOTOR_CTRL)
  readings = [0.2, 1.0, 0.15, 0.0, -400.0]  # The tared axial force: -load.
  torch.testing.assert_close(term.readings, torch.tensor(readings).expand(4, -1))
  # The command is held: every substep writes the same effort target.
  for _ in range(3):
    term.apply_actions()
    torch.testing.assert_close(env.scene["robot"].effort, ctrl)

  # Tension (the hanging leg) is no load: late stance ends.
  _read(env, load=-30.0, tare=term.tare_n)
  term.process_actions(torch.zeros(4, 2))
  assert _states(term) == ["e_swing"] * 4
  # The same step under other policy rows commands the same: the rows are ignored.
  before = term.ctrl.clone()
  term.fsm.state[:] = core.L_STANCE
  term.process_actions(torch.full((4, 2), 0.9))
  term.fsm.state[:] = core.L_STANCE
  term.process_actions(torch.full((4, 2), -0.9))
  torch.testing.assert_close(term.ctrl, before)


def test_command_clips_at_the_motor_limits(msk_model: mujoco.MjModel) -> None:
  env = _fake_env(msk_model)
  # The sdk rules keep an unloaded leg in early stance whatever its ankle.
  term = _term(env, rules="sdk", initial_state="e_stance")
  # e_stance: knee 99.4 N m/rad toward 5 deg, ankle 19.9 toward -2 deg, so a
  # flexed knee and a dorsiflexed ankle push back past the limits; then the other way.
  for sign in (1.0, -1.0):
    term.fsm.reset()
    _read(env, knee=sign * 2.0, ankle=sign * 10.0, tare=term.tare_n)
    term.process_actions(torch.zeros(4, 2))
    assert _states(term) == ["e_stance"] * 4
    limit = torch.full((4, 2), -sign * OSL_MOTOR_CTRL)
    torch.testing.assert_close(term.ctrl, limit)
    torch.testing.assert_close(term.raw_action, limit / OSL_MOTOR_CTRL)
    assert term.raw_action.abs().max() <= 1.0  # What the osl group records.
    assert float(term.torque.abs().min()) > 168.192  # Before the clip.
    for key, limit_nm in (("fsm_knee_torque", 142.272), ("fsm_ankle_torque", 168.192)):
      torch.testing.assert_close(term.components[key], torch.full((4,), limit_nm))


def test_command_follows_the_terms_joint_order(msk_model: mujoco.MjModel) -> None:
  """The term's columns are the entity's joint order: here ankle first."""
  env = _fake_env(msk_model, targets=(ANKLE, KNEE))
  term = _term(env, initial_state="e_stance")
  assert term.nm_per_ctrl.tolist() == pytest.approx(GAINS)  # Knee, ankle.
  _read(env, knee=0.3, ankle=-0.05, tare=term.tare_n)
  term.process_actions(torch.zeros(4, 2))
  knee = 99.372 * (rad(5) - 0.3)
  ankle = 19.874 * (rad(-2) + 0.05)
  ctrl = torch.tensor([ankle / GAINS[1], knee / GAINS[0]]).expand(4, -1)
  torch.testing.assert_close(term.ctrl, ctrl)
  torch.testing.assert_close(term.torque, torch.tensor([knee, ankle]).expand(4, -1))
  with pytest.raises(ValueError, match="one knee and one ankle"):
    _term(_fake_env(msk_model, targets=(KNEE, KNEE)))


def test_pose_guess_on_the_first_step_after_a_reset(msk_model: mujoco.MjModel) -> None:
  """Reset envs start in early stance with the guess pending; the next step guesses
  from its own reading, then updates on it. Other envs and later steps update only."""
  env = _fake_env(msk_model)
  term = _term(env, rules="sdk", swing_load_exit=False)
  # Unloaded, the knee flexing at 40 deg: early swing, which the sdk rules keep
  # (knee under 50 deg) on the same reading.
  _read(env, knee=rad(40), knee_velocity=2.0, tare=term.tare_n)
  term.process_actions(torch.zeros(4, 2))
  assert _states(term) == ["e_swing"] * 4
  # A later step guesses nothing: an extending knee stays in early swing.
  _read(env, knee=rad(40), knee_velocity=-1.0, tare=term.tare_n)
  term.process_actions(torch.zeros(4, 2))
  assert _states(term) == ["e_swing"] * 4

  term.reset(torch.tensor([1, 2]))
  assert _states(term) == ["e_swing", "e_stance", "e_stance", "e_swing"]
  # Env 1 starts unloaded and extending (late swing); env 2 loaded (early stance,
  # which the sdk rules leave only with the ankle past 6 deg too). Env 3's load
  # takes it back to stance through the sdk rules' exit, which is off here: it
  # stays.
  _read(env, knee=rad(40), knee_velocity=-1.0, tare=term.tare_n)
  _read(env, knee=rad(10), load=0.5 * WEIGHT, ankle=rad(2), tare=term.tare_n, rows=[2])
  _read(env, knee=rad(40), load=0.5 * WEIGHT, tare=term.tare_n, rows=[3])
  term.process_actions(torch.zeros(4, 2))
  assert _states(term) == ["e_swing", "l_swing", "e_stance", "e_swing"]

  # The guess only initializes: a first step that guesses late swing and whose
  # update then enters early stance (knee under 30 deg) counts no cycle.
  term = _term(env, rules="myosuite", initial_state="pose")
  _read(env, knee=rad(10), knee_velocity=-1.0, tare=term.tare_n)
  term.process_actions(torch.zeros(4, 2))
  assert _states(term) == ["e_stance"] * 4
  assert term.components["fsm_cycle"].tolist() == [0.0] * 4
  term.reset(torch.tensor([1]))
  _read(env, knee=rad(10), knee_velocity=-1.0, ankle=rad(8), tare=term.tare_n)
  term.process_actions(torch.zeros(4, 2))  # Env 1 guesses again; the rest move.
  assert term.components["fsm_cycle"].tolist() == [0.0] * 4

  # Under "e_stance", a reset env starts in early stance and simply updates.
  term = _term(env, rules="sdk", swing_load_exit=False, initial_state="e_stance")
  _read(env, knee=rad(40), knee_velocity=2.0, tare=term.tare_n)
  term.process_actions(torch.zeros(4, 2))
  assert _states(term) == ["e_stance"] * 4


def test_components_report_the_step(msk_model: mujoco.MjModel) -> None:
  env = _fake_env(msk_model, num_envs=2)
  term = _term(env, body_mass=WEIGHT / 9.81, initial_state="e_stance")
  assert tuple(term.components) == mdp.FSM_DIAGNOSTICS
  # MyoSuite's rules cycle one state per update before foot contact.
  _read(env, knee=rad(10), knee_velocity=-1.0, ankle=rad(8), tare=term.tare_n)
  seen = []
  for _ in range(4):
    term.process_actions(torch.zeros(2, 2))
    seen.append({k: v.tolist() for k, v in term.components.items()})
  order = ["l_stance", "e_swing", "l_swing", "e_stance"]
  for state, step in zip(order, seen, strict=True):
    for name in core.STATES:
      assert step[f"fsm_{name}"] == [float(name == state)] * 2
    assert step["fsm_cycle"] == [float(state == "e_stance")] * 2
    assert step["fsm_load_bw"] == [0.0] * 2
  commanded = (term.ctrl * term.nm_per_ctrl).abs()
  torch.testing.assert_close(term.components["fsm_knee_torque"], commanded[:, 0])
  torch.testing.assert_close(term.components["fsm_ankle_torque"], commanded[:, 1])
  _read(env, load=200.0, tare=term.tare_n)
  term.process_actions(torch.zeros(2, 2))
  torch.testing.assert_close(term.components["fsm_load_bw"], torch.full((2,), 0.25))
  # A reset zeroes the reset envs' command (MuJoCo's ctrl after a reset), and only
  # theirs, and keeps every env's last reading and diagnostics: the runners read
  # them after step(), which has reset the envs that ended an episode.
  kept = [value.clone() for value in (*term.components.values(), term.readings)]
  ctrl, torque = term.ctrl.clone(), term.torque.clone()
  assert bool(ctrl.all())
  term.reset(torch.tensor([0]))
  assert not term.ctrl[0].any() and not term.raw_action[0].any()
  torch.testing.assert_close(term.ctrl[1], ctrl[1])
  for value, before in zip(
    (*term.components.values(), term.readings), kept, strict=True
  ):
    torch.testing.assert_close(value, before)
  torch.testing.assert_close(term.torque, torque)
  assert term.state[0] == core.E_STANCE
  # The metric function reads them; anything absent is zero.
  manager = SimpleNamespace(get_term=lambda name: {"osl_motors": term}[name])
  metric_env = cast(
    Any, SimpleNamespace(num_envs=2, device="cpu", action_manager=manager)
  )
  assert mdp.fsm_component(metric_env, "fsm_load_bw") is term.components["fsm_load_bw"]
  assert not mdp.fsm_component(metric_env, "fsm_unknown").any()
  manager.get_term = lambda name: SimpleNamespace()
  assert mdp.fsm_component(metric_env, "fsm_load_bw").shape == (2,)


def test_policy_controller_is_a_plain_effort_action(msk_model: mujoco.MjModel) -> None:
  env = _fake_env(msk_model)
  term = _term(env, controller="policy")
  _read(env, knee=1.0, load=500.0, tare=term.tare_n)
  actions = torch.tensor([[0.5, -2.0]]).expand(4, -1)
  term.process_actions(actions)
  torch.testing.assert_close(term.raw_action, actions)
  expected = torch.tensor([[0.5 * OSL_MOTOR_CTRL, -OSL_MOTOR_CTRL]]).expand(4, -1)
  torch.testing.assert_close(term.ctrl, expected)
  assert _states(term) == ["e_stance"] * 4
  assert not any(value.any() for value in term.components.values())
  assert term.settings()["controller"] == "policy"


@pytest.mark.parametrize("controller", ["fsm", "policy"])
def test_reset_after_inference_mode_steps(
  msk_model: mujoco.MjModel, controller: str
) -> None:
  """A step under inference mode followed by a reset outside it, as a script that
  steps in inference mode and resets after it does: both controllers reset."""
  env = _fake_env(msk_model)
  term = _term(env, controller=controller)
  _read(env, knee=0.5, knee_velocity=1.0, load=300.0, tare=term.tare_n)
  with torch.inference_mode():
    term.process_actions(torch.full((4, 2), 0.5))
  term.reset(torch.tensor([0, 1]))
  assert not term.raw_action[:2].any()


def test_term_refusals(msk_model: mujoco.MjModel) -> None:
  without_cell = tuple(n for n in sensors.OSL_SENSORS if n != sensors.ANKLE_FORCE)
  with pytest.raises(ValueError, match="with_osl_sensors"):
    _cfg().build(_fake_env(msk_model, sensor_names=without_cell))
  env = _fake_env(msk_model)
  with pytest.raises(ValueError, match="sdk rules"):
    _cfg(swing_load_exit=True).build(env)
  with pytest.raises(ValueError, match="Unknown rules"):
    _cfg(rules="hkic").build(env)
  with pytest.raises(ValueError, match="initial_state"):
    _cfg(initial_state="standing").build(env)
  with pytest.raises(ValueError, match="controller"):
    _cfg(controller="network").build(env)
  for mass in (0.0, -80.0, math.nan):
    with pytest.raises(ValueError, match="body mass"):
      _cfg(body_mass=mass).build(env)


@pytest.mark.parametrize("model, socket", [("robot", "rigid"), ("msk", "compliant")])
def test_nominal_motor_gains_read_gain_times_gear(model: Model, socket: Socket) -> None:
  """The robot model writes 49.4 / 58.4 as the gear, the musculoskeletal one as the
  gain: the torque per ctrl unit is their product on both."""
  compiled = _scene_model(model, socket)
  gains, ranges = mdp.nominal_motor_gains(compiled, "robot/", [KNEE, ANKLE])
  assert gains == pytest.approx(GAINS)
  assert ranges == [(-OSL_MOTOR_CTRL, OSL_MOTOR_CTRL)] * 2
  with pytest.raises(ValueError, match="one motor"):
    mdp.nominal_motor_gains(compiled, "robot/", ["no_such_joint"])
  (knee_motor, _) = _motor_ids(compiled, (KNEE, ANKLE))
  compiled.actuator_biastype[knee_motor] = mujoco.mjtBias.mjBIAS_AFFINE
  with pytest.raises(ValueError, match="fixed-gain"):
    mdp.nominal_motor_gains(compiled, "robot/", [KNEE])


def test_settings_are_plain_and_nominal(msk_model: mujoco.MjModel) -> None:
  """The resolved settings a checkpoint stores, on the real model: the tare, the
  motor gains and the body mass are the nominal model's."""
  env = _fake_env(msk_model)
  settings = _term(env).settings()
  json.dumps(settings)  # Plain data.
  assert {
    "controller": "fsm",
    "rules": "myosuite",
    "every": False,
    "knee_velocity": rad(3),
    "swing_load_exit": False,
    "body_mass": None,
    "initial_state": "pose",
    "update": mdp.UPDATE,
  }.items() <= settings.items()
  assert settings["tare_n"] == pytest.approx(19.9143, abs=1e-3)
  assert settings["nm_per_ctrl"] == pytest.approx(NM_PER_CTRL)
  assert settings["ctrl_limits"]["ankle"] == pytest.approx([-2.88, 2.88])
  mass = float(msk_model.body_mass.sum())  # The world and the floor are massless.
  assert (
    settings["body_mass_kg"]
    == pytest.approx(mass)
    == mdp.entity_mass(msk_model, "robot/")
  )
  assert mass == pytest.approx(82.289, abs=1e-3)
  assert settings["body_weight_n"] == pytest.approx(mass * 9.81)
  assert settings["impedance"] == core.impedance_settings()
  assert settings["thresholds"] == core.threshold_settings()
  assert set(FSM_FLAGS) <= set(settings)
  sdk = _term(env, rules="sdk", body_mass=80.0).settings()
  assert sdk["swing_load_exit"] is True and sdk["every"] is True
  assert (sdk["body_mass"], sdk["body_mass_kg"]) == (80.0, 80.0)


def test_fsm_settings_reads_the_env_term(msk_model: mujoco.MjModel) -> None:
  term = _term(_fake_env(msk_model))
  env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))
  wrapped = SimpleNamespace(unwrapped=env)
  assert fsm_settings(wrapped) == fsm_settings(env) == term.settings()
  stage2 = JointEffortActionCfg(entity_name="robot", actuator_names=OSL_TORQUE_JOINTS)
  plain = stage2.build(_fake_env(msk_model))
  env.action_manager.get_term = lambda name: plain
  with pytest.raises(TypeError, match="not a stage-1.5 env"):
    fsm_settings(env)


##
# The env config.
##


def _delays(cfg: ManagerBasedRlEnvCfg) -> dict[str, tuple[int, int]]:
  """(min, max) actuator lags in physics steps, OSL motors vs human actuators."""
  entity = cfg.scene.entities["robot"]
  assert entity.articulation is not None
  lags = {}
  for act in entity.articulation.actuators:
    key = "osl" if tuple(act.target_names_expr) == OSL_TORQUE_JOINTS else "human"
    lags[key] = (act.delay_min_lag, act.delay_max_lag)
  return lags


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reward", REWARDS)
@pytest.mark.parametrize("socket", SOCKETS)
@pytest.mark.parametrize("model", MODELS)
def test_env_cfg_contract(
  model: Model, socket: Socket, reward: Reward, mode: Mode
) -> None:
  cfg = osl_fsm_env_cfg(model, socket, reward, mode)
  base = osl_split_env_cfg(model, socket, reward, mode)
  assert type(cfg) is OslFsmEnvCfg and cfg.osl_fsm == OslFsmCfg()

  # The OSL motors: stage 1's term in its slot, under the controller.
  assert list(cfg.actions) == list(base.actions)
  term, stage1 = cfg.actions["osl_motors"], base.actions["osl_motors"]
  assert isinstance(term, mdp.OslFsmActionCfg) and type(stage1) is JointEffortActionCfg
  for f in fields(stage1):
    assert getattr(term, f.name) == getattr(stage1, f.name), f.name
  assert (term.scale, term.offset) == (OSL_MOTOR_CTRL, 0.0)
  assert term.clip == {".*": (-OSL_MOTOR_CTRL, OSL_MOTOR_CTRL)}
  assert (term.rules, term.swing_load_exit, term.body_mass) == ("myosuite", None, None)
  assert (term.initial_state, term.controller) == ("pose", "fsm")
  for name in base.actions:
    if name != "osl_motors":
      assert cfg.actions[name] == base.actions[name]

  # No OSL motor delay; the human's is stage 2's.
  osl, human = _delays(cfg), _delays(base)
  assert osl["osl"] == (0, 0) and osl["human"] == human["human"]
  assert human["human"] == ((15, 15) if mode == "validation" else (10, 20))

  # The osl group: clean and immediate in every mode, 25 frames, 600 dims.
  group = cfg.observations["osl"]
  assert group == clean_osl_group()
  validation = split_mdp.OslObsCfg(delay_steps=(0, 0), validation_delay_steps=0)
  assert group == split_mdp.osl_observation_group("validation", validation)
  assert tuple(group.terms) == split_mdp.OSL_OBS_TERMS and not group.enable_corruption
  for name, obs_term in group.terms.items():
    assert (obs_term.delay_min_lag, obs_term.delay_max_lag) == (0, 0), name
    assert obs_term.noise is None and not obs_term.params.get("randomize", False)
    assert obs_term.history_length == 25
  assert group.terms["loadcell"].params["ankle"]
  assert split_mdp.osl_obs_dim(split_mdp.OslObsCfg()) == 600

  # One mean metric per diagnostic; everything else is stage 2's.
  extra = {name for name in cfg.metrics if name not in base.metrics}
  assert extra == set(mdp.FSM_DIAGNOSTICS)
  for name in mdp.FSM_DIAGNOSTICS:
    metric = cfg.metrics[name]
    assert metric.func is mdp.fsm_component and metric.params == {"name": name}
    assert metric.reduce == "mean" and not metric.per_substep
  for name, metric in base.metrics.items():
    assert cfg.metrics[name] == metric
  assert cfg.observations["actor"] == base.observations["actor"]
  assert set(cfg.observations) == set(base.observations) == {"actor", "osl"}
  assert cfg.rewards == base.rewards and cfg.terminations == base.terminations
  assert cfg.commands == base.commands and set(cfg.events) == set(base.events)
  for name, event in base.events.items():
    assert cfg.events[name].func is event.func
    assert cfg.events[name].params == event.params
  assert cfg.sim == base.sim and cfg.decimation == base.decimation
  assert cfg.scene.sensors == base.scene.sensors
  assert cfg.scene.num_envs == base.scene.num_envs
  assert cfg.episode_length_s == base.episode_length_s
  assert cfg.osl_touch_sites == base.osl_touch_sites


def test_env_cfg_options() -> None:
  settings = OslFsmCfg(rules="sdk", swing_load_exit=False, body_mass=75.0)
  dr_cfg = OslDrCfg(osl_delay_s=(0.0, 0.004), foot_friction=(0.8, 1.0))
  cfg = osl_fsm_env_cfg(dr_cfg=dr_cfg, fsm=settings)
  assert cfg.osl_fsm == settings and cfg.osl_fsm is not settings  # A copy.
  term = cfg.actions["osl_motors"]
  assert isinstance(term, mdp.OslFsmActionCfg)
  assert (term.rules, term.swing_load_exit, term.body_mass) == ("sdk", False, 75.0)
  assert _delays(cfg)["osl"] == (0, 2)
  assert cfg.events["foot_friction"].params["ranges"] == (0.8, 1.0)
  with pytest.raises(ValueError, match="human"):
    osl_fsm_env_cfg(dr_cfg=OslDrCfg(osl_delay_s=(0.0, 0.0), obs_noise=True))


@dataclass
class _Cli:
  """train.py's ``TrainConfig.env``: the same ``--env.*`` flags."""

  env: ManagerBasedRlEnvCfg


def _cli_env(*args: str) -> OslFsmEnvCfg:
  cfg = tyro.cli(
    _Cli, args=list(args), default=_Cli(env=osl_fsm_env_cfg()), config=mjlab.TYRO_FLAGS
  )
  assert isinstance(cfg.env, OslFsmEnvCfg)
  return cfg.env


def _fsm_fields(cfg: ManagerBasedRlEnvCfg) -> tuple[Any, ...]:
  term = cfg.actions["osl_motors"]
  assert isinstance(term, mdp.OslFsmActionCfg)
  return term.rules, term.swing_load_exit, term.body_mass, term.initial_state


def test_cli_reaches_the_action_term(capsys: pytest.CaptureFixture[str]) -> None:
  """tyro rebuilds the env config after the overrides, so ``--env.osl-fsm.*``
  reaches the action term. The term's own copies are no flags (they would be
  overwritten), and the myosuite rules refuse the swing exit."""
  cfg = _cli_env(
    "--env.osl-fsm.rules",
    "sdk",
    "--env.osl-fsm.swing-load-exit",
    "False",
    "--env.osl-fsm.body-mass",
    "80",
    "--env.osl-fsm.initial-state",
    "e_stance",
    "--env.osl-touch-sites",
    "original",
  )
  assert cfg.osl_fsm == OslFsmCfg("sdk", False, 80.0, "e_stance")
  assert _fsm_fields(cfg) == ("sdk", False, 80.0, "e_stance")
  assert cfg.osl_touch_sites == "original"
  assert _fsm_fields(_cli_env()) == ("myosuite", None, None, "pose")
  for args in (
    ("--env.actions.osl-motors.rules", "sdk"),
    ("--env.osl-fsm.swing-load-exit", "True"),
    ("--env.osl-fsm.rules", "hkic"),
  ):
    with pytest.raises(SystemExit) as error:
      _cli_env(*args)
    assert error.value.code == 2
  # tyro's error boxes wrap at the terminal's width: read them as one line.
  err = " ".join(re.sub(r"[│╭╮╰╯─]", " ", capsys.readouterr().err).split())
  assert "Unrecognized options: --env.actions.osl-motors.rules" in err
  assert "The load-based swing exit belongs to the sdk rules" in err


def test_post_init_syncs_and_refuses() -> None:
  """Construct, mutate, rebuild: what tyro does, by hand."""
  cfg = osl_fsm_env_cfg()
  cfg.osl_fsm.rules = "sdk"
  cfg.osl_fsm.initial_state = "e_stance"
  cfg.__post_init__()
  assert _fsm_fields(cfg) == ("sdk", None, None, "e_stance")
  rebuilt = OslFsmEnvCfg(**{f.name: getattr(cfg, f.name) for f in fields(cfg)})
  assert _fsm_fields(rebuilt) == _fsm_fields(cfg)
  standing = cast(Any, "standing")
  for bad, match in (
    (OslFsmCfg("myosuite", True), "sdk rules.*--env.osl-fsm.rules myosuite"),
    (OslFsmCfg(body_mass=0.0), "--env.osl-fsm.body-mass"),
    (OslFsmCfg(initial_state=standing), "--env.osl-fsm.initial-state"),
  ):
    with pytest.raises(ValueError, match=match):
      check_osl_fsm(bad)
    cfg.osl_fsm = bad
    with pytest.raises(ValueError, match=match):
      cfg.__post_init__()
  cfg = osl_fsm_env_cfg()
  cfg.observations["osl"].terms["loadcell"].params["ankle"] = False
  with pytest.raises(ValueError, match="ankle load cell"):
    sync_osl_fsm(cfg)
  cfg = osl_fsm_env_cfg()
  cfg.actions["osl_motors"] = osl_split_env_cfg().actions["osl_motors"]
  with pytest.raises(TypeError, match="OslFsmActionCfg"):
    sync_osl_fsm(cfg)


def test_configure_fsm_from_a_checkpoint(msk_model: mujoco.MjModel) -> None:
  """A checkpoint's stored settings (or an OslFsmCfg) set the env config's
  controller; the env it builds resolves the same settings."""
  stored = _term(
    _fake_env(msk_model), rules="sdk", body_mass=81.0, initial_state="e_stance"
  ).settings()
  cfg = osl_fsm_env_cfg()
  applied = configure_fsm(cfg, stored)
  assert applied == cfg.osl_fsm == OslFsmCfg("sdk", True, 81.0, "e_stance")
  term_cfg = cfg.actions["osl_motors"]
  assert isinstance(term_cfg, mdp.OslFsmActionCfg)
  rebuilt = term_cfg.build(_fake_env(msk_model)).settings()
  assert {key: rebuilt[key] for key in FSM_FLAGS} == {
    key: stored[key] for key in FSM_FLAGS
  }
  assert configure_fsm(cfg, OslFsmCfg()) == OslFsmCfg()
  assert _fsm_fields(cfg) == ("myosuite", None, None, "pose")
  with pytest.raises(ValueError, match="lack"):
    configure_fsm(cfg, {"rules": "sdk"})
  with pytest.raises(TypeError, match="stage-1.5"):
    configure_fsm(osl_split_env_cfg(), OslFsmCfg())


def test_modules_import_cleanly() -> None:
  """Every module of this unit imports in a fresh interpreter: the task walker
  re-raises import errors, so a broken import would stop every task's launch."""
  modules = (
    "mjlab.tasks.osl_fsm.fsm",
    "mjlab.tasks.osl_fsm.mdp",
    "mjlab.tasks.osl_fsm.config.env_cfgs",
  )
  for module in modules:  # Each first, on its own: no import order hides a cycle.
    code = f"import {module}"
    result = subprocess.run(
      [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]


##
# The real env (slow; CPU works).
##

UNICYCLE_LIBRARY = os.environ.get("MJLAB_OSL_UNICYCLE_LIBRARY", "")


@pytest.mark.slow
@pytest.mark.skipif(
  not UNICYCLE_LIBRARY, reason="set MJLAB_OSL_UNICYCLE_LIBRARY to a library dir"
)
def test_fsm_env_rollout() -> None:
  """The stage-1.5 env on the real library, 40 steps of zero actions (the human
  falls, so auto-resets come too): the controller reads exactly the clean osl
  group's newest frame (the observation from the step before, post-reset frames
  included), MuJoCo receives its ctrl and applies it through the randomized motor
  gain, its states change, and the host model it takes its constants from is the
  nominal one."""
  cfg = osl_fsm_env_cfg("msk", "compliant", "clf", "train")
  n = 8
  cfg.scene.num_envs = n
  motion = cfg.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = UNICYCLE_LIBRARY
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  term = env.action_manager.get_term("osl_motors")
  assert isinstance(term, mdp.OslFsmAction)

  # The constants come from the host model, which the randomization left nominal.
  nominal = Scene(cfg.scene, "cpu").compile()
  cfg.sim.mujoco.apply(nominal)
  mj_model = env.sim.mj_model
  for name in ("body_mass", "actuator_gainprm", "actuator_gear", "body_inertia"):
    np.testing.assert_array_equal(getattr(mj_model, name), getattr(nominal, name))
  motors = _motor_ids(mj_model, term.target_names)  # The term's column order.
  drawn = env.sim.model.actuator_gainprm[:, motors, 0].cpu()
  nominal_gains = torch.as_tensor(
    nominal.actuator_gainprm[motors, 0], dtype=drawn.dtype
  )
  assert not torch.allclose(drawn, nominal_gains)  # The draws are the plant's.
  settings = term.settings()
  assert settings["tare_n"] == pytest.approx(19.9143, abs=1e-3)
  assert settings["nm_per_ctrl"] == pytest.approx(NM_PER_CTRL)
  assert settings["body_mass_kg"] == pytest.approx(float(nominal.body_mass.sum()))
  assert settings["body_weight_n"] == pytest.approx(807.2557, abs=1e-3)

  # The osl group's newest frame of each reading's channel.
  obs_cfg = split_mdp.OslObsCfg()
  history = obs_cfg.history_length
  actpack = split_mdp.osl_actpack_dim(obs_cfg)
  newest = (history - 1) * actpack
  loadcell = history * actpack + (history - 1) * split_mdp.osl_loadcell_dim(obs_cfg)
  # actpack frame: knee, ankle, knee velocity, ankle velocity, ...; loadcell: the
  # ankle cell's force (x, y, z) first.
  columns = [newest, newest + 2, newest + 1, newest + 3, loadcell + 1]
  assert len(columns) == len(mdp.READINGS)

  obs, _ = env.reset()
  zero = torch.zeros(n, env.action_manager.total_action_dim, device=device)
  states = []
  for step in range(40):
    if step == 10:  # A reset whatever the falls: its envs read their start frame.
      obs, _ = env.reset(env_ids=torch.tensor([0, 1], device=device))
      assert term.state[:2].tolist() == [core.E_STANCE] * 2
    osl = obs["osl"]
    assert isinstance(osl, torch.Tensor) and osl.shape == (n, 600)
    seen = osl[:, columns].clone()
    obs, _, terminated, truncated, _ = env.step(zero)
    # Bit for bit: the same sensors, the same float32 tare. The reading of an env
    # that ended its episode within the step stays, as its diagnostics do.
    if step == 10 and device == "cuda":
      # The partial reset re-ran the forward for every env, while the envs it did
      # not reset keep their previous newest frame. On the GPU that forward
      # re-solves their contacts from another warm start (mujoco_warp is not
      # bitwise reproducible), so their load reads may differ by up to ~0.1 N:
      # compare the reset envs only. A training step never runs a forward between
      # the observation and the next control step (verified bit-exact over 200
      # steps x 64 envs with auto-resets).
      torch.testing.assert_close(term.readings[:2], seen[:2], rtol=0.0, atol=0.0)
    else:
      torch.testing.assert_close(term.readings, seen, rtol=0.0, atol=0.0)
    states.append(term.state.clone())
    # MuJoCo's ctrl is the held command; an env reset within the step had both
    # zeroed, and its post-reset forward applied that zero.
    ctrl = env.sim.data.ctrl[:, motors]
    torch.testing.assert_close(ctrl, term.ctrl)
    reset = terminated | truncated
    assert not ctrl[reset].any()
    force = env.sim.data.actuator_force[:, motors]
    expected = ctrl * env.sim.model.actuator_gainprm[:, motors, 0]
    torch.testing.assert_close(force, expected)
  visited = torch.stack(states)
  assert bool((visited != visited[:1]).any()), "no state changed in 40 steps"
  env.close()
