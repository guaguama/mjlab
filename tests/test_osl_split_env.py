"""Tests for the stage-2 split environment (``mjlab.tasks.osl_split``): the OSL
sensors, the OSL actor's observation terms and the env config.

CPU only. The sensors are compiled with plain MuJoCo; the observation terms run on a
fake env, alone and through mjlab's ``ObservationManager``; the configs are compared
term for term with stage 1's. Only the slow test builds the real env
(``MJLAB_OSL_UNICYCLE_LIBRARY``).
"""

from __future__ import annotations

import copy
import math
import os
import re
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
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
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
  ObservationTermCfg,
)
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_gait_library.config.dr import FOOT_GEOMS, OslDrCfg
from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg
from mjlab.tasks.osl_gait_library.export import read_config
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward
from mjlab.tasks.osl_mimic.mdp import mimic_component
from mjlab.tasks.osl_split import mdp
from mjlab.tasks.osl_split.config import sensors
from mjlab.tasks.osl_split.config.env_cfgs import (
  OslSplitEnvCfg,
  osl_split_env_cfg,
  sync_osl_group,
)
from mjlab.utils.buffers import DelayBuffer
from mjlab.utils.os import dump_yaml

MODELS: tuple[Model, ...] = ("robot", "msk")
SOCKETS: tuple[Socket, ...] = ("rigid", "compliant")
REWARDS: tuple[Reward, ...] = ("mimic", "clf")
MODES: tuple[Mode, ...] = ("train", "play", "validation")
PHYSICAL_EVENTS = {"torso_inertia", "foot_friction", "joint_armature"}
"""Stage 1's startup draws, with ``osl_motor_strength`` and each model's own."""

CELLS = (  # Site, force, torque, and the hanging reading along +y (N).
  (sensors.ANKLE_LOAD_SITE, sensors.ANKLE_FORCE, sensors.ANKLE_LOAD_TORQUE, 19.9),
  (sensors.SOCKET_LOAD_SITE, sensors.SOCKET_FORCE, sensors.SOCKET_TORQUE, 44.5),
)
"""2.03 kg hang below the ankle cell and 4.54 kg below the socket cell."""
ACTPACK_STDS = (0.002,) * 2 + (0.05,) * 2 + (0.02,) * 3 + (0.1,) * 3
LOADCELL_STDS = ((1.0,) * 3 + (0.1,) * 3) * 2
"""Both cells'; the default set has the ankle cell's, the first six."""
CONTACT_STDS = (1.0, 1.0)
ALL_SPECS = (*sensors.OSL_SENSOR_SPECS, *sensors.OSL_CONTACT_SPECS)
"""Every sensor an ``osl`` term can read: the added ones and the foot contacts."""
SCENE_NAMES = {spec.name: sensors.scene_sensor_name(spec.sensor) for spec in ALL_SPECS}
"""Each sensor's scene name: ``robot/<name>``, and the contacts' ``robot/r_foot``
and ``robot/r_toes``."""
EXTRA_STDS = (0.01,) * 3 + (1.0,) * 2
"""Projected gravity's, then the applied torque's."""
QUIET_ACTPACK = dict.fromkeys(
  (
    "encoder_pos_noise",
    "encoder_vel_noise",
    "gyro_noise",
    "acc_noise",
    "gravity_noise",
    "applied_torque_noise",
  ),
  0.0,
)
"""The actpack's noise stds at zero, to see its per-episode errors alone."""
QUIET_LOADCELL = dict.fromkeys(("force_noise", "torque_noise"), 0.0)

##
# Sensors.
##


def _spec_fn(model: Model, socket: Socket) -> Callable[[], mujoco.MjSpec]:
  return lambda: get_spec(model, socket)


@pytest.fixture(scope="module")
def scene_model() -> mujoco.MjModel:
  """The robot with the OSL sensors attached under ``robot/`` over a floor, as
  mjlab's scene attaches entities."""
  scene = mujoco.MjSpec()
  scene.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(5.0, 5.0, 0.1))
  entity = sensors.with_osl_sensors(_spec_fn("robot", "rigid"))()
  scene.attach(entity, prefix="robot/", frame=scene.worldbody.add_frame())
  return scene.compile()


@pytest.mark.parametrize(
  "model, socket", [("robot", "rigid"), ("robot", "compliant"), ("msk", "compliant")]
)
def test_sensors_compile_on_every_model(model: Model, socket: Socket) -> None:
  stage1_sensors = len(get_spec(model, socket).sensors)
  compiled = sensors.with_osl_sensors(_spec_fn(model, socket))().compile()
  assert compiled.nsensor == stage1_sensors + len(sensors.OSL_SENSORS)
  for touch in ("r_foot", "r_toes", "l_foot", "l_toes"):  # The human's, untouched.
    assert compiled.sensor(touch).type[0] == mujoco.mjtSensor.mjSENS_TOUCH
  for spec in sensors.OSL_SENSOR_SPECS:
    sensor = compiled.sensor(spec.name)
    assert int(sensor.dim[0]) == spec.dim and int(sensor.type[0]) == spec.type
    assert int(sensor.objtype[0]) == spec.objtype
    target = mujoco.mj_id2name(compiled, spec.objtype, int(sensor.objid[0]))
    assert target == spec.objname
    # No MuJoCo saturation, the IMU's included: the observation saturates.
    assert float(sensor.cutoff[0]) == 0.0
  assert sensors.GYRO_RANGE == pytest.approx(math.radians(1000.0))
  assert sensors.ACC_RANGE == pytest.approx(4 * 9.80665)
  site = compiled.site(sensors.OSL_IMU_SITE)
  assert compiled.body(int(site.bodyid[0])).name == sensors.OSL_IMU_BODY
  np.testing.assert_allclose(site.pos, sensors.OSL_IMU_POS)
  np.testing.assert_allclose(site.quat, (1.0, 0.0, 0.0, 0.0))

  # The hanging leg: each cell carries exactly its distal weight, along +y (up the
  # limb) in the upright reference pose, with almost no moment.
  tare = sensors.static_load_cell_tare(compiled, prefix="")
  assert set(tare) == set(sensors.LOAD_CELL_SENSORS)
  gravity = -float(compiled.opt.gravity[2])
  for site_name, force, torque, hanging in CELLS:
    body = int(compiled.site(site_name).bodyid[0])
    weight = float(compiled.body_subtreemass[body]) * gravity
    assert np.linalg.norm(tare[force]) == pytest.approx(weight, rel=1e-6)
    assert tare[force][1] == pytest.approx(hanging, abs=0.1)
    assert np.abs(tare[force][[0, 2]]).max() < 0.05
    assert np.abs(tare[torque]).max() < 0.5


def test_add_osl_sensors_refuses_twice_and_other_models() -> None:
  spec = sensors.add_osl_sensors(get_spec("robot", "rigid"))
  with pytest.raises(ValueError, match="already has a site"):
    sensors.add_osl_sensors(spec)
  with pytest.raises(ValueError, match="OSL"):
    sensors.add_osl_sensors(mujoco.MjSpec())
  # A sensor added after get_spec's deletions, which MjSpec's name lookup misses.
  spec = get_spec("robot", "rigid")
  spec.add_sensor(
    name=sensors.KNEE_POS,
    type=mujoco.mjtSensor.mjSENS_JOINTPOS,
    objtype=mujoco.mjtObj.mjOBJ_JOINT,
    objname=OSL_TORQUE_JOINTS[0],
  )
  assert spec.sensor(sensors.KNEE_POS) is None  # The MuJoCo 3.11 quirk.
  with pytest.raises(ValueError, match="already has the sensors"):
    sensors.add_osl_sensors(spec)


def test_scene_prefix_and_tare(scene_model: mujoco.MjModel) -> None:
  """mjlab's ``robot/`` prefix reaches the site and every sensor, and the tare is
  the bare entity's: the floor and the scene change nothing."""
  for name in (*sensors.OSL_SENSORS, sensors.OSL_IMU_SITE):
    assert sensors.scene_sensor_name(name) == f"robot/{name}"
  for name in sensors.OSL_SENSORS:
    assert scene_model.sensor(sensors.scene_sensor_name(name)).id >= 0
  assert scene_model.site(sensors.scene_sensor_name(sensors.OSL_IMU_SITE)).id >= 0
  bare = sensors.with_osl_sensors(_spec_fn("robot", "rigid"))().compile()
  in_scene = sensors.static_load_cell_tare(scene_model)
  for name, reading in sensors.static_load_cell_tare(bare, prefix="").items():
    np.testing.assert_allclose(in_scene[name], reading, atol=1e-9)
  assert scene_model.opt.disableflags == 0  # The tare works on a copy.


##
# Latency: one lag per env per episode.
##


def test_delay_hold_prob_one_holds_one_lag_per_episode() -> None:
  """mjlab's DelayBuffer, driven as the observation manager drives it: with
  ``hold_prob=1`` each env keeps the lag it drew until its next reset."""
  n = 64
  buffer = DelayBuffer(
    min_lag=0,
    max_lag=2,
    batch_size=n,
    per_env=True,
    hold_prob=1.0,
    generator=torch.Generator().manual_seed(0),
  )
  frame = torch.zeros(n, 1)

  def step() -> torch.Tensor:
    buffer.append(frame)
    buffer.compute()
    return buffer.current_lags.clone()

  first = step()
  assert set(first.tolist()) == {0, 1, 2}
  for _ in range(30):
    torch.testing.assert_close(step(), first)
  buffer.reset(batch_ids=torch.arange(n // 2))
  redrawn = step()
  torch.testing.assert_close(redrawn[n // 2 :], first[n // 2 :])
  assert not torch.equal(redrawn[: n // 2], first[: n // 2])
  for _ in range(10):
    torch.testing.assert_close(step(), redrawn)


##
# The observation terms on a fake env.
##


def _fake_env(model: mujoco.MjModel, num_envs: int = 4) -> Any:
  """Every OSL sensor as a ``.data`` tensor, the site rotations and the OSL action
  term's raw command; the real compiled model for name lookups and the tare."""
  scene = {
    SCENE_NAMES[spec.name]: SimpleNamespace(data=torch.zeros(num_envs, spec.dim))
    for spec in ALL_SPECS
  }
  site_xmat = torch.eye(3).expand(num_envs, model.nsite, 3, 3).clone()
  action = SimpleNamespace(raw_action=torch.zeros(num_envs, 2))
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=scene,
    sim=SimpleNamespace(mj_model=model, data=SimpleNamespace(site_xmat=site_xmat)),
    action_manager=SimpleNamespace(get_term=lambda name: action),
    action=action,
  )


def _set(env: Any, name: str, value: torch.Tensor | list[float]) -> None:
  data = torch.as_tensor(value, dtype=torch.float32)
  env.scene[SCENE_NAMES[name]].data = data.expand(env.num_envs, -1)


def _actpack(env: Any, **params: Any) -> mdp.osl_actpack:
  return mdp.osl_actpack(ObservationTermCfg(func=mdp.osl_actpack, params=params), env)


def _load_cells(env: Any, **params: Any) -> mdp.osl_load_cells:
  cfg = ObservationTermCfg(func=mdp.osl_load_cells, params=params)
  return mdp.osl_load_cells(cfg, env)


def _contact(env: Any, **params: Any) -> mdp.osl_foot_contact:
  cfg = ObservationTermCfg(func=mdp.osl_foot_contact, params=params)
  return mdp.osl_foot_contact(cfg, env)


def _changed_rows(now: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
  """(N,) whether each row differs anywhere."""
  return (now != before).flatten(1).any(dim=1)


def test_actpack_reads_the_sensors_exactly_without_randomization(
  scene_model: mujoco.MjModel,
) -> None:
  env = _fake_env(scene_model)
  _set(env, sensors.KNEE_POS, [0.4])
  _set(env, sensors.ANKLE_POS, [-0.1])
  _set(env, sensors.KNEE_VEL, [2.0])
  _set(env, sensors.ANKLE_VEL, [-1.5])
  _set(env, sensors.IMU_GYRO, [0.1, -0.2, 0.3])
  _set(env, sensors.IMU_ACC, [1.0, 9.8, -0.5])
  term = _actpack(env, randomize=False)
  expected = torch.tensor([0.4, -0.1, 2.0, -1.5, 0.1, -0.2, 0.3, 1.0, 9.8, -0.5])
  torch.testing.assert_close(term(env), expected.expand(4, -1))
  term.reset(torch.tensor([0, 2]))
  torch.testing.assert_close(term(env), expected.expand(4, -1))
  torch.testing.assert_close(term.mount, torch.eye(3).expand(4, 3, 3))


def test_actpack_draws_per_episode_errors_on_reset(
  scene_model: mujoco.MjModel,
) -> None:
  torch.manual_seed(0)
  n = 8
  env = _fake_env(scene_model, num_envs=n)
  term = _actpack(
    env,
    randomize=True,
    encoder_bias_rad=0.015,
    gyro_bias=0.02,
    acc_bias=0.2,
    imu_mount_deg=3.0,
    **QUIET_ACTPACK,
  )
  for bias, bound in (
    (term.encoder_bias, 0.015),
    (term.gyro_bias, 0.02),
    (term.acc_bias, 0.2),
  ):
    assert float(bias.abs().max()) <= bound and float(bias.abs().min()) > 0.0
  # A proper rotation within 3 deg about each axis, so within 9 deg in all.
  torch.testing.assert_close(
    term.mount @ term.mount.transpose(-1, -2), torch.eye(3).expand(n, 3, 3)
  )
  torch.testing.assert_close(torch.linalg.det(term.mount), torch.ones(n))
  trace = term.mount.diagonal(dim1=-2, dim2=-1).sum(-1)
  angle = torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
  assert float(angle.max()) <= math.radians(9.0) and float(angle.min()) > 0.0

  gyro = torch.tensor([0.1, -0.2, 0.3])
  acc = torch.tensor([1.0, 9.8, -0.5])
  _set(env, sensors.IMU_GYRO, gyro)
  _set(env, sensors.IMU_ACC, acc)
  _set(env, sensors.KNEE_POS, [0.4])
  _set(env, sensors.KNEE_VEL, [2.0])
  out = term(env)
  to_imu = term.mount.transpose(-1, -2)
  torch.testing.assert_close(out[:, 0], 0.4 + term.encoder_bias[:, 0])
  torch.testing.assert_close(out[:, 2], torch.full((n,), 2.0))  # No velocity bias.
  torch.testing.assert_close(out[:, 4:7], to_imu @ gyro + term.gyro_bias)
  torch.testing.assert_close(out[:, 7:10], to_imu @ acc + term.acc_bias)

  names = ("encoder_bias", "gyro_bias", "acc_bias", "mount")
  before = {name: getattr(term, name).clone() for name in names}
  term.reset(torch.tensor([1, 5]))
  reset = torch.zeros(n, dtype=torch.bool)
  reset[[1, 5]] = True
  for name, value in before.items():
    torch.testing.assert_close(_changed_rows(getattr(term, name), value), reset)


def test_actpack_saturates_after_mount_and_bias(scene_model: mujoco.MjModel) -> None:
  torch.manual_seed(1)
  env = _fake_env(scene_model)
  term = _actpack(env, randomize=True, **QUIET_ACTPACK)
  _set(env, sensors.IMU_GYRO, [100.0, -100.0, 0.0])
  _set(env, sensors.IMU_ACC, [0.0, 1000.0, -1000.0])
  out = term(env)
  gyro, acc = out[:, 4:7], out[:, 7:10]
  limits = torch.tensor([1.0, -1.0]).expand(4, -1)
  torch.testing.assert_close(gyro[:, :2], sensors.GYRO_RANGE * limits)
  torch.testing.assert_close(acc[:, 1:], sensors.ACC_RANGE * limits)
  assert float(gyro[:, 2].abs().max()) < sensors.GYRO_RANGE  # The mount's leak.


def test_imu_saturates_once_from_mujoco_to_observation(
  scene_model: mujoco.MjModel,
) -> None:
  """MuJoCo's IMU reads past the ActPack's ranges unclipped, so the term's clamp,
  after the mount rotation and the bias, is the only saturation. A MuJoCo cutoff
  would clip the site-frame vector first: a saturated axis would then read the
  range minus the bias, and the mount would rotate an already clipped vector."""
  data = mujoco.MjData(scene_model)
  data.qpos[:] = scene_model.qpos0
  # The whole model spinning about its root at 30 rad/s: the gyro reads about
  # that, the accelerometer the centripetal acceleration.
  dof = int(scene_model.joint("robot/root").dofadr[0])
  data.qvel[dof + 3 : dof + 6] = (0.0, 0.0, 30.0)
  mujoco.mj_inverse(scene_model, data)

  def raw(name: str) -> torch.Tensor:
    sensor = scene_model.sensor(sensors.scene_sensor_name(name))
    adr, dim = int(sensor.adr[0]), int(sensor.dim[0])
    return torch.as_tensor(data.sensordata[adr : adr + dim], dtype=torch.float32)

  gyro, acc = raw(sensors.IMU_GYRO), raw(sensors.IMU_ACC)
  assert float(gyro.abs().max()) > 1.5 * sensors.GYRO_RANGE
  assert float(acc.abs().max()) > 1.5 * sensors.ACC_RANGE

  env = _fake_env(scene_model, num_envs=1)
  _set(env, sensors.IMU_GYRO, gyro)
  _set(env, sensors.IMU_ACC, acc)
  term = _actpack(env, randomize=False)
  # One episode's errors, set by hand: 3 deg about z, biases along y.
  c, s = math.cos(math.radians(3.0)), math.sin(math.radians(3.0))
  term.mount[:] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  term.gyro_bias[:] = torch.tensor([0.0, -0.02, 0.0])
  term.acc_bias[:] = torch.tensor([0.0, -0.2, 0.0])
  out = term(env)[0]
  to_imu = term.mount[0].T
  for reading, vector, bias, bound in (
    (out[4:7], gyro, term.gyro_bias[0], sensors.GYRO_RANGE),
    (out[7:10], acc, term.acc_bias[0], sensors.ACC_RANGE),
  ):
    torch.testing.assert_close(reading, (to_imu @ vector + bias).clamp(-bound, bound))
    assert float(reading.abs().max()) == pytest.approx(bound)  # Saturated.


def test_actpack_extras(scene_model: mujoco.MjModel) -> None:
  torch.manual_seed(2)
  env = _fake_env(scene_model)
  term = _actpack(
    env, randomize=True, projected_gravity=True, applied_torque=True, **QUIET_ACTPACK
  )
  _set(env, sensors.KNEE_TORQUE, [12.0])
  _set(env, sensors.ANKLE_TORQUE, [-30.0])
  # The OSL frames in the upright reference pose: x forward (world -y), y up the
  # limb, z to the right (world -x). A level mount reads gravity along -y.
  site = scene_model.site(sensors.scene_sensor_name(sensors.OSL_IMU_SITE)).id
  upright = torch.tensor([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
  env.sim.data.site_xmat[:, site] = upright
  out = term(env)
  assert out.shape == (4, 15)
  to_imu = term.mount.transpose(-1, -2)
  torch.testing.assert_close(out[:, 10:13], to_imu @ torch.tensor([0.0, -1.0, 0.0]))
  torch.testing.assert_close(out[:, 10:13].norm(dim=-1), torch.ones(4))
  torch.testing.assert_close(out[:, 13:], torch.tensor([12.0, -30.0]).expand(4, -1))


def test_load_cells_subtract_the_tare(scene_model: mujoco.MjModel) -> None:
  torch.manual_seed(3)
  n = 6
  env = _fake_env(scene_model, num_envs=n)
  tare = sensors.static_load_cell_tare(scene_model)
  for name, reading in tare.items():
    _set(env, name, torch.as_tensor(reading))
  both = _load_cells(env, socket=True, randomize=False)
  assert both.tare.shape == (12,)
  torch.testing.assert_close(both(env), torch.zeros(n, 12), atol=1e-5, rtol=0.0)
  ankle = _load_cells(env, socket=False, randomize=False)
  torch.testing.assert_close(ankle(env), torch.zeros(n, 6), atol=1e-5, rtol=0.0)
  socket = _load_cells(env, ankle=False, socket=True, randomize=False)
  assert socket.tare.shape == (6,)
  torch.testing.assert_close(socket.tare, both.tare[6:])
  torch.testing.assert_close(socket(env), torch.zeros(n, 6), atol=1e-5, rtol=0.0)
  with pytest.raises(ValueError, match="at least one load cell"):
    _load_cells(env, ankle=False, socket=False, randomize=False)
  # Loaded in stance, a tared cell reads minus the transmitted load.
  _set(env, sensors.ANKLE_FORCE, torch.as_tensor(tare[sensors.ANKLE_FORCE]) - 800.0)
  torch.testing.assert_close(ankle(env)[:, :3], torch.full((n, 3), -800.0))

  term = _load_cells(
    env, socket=True, randomize=True, force_bias=2.0, torque_bias=0.2, **QUIET_LOADCELL
  )
  bound = torch.tensor([2.0] * 3 + [0.2] * 3).repeat(2)
  assert bool((term.bias.abs() <= bound).all()) and bool((term.bias != 0).all())
  torch.testing.assert_close(term(env)[:, 3:], term.bias[:, 3:], atol=1e-5, rtol=0.0)
  before = term.bias.clone()
  term.reset(torch.tensor([0, 4]))
  reset = torch.tensor([True, False, False, False, True, False])
  torch.testing.assert_close(_changed_rows(term.bias, before), reset)


def test_terms_add_their_own_white_noise_when_randomized(
  scene_model: mujoco.MjModel,
) -> None:
  """Each class term adds one Gaussian draw per component and sample, with stds
  that follow its own switches, and only when ``randomize``. The noise comes after
  the IMU's clamp, where the observation manager used to add it."""
  torch.manual_seed(5)
  n = 2048
  env = _fake_env(scene_model, num_envs=n)
  _set(env, sensors.IMU_ACC, [0.0, 1000.0, 0.0])  # Saturated along y.
  extras = {"projected_gravity": True, "applied_torque": True}
  actpack = _actpack(env, randomize=True, **extras)
  for term, stds in (
    (actpack, ACTPACK_STDS + EXTRA_STDS),
    (_load_cells(env, randomize=True, socket=False), LOADCELL_STDS[:6]),
  ):
    assert term.noise_std.tolist() == pytest.approx(stds)
    noise = term(env) - term(env)  # The same errors, two draws.
    torch.testing.assert_close(
      noise.std(dim=0) / math.sqrt(2.0), torch.tensor(stds), rtol=0.1, atol=0.0
    )
  acc_y = actpack(env)[:, 8]
  assert abs(float(acc_y.mean()) - sensors.ACC_RANGE) < 0.02
  assert float(acc_y.max()) > sensors.ACC_RANGE  # The noise rides on the clamp.

  clean = _actpack(env, randomize=False, **extras)
  assert clean.noise_std.tolist() == pytest.approx(ACTPACK_STDS + EXTRA_STDS)
  assert torch.equal(clean(env), clean(env))


def test_last_action_is_the_clamped_raw_command(scene_model: mujoco.MjModel) -> None:
  env = _fake_env(scene_model, num_envs=2)
  env.action.raw_action = torch.tensor([[2.0, -3.0], [0.5, -0.25]])
  out = mdp.osl_last_action(env)
  torch.testing.assert_close(out, torch.tensor([[1.0, -1.0], [0.5, -0.25]]))


def test_contact_sensors_are_the_models_own(scene_model: mujoco.MjModel) -> None:
  """The foot contacts read the model XML's r_foot and r_toes touch sensors on the
  OSL foot's touch sites; nothing is added for them."""
  assert [s.name for s in sensors.OSL_CONTACT_SPECS] == [
    sensors.HEEL_CONTACT,
    sensors.TOE_CONTACT,
  ]
  for spec in sensors.OSL_CONTACT_SPECS:
    sensor = scene_model.sensor(sensors.scene_sensor_name(spec.sensor))
    assert sensor.type[0] == mujoco.mjtSensor.mjSENS_TOUCH == spec.type
    assert sensor.objtype[0] == mujoco.mjtObj.mjOBJ_SITE and sensor.dim[0] == 1
    site = mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_SITE, sensor.objid[0])
    assert site == f"robot/{spec.objname}"
    assert spec.name not in sensors.OSL_SENSORS
  assert [s.sensor for s in sensors.OSL_CONTACT_SPECS] == ["r_foot", "r_toes"]


def test_contact_term_reads_the_touch_sensors_raw(
  scene_model: mujoco.MjModel,
) -> None:
  """Heel then toe, raw (no tare, scale or clamp); a switch leaves its channel out.
  Randomized: a per-episode bias within +-contact_bias, redrawn only for the envs
  that reset, then white noise of std contact_noise."""
  n = 6
  env = _fake_env(scene_model, num_envs=n)
  _set(env, sensors.HEEL_CONTACT, [350.0])
  _set(env, sensors.TOE_CONTACT, [0.0])
  both = _contact(env, randomize=False)
  torch.testing.assert_close(both(env), torch.tensor([350.0, 0.0]).expand(n, -1))
  toe = _contact(env, heel=False, randomize=False)
  torch.testing.assert_close(toe(env), torch.zeros(n, 1))
  _set(env, sensors.TOE_CONTACT, [1200.0])  # No saturation.
  torch.testing.assert_close(toe(env), torch.full((n, 1), 1200.0))
  with pytest.raises(ValueError, match="at least one foot contact"):
    _contact(env, heel=False, toe=False)
  assert mdp.term_channels("contact", {}) == [
    (sensors.HEEL_CONTACT, "N", sensors.HEEL_CONTACT),
    (sensors.TOE_CONTACT, "N", sensors.TOE_CONTACT),
  ]

  torch.manual_seed(7)
  term = _contact(env, randomize=True, contact_bias=2.0, contact_noise=0.0)
  assert bool((term.bias.abs() <= 2.0).all()) and bool((term.bias != 0).all())
  torch.testing.assert_close(term(env), torch.tensor([350.0, 1200.0]) + term.bias)
  before = term.bias.clone()
  term.reset(torch.tensor([1, 3]))
  reset = torch.tensor([False, True, False, True, False, False])
  torch.testing.assert_close(_changed_rows(term.bias, before), reset)

  many = _fake_env(scene_model, num_envs=4096)
  noisy = _contact(many, randomize=True, contact_bias=0.0)
  assert noisy.noise_std.tolist() == pytest.approx(CONTACT_STDS)
  torch.testing.assert_close(
    noisy(many).std(dim=0), torch.tensor(CONTACT_STDS), rtol=0.05, atol=0.0
  )


##
# The observation group.
##


@pytest.mark.parametrize("mode", MODES)
def test_observation_group_by_mode(mode: Mode) -> None:
  cfg = mdp.OslObsCfg()
  group = mdp.osl_observation_group(mode, cfg)
  train = mode == "train"
  assert (
    tuple(group.terms)
    == mdp.OSL_OBS_TERMS
    == ("actpack", "loadcell", "contact", "last_action")
  )
  assert mdp.OSL_BUS_TERMS == ("actpack", "loadcell", "contact")
  assert group.concatenate_terms and group.history_length is None
  assert group.enable_corruption == train
  for term in group.terms.values():
    assert term.history_length == 25 and term.flatten_history_dim
  lags = (1, 1) if mode == "validation" else (0, 2)
  for name in mdp.OSL_BUS_TERMS:
    term = group.terms[name]
    assert (term.delay_min_lag, term.delay_max_lag) == lags
    assert term.delay_per_env and term.delay_hold_prob == 1.0
    assert term.delay_update_period == 0
    # The term adds its own noise, under randomize; the stds are always there.
    assert term.params["randomize"] == train
    assert term.noise is None
    assert (
      mdp.osl_noise_std(name, term.params)
      == {
        "actpack": ACTPACK_STDS,
        "loadcell": LOADCELL_STDS[:6],
        "contact": CONTACT_STDS,
      }[name]
    )
  # The real leg's set: the ankle cell, not the socket's, and both contacts.
  loadcell, contact = group.terms["loadcell"].params, group.terms["contact"].params
  assert loadcell["ankle"] and not loadcell["socket"]
  assert contact["heel"] and contact["toe"]
  assert contact["contact_bias"] == 2.0 and contact["contact_noise"] == 1.0
  assert mdp.osl_sensor_set(group) == {
    "encoders": True,
    "imu": True,
    "ankle_load_cell": True,
    "socket_load_cell": False,
    "heel_contact": True,
    "toe_contact": True,
  }
  last = group.terms["last_action"]
  assert last.func is mdp.osl_last_action
  assert last.params == {"action_name": "osl_motors"}
  assert last.noise is None and last.delay_max_lag == 0
  assert mdp.osl_noise_std("last_action", last.params) is None
  assert mdp.osl_actpack_dim(cfg) == 10 and mdp.osl_loadcell_dim(cfg) == 6
  assert mdp.osl_contact_dim(cfg) == 2
  assert mdp.osl_obs_frame_dim(cfg) == 20 and mdp.osl_obs_dim(cfg) == 500
  legacy = mdp.OslObsCfg(**mdp.LEGACY_OSL_SENSORS)
  assert mdp.osl_obs_dim(legacy) == 600


def test_observation_options() -> None:
  cfg = mdp.OslObsCfg(
    projected_gravity=True,
    applied_torque=True,
    socket_load_cell=True,
    toe_contact=False,
  )
  group = mdp.osl_observation_group("train", cfg)
  actpack, loadcell = group.terms["actpack"], group.terms["loadcell"]
  contact = group.terms["contact"]
  assert actpack.params["projected_gravity"] and actpack.params["applied_torque"]
  assert loadcell.params["socket"] and not contact.params["toe"]
  stds = mdp.osl_noise_std("actpack", actpack.params)
  assert stds == ACTPACK_STDS + EXTRA_STDS
  assert stds is not None and len(stds) == mdp.osl_actpack_dim(cfg) == 15
  assert mdp.osl_noise_std("loadcell", loadcell.params) == LOADCELL_STDS
  assert mdp.osl_noise_std("contact", contact.params) == CONTACT_STDS[:1]
  assert mdp.osl_loadcell_dim(cfg) == 12 and mdp.osl_contact_dim(cfg) == 1
  assert mdp.osl_obs_frame_dim(cfg) == 30
  # A sensor term whose sensors are all off is left out (the drop rule).
  drops: tuple[tuple[dict[str, Any], str], ...] = (
    ({"heel_contact": False, "toe_contact": False}, "contact"),
    ({"ankle_load_cell": False}, "loadcell"),
  )
  for off, gone in drops:
    dropped = mdp.osl_observation_group("train", mdp.OslObsCfg(**off))
    assert gone not in dropped.terms and len(dropped.terms) == 3
  quiet = mdp.OslObsCfg(gyro_noise=0.0, acc_noise=0.0)
  actpack = mdp.osl_observation_group("train", quiet).terms["actpack"]
  assert mdp.osl_noise_std("actpack", actpack.params) == ACTPACK_STDS[:4] + (0.0,) * 6
  with pytest.raises(ValueError, match="delay_steps"):
    mdp.OslObsCfg(delay_steps=(2, 1))
  with pytest.raises(ValueError, match="history_length"):
    mdp.OslObsCfg(history_length=-1)
  with pytest.raises(ValueError, match="force_noise"):
    mdp.OslObsCfg(force_noise=-1.0)
  with pytest.raises(ValueError, match="mode"):
    mdp.osl_observation_group(cast(Mode, "eval"))


def _observe(manager: ObservationManager, env: Any, value: float) -> torch.Tensor:
  """Set every sensor and the OSL command to ``value`` and run one step's
  observation."""
  for spec in ALL_SPECS:
    _set(env, spec.name, torch.full((spec.dim,), value))
  env.action.raw_action = torch.full((env.num_envs, 2), value)
  out = manager.compute(update_history=True)["osl"]
  assert isinstance(out, torch.Tensor)
  return out


def test_osl_group_layout_through_the_observation_manager(
  scene_model: mujoco.MjModel,
) -> None:
  """Validation: clean readings at a one-step lag, term-major and oldest to newest
  per term; the command is immediate."""
  n = 3
  env = _fake_env(scene_model, num_envs=n)
  cfg = mdp.OslObsCfg(history_length=4)
  group = mdp.osl_observation_group("validation", cfg)
  manager = ObservationManager({"osl": group}, env)
  assert manager.group_obs_dim["osl"] == (mdp.osl_obs_dim(cfg),) == (80,)
  for step in range(5):
    _observe(manager, env, 0.01 * step)
  out = _observe(manager, env, 0.05)
  actpack = out[:, :40].reshape(n, 4, 10)
  loadcell = out[:, 40:64].reshape(n, 4, 6)
  contact = out[:, 64:72].reshape(n, 4, 2)
  command = out[:, 72:].reshape(n, 4, 2)
  # Steps 2 to 5 fill the history: the buses show steps 1 to 4, the command 2 to 5.
  bus_steps = 0.01 * torch.arange(1, 5, dtype=torch.float32)
  ankle_fx_tare = float(
    sensors.static_load_cell_tare(scene_model)[sensors.ANKLE_FORCE][0]
  )
  torch.testing.assert_close(actpack[:, :, 0], bus_steps.expand(n, -1))
  torch.testing.assert_close(
    loadcell[:, :, 0], (bus_steps - ankle_fx_tare).expand(n, -1)
  )
  # The contacts are raw (no tare) and lag on their own bus.
  torch.testing.assert_close(contact, bus_steps[:, None].expand(n, 4, 2))
  command_steps = 0.01 * torch.arange(2, 6, dtype=torch.float32)
  torch.testing.assert_close(command[:, :, 0], command_steps.expand(n, -1))


def test_osl_group_randomization_through_the_observation_manager(
  scene_model: mujoco.MjModel,
) -> None:
  """Training: each bus holds one lag per env per episode, independently of the
  other; a reset redraws the reset envs' lags and errors only; the noise has the
  configured per-component stds and reaches every frame of the history."""
  torch.manual_seed(4)
  n = 256
  env = _fake_env(scene_model, num_envs=n)
  manager = ObservationManager({"osl": mdp.osl_observation_group("train")}, env)
  delay = manager._group_obs_term_delay_buffer["osl"]
  assert set(delay) == {"actpack", "loadcell", "contact"}
  actpack = manager._group_obs_term_cfgs["osl"][0].func
  assert isinstance(actpack, mdp.osl_actpack)

  _observe(manager, env, 0.0)
  lags = {name: buffer.current_lags.clone() for name, buffer in delay.items()}
  assert set(lags["actpack"].tolist()) == {0, 1, 2}
  assert not torch.equal(lags["actpack"], lags["loadcell"])
  assert not torch.equal(lags["contact"], lags["loadcell"])
  for step in range(1, 10):
    _observe(manager, env, 0.01 * step)
    for name, buffer in delay.items():
      torch.testing.assert_close(buffer.current_lags, lags[name])

  reset = torch.arange(n) < n // 2
  bias = actpack.encoder_bias.clone()
  manager.reset(env_ids=torch.arange(n // 2))
  _observe(manager, env, 0.1)
  torch.testing.assert_close(_changed_rows(actpack.encoder_bias, bias), reset)
  for name, buffer in delay.items():
    torch.testing.assert_close(buffer.current_lags[~reset], lags[name][~reset])
    assert not torch.equal(buffer.current_lags[reset], lags[name][reset])

  # Constant readings and no reset: an env's frames differ by their noise draws
  # alone, one per step, drawn before the delay and the history.
  for _ in range(30):
    _observe(manager, env, 0.07)
  frames = _observe(manager, env, 0.07)[:, : 25 * 10].reshape(n, 25, 10)
  std = frames.var(dim=1).mean(dim=0).sqrt()
  torch.testing.assert_close(std, torch.tensor(ACTPACK_STDS), rtol=0.1, atol=0.0)


SWITCH_FLIPS: tuple[tuple[str, str, bool, dict[str, Any], dict[str, Any]], ...] = (
  ("actpack", "projected_gravity", True, {}, {"projected_gravity": True}),
  ("actpack", "applied_torque", True, {}, {"applied_torque": True}),
  ("loadcell", "socket", True, {}, {"socket_load_cell": True}),
  (
    "loadcell",
    "ankle",
    False,
    {"socket_load_cell": True},
    {"socket_load_cell": True, "ankle_load_cell": False},
  ),
  ("contact", "heel", False, {}, {"heel_contact": False}),
  ("contact", "toe", False, {}, {"toe_contact": False}),
)
"""A switch the train CLI can flip on a term's params, the ``OslObsCfg`` fields of
the group it is flipped on, and those of the group it gives."""


@pytest.mark.parametrize("term, switch, value, base, field", SWITCH_FLIPS)
def test_switch_flipped_on_a_term_param(
  scene_model: mujoco.MjModel,
  term: str,
  switch: str,
  value: bool,
  base: dict[str, Any],
  field: dict[str, Any],
) -> None:
  """A switch flipped on a term's params, as the train CLI flips it
  (``--env.observations.osl.terms.<term>.params.<switch>``), steps through the
  observation manager at the switched dims, since the term's noise follows its own
  params. The group is the one ``OslObsCfg`` builds with that switch, and the
  validation group derived from it has the same dims."""
  torch.manual_seed(6)
  n, history = 8, 3
  group = mdp.osl_observation_group(
    "train", mdp.OslObsCfg(history_length=history, **base)
  )
  group.terms[term].params[switch] = value
  cfg = mdp.OslObsCfg(history_length=history, **field)
  env = _fake_env(scene_model, num_envs=n)
  manager = ObservationManager({"osl": group}, env)
  dims = (mdp.osl_obs_dim(cfg),)
  assert manager.group_obs_dim["osl"] == dims
  for step in range(2 * history):
    assert _observe(manager, env, 0.01 * step).shape == (n, *dims)
  stds = mdp.osl_noise_std(term, group.terms[term].params)
  noise_std = manager.get_term_cfg("osl", term).func.noise_std
  assert stds is not None and noise_std.tolist() == pytest.approx(stds)
  assert group == mdp.osl_observation_group("train", cfg)

  validation = mdp.osl_validation_group(group)
  validation_env = _fake_env(scene_model, num_envs=n)
  dims_seen = ObservationManager({"osl": validation}, validation_env).group_obs_dim
  assert dims_seen == manager.group_obs_dim


@pytest.mark.parametrize(
  "cfg",
  [
    mdp.OslObsCfg(),
    mdp.OslObsCfg(projected_gravity=True, applied_torque=True, heel_contact=False),
    mdp.OslObsCfg(**mdp.LEGACY_OSL_SENSORS),
    mdp.OslObsCfg(
      history_length=4, delay_steps=(1, 3), validation_delay_steps=2, gyro_noise=0.0
    ),
  ],
  ids=["default", "switches", "legacy", "lags"],
)
def test_validation_group_derives_from_the_training_group(cfg: mdp.OslObsCfg) -> None:
  """What the runner builds its validation env from: the training env's group,
  copied with the randomization off and every bus, contact's included, at the
  validation lag, which is the validation group the task builds from the same
  config; the per-bus lags (the middle of each bus's range) give the same group."""
  train = mdp.osl_observation_group("train", cfg)
  untouched = copy.deepcopy(train)
  validation = mdp.osl_validation_group(train, cfg.validation_delay_steps)
  assert validation == mdp.osl_observation_group("validation", cfg)
  assert train == untouched
  assert not validation.enable_corruption
  lag = cfg.validation_delay_steps
  for name, term in validation.terms.items():
    assert term.noise is None and not term.params.get("randomize", False)
    lags = (0, 0) if name == "last_action" else (lag, lag)
    assert (term.delay_min_lag, term.delay_max_lag) == lags
  default = mdp.osl_validation_group(mdp.osl_observation_group("train"))
  assert default == mdp.osl_observation_group("validation")
  lags = mdp.osl_validation_lags(train)
  assert set(lags) == {n for n in mdp.OSL_BUS_TERMS if n in train.terms}
  assert set(lags.values()) == {sum(cfg.delay_steps) // 2}
  assert mdp.osl_validation_group(train, lags) == mdp.osl_validation_group(
    train, sum(cfg.delay_steps) // 2
  )
  with pytest.raises(ValueError, match="validation_delay_steps"):
    mdp.osl_validation_group(train, -1)
  with pytest.raises(ValueError, match="No validation lag"):
    mdp.osl_validation_group(train, {"actpack": 1})


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
  cfg = osl_split_env_cfg(model, socket, reward, mode)
  base = osl_gait_library_env_cfg(model, socket, reward, mode, "unicycle")

  # The human reads stage 1's actor group, term for term; the critic copy is gone.
  assert set(cfg.observations) == {"actor", "osl"}
  actor, stage1 = cfg.observations["actor"], base.observations["actor"]
  assert list(actor.terms) == list(stage1.terms)
  assert list(actor.terms)[-2:] == ["twist", "phase"]
  for name, term in actor.terms.items():
    assert term.func is stage1.terms[name].func
    assert term.params == stage1.terms[name].params
    assert term.noise is None and term.delay_max_lag == 0 and term.history_length == 0
  assert not actor.enable_corruption

  osl = cfg.observations["osl"]
  assert isinstance(cfg, OslSplitEnvCfg)
  assert tuple(osl.terms) == mdp.OSL_OBS_TERMS
  assert osl.enable_corruption == (mode == "train")
  lags = (1, 1) if mode == "validation" else (0, 2)
  for name in mdp.OSL_BUS_TERMS:
    term = osl.terms[name]
    assert (term.delay_min_lag, term.delay_max_lag) == lags
    assert term.params["randomize"] == (mode == "train")
    assert term.noise is None  # The class term adds its own, under randomize.
    assert term.history_length == 25

  # The same team reward under the same name, now with channels.
  assert set(cfg.rewards) == set(base.rewards) == {"mimic", "twist"}
  mimic, stage1_mimic = cfg.rewards["mimic"], base.rewards["mimic"]
  assert mimic.func is mdp.split_mimic_reward and mimic.weight == 1.0
  assert mimic.params == {**stage1_mimic.params, "channels": mdp.SplitRewardCfg()}
  assert (mimic.params["clf"] is not None) == (reward == "clf")
  twist, stage1_twist = cfg.rewards["twist"], base.rewards["twist"]
  assert twist.func is stage1_twist.func
  assert (twist.weight, twist.params) == (stage1_twist.weight, stage1_twist.params)
  for name, metric in base.metrics.items():
    assert cfg.metrics[name].func is metric.func
    assert cfg.metrics[name].params == metric.params
  split = {name for name in cfg.metrics if name not in base.metrics}
  clf_only = set(mdp.SPLIT_CLF_DIAGNOSTICS)
  assert clf_only <= set(mdp.SPLIT_DIAGNOSTICS)
  diagnostics = set(mdp.SPLIT_DIAGNOSTICS) - (set() if reward == "clf" else clf_only)
  assert split == diagnostics | set(mdp.SPLIT_CONTACT_DIAGNOSTICS)
  for name in split:
    assert cfg.metrics[name].func is mimic_component
    assert cfg.metrics[name].params == {"name": name}
  # The scene adds the leg-contact sensor only (test_leg_contact_sensor).
  assert [sensor.name for sensor in cfg.scene.sensors] == [sensors.LEG_CONTACT_SENSOR]
  assert base.scene.sensors == ()

  # Everything else is stage 1's: physics randomization (actuator delays included),
  # terminations, command, simulation, actions.
  assert set(cfg.events) == set(base.events)
  strength = {"servo_gains"} if model == "robot" else {"muscle_strength"}
  physical = PHYSICAL_EVENTS | strength | {"osl_motor_strength"}
  if mode == "validation":
    assert not physical & set(cfg.events)
    assert _delays(cfg) == {"osl": (5, 5), "human": (15, 15)}
  else:
    assert physical <= set(cfg.events)
    assert ("push_robot" in cfg.events) == (mode == "train")
    assert _delays(cfg) == {"osl": (0, 10), "human": (10, 20)}
  assert "osl_encoder_bias" not in cfg.events
  assert set(cfg.terminations) == set(base.terminations)
  for name, term in base.terminations.items():
    assert cfg.terminations[name].func is term.func
    assert cfg.terminations[name].params == term.params
  assert cfg.commands == base.commands and list(cfg.actions) == list(base.actions)
  assert cfg.sim == base.sim and cfg.decimation == base.decimation
  assert cfg.episode_length_s == base.episode_length_s
  assert cfg.scene.num_envs == base.scene.num_envs
  entity, stage1_entity = cfg.scene.entities["robot"], base.scene.entities["robot"]
  assert entity.articulation == stage1_entity.articulation
  assert entity.init_state == stage1_entity.init_state
  assert entity.collisions == stage1_entity.collisions
  assert entity.spec_fn is not stage1_entity.spec_fn


def test_env_cfg_spec_fn_adds_the_sensors_over_the_foot_priority() -> None:
  cfg = osl_split_env_cfg("robot", "rigid", "mimic", "train")
  spec = cfg.scene.entities["robot"].spec_fn()
  assert spec.site(sensors.OSL_IMU_SITE) is not None
  # By name, not spec.sensor(): see add_osl_sensors.
  assert set(sensors.OSL_SENSORS) <= {sensor.name for sensor in spec.sensors}
  feet = [g for g in spec.geoms if re.fullmatch(FOOT_GEOMS, g.name)]
  assert len(feet) == 8 and {int(g.priority) for g in feet} == {1}


def test_leg_contact_sensor() -> None:
  """The split scene carries a contact sensor between the OSL's subtree and the
  biological left leg's, kept over a control step's substeps, with two metrics. In
  plain MuJoCo it reads nothing standing and a hull strike once the right hip
  adducts the prosthesis into the left leg (a sensor changes no physics)."""
  from mjlab.scene import Scene
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  cfg = osl_split_env_cfg("robot", "rigid", "clf", "train")
  (sensor,) = [s for s in cfg.scene.sensors if s.name == sensors.LEG_CONTACT_SENSOR]
  assert isinstance(sensor, ContactSensorCfg)
  assert sensor.primary == ContactMatch(
    mode="subtree", pattern=f"^{sensors.OSL_ROOT_BODY}$", entity="robot"
  )
  assert sensor.secondary == ContactMatch(
    mode="subtree", pattern=sensors.LEFT_LEG_BODY, entity="robot"
  )
  assert sensor.reduce == "netforce" and sensor.history_length == cfg.decimation
  for name in mdp.SPLIT_CONTACT_DIAGNOSTICS:
    assert cfg.metrics[name].func is mimic_component
    assert cfg.metrics[name].params == {"name": name}
  assert cfg.metrics["split_leg_contact"].reduce == "mean"  # A contact rate.
  assert cfg.metrics["split_leg_contact_force"].reduce == "max"  # The hardest hit.

  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, "cpu").compile()
  (index,) = [
    i for i in range(model.nsensor) if model.sensor(i).name.startswith(sensor.name)
  ]
  assert model.sensor_type[index] == mujoco.mjtSensor.mjSENS_CONTACT
  assert model.body(model.sensor_objid[index]).name == f"robot/{sensors.OSL_ROOT_BODY}"
  assert model.body(model.sensor_refid[index]).name == f"robot/{sensors.LEFT_LEG_BODY}"
  data = mujoco.MjData(model)
  hip = model.joint("robot/hip_adduction_r").qposadr[0]
  free = model.jnt_qposadr[0]
  readings = []
  for angle in (0.0, 0.3):
    mujoco.mj_resetData(model, data)
    data.qpos[free + 2] += 1.0  # Off the floor.
    data.qpos[hip] = angle
    mujoco.mj_forward(model, data)
    adr = model.sensor_adr[index]
    readings.append(float(np.linalg.norm(data.sensordata[adr : adr + 3])))
  assert readings[0] == 0.0 and readings[1] > 1.0


def test_env_cfg_options_and_refusals() -> None:
  for dr_cfg in (OslDrCfg(encoder_bias=True), OslDrCfg(obs_noise=True)):
    with pytest.raises(ValueError, match="human"):
      osl_split_env_cfg(dr_cfg=dr_cfg)
  cfg = osl_split_env_cfg(
    dr_cfg=OslDrCfg(foot_friction=(0.8, 1.0)),
    osl_obs=mdp.OslObsCfg(history_length=10, socket_load_cell=False),
  )
  assert cfg.events["foot_friction"].params["ranges"] == (0.8, 1.0)
  loadcell = cfg.observations["osl"].terms["loadcell"]
  assert loadcell.history_length == 10 and not loadcell.params["socket"]


def _dim(cfg: ManagerBasedRlEnvCfg) -> int:
  return mdp.osl_layout_dim(mdp.osl_obs_layout(cfg.observations["osl"]))


def test_drop_rule_and_refusals_on_the_env_cfg() -> None:
  """A sensor term whose switches are all off leaves the group however they were
  turned off (the config or its params, as the CLI does); heel contact is refused
  under the original touch sites, whose r_foot zone never fires, but the toe is
  not; unknown terms and a group without a sensor are refused."""
  cfg = osl_split_env_cfg("robot", "rigid", "mimic", "train")
  assert _dim(cfg) == 500
  off = mdp.OslObsCfg(heel_contact=False, toe_contact=False)
  no_contact = osl_split_env_cfg("robot", "rigid", "mimic", "train", osl_obs=off)
  assert list(no_contact.observations["osl"].terms) == [
    "actpack",
    "loadcell",
    "last_action",
  ]
  assert _dim(no_contact) == 450
  # Switched off on the params, then constructed (as tyro does after the CLI).
  group = copy.deepcopy(cfg.observations["osl"])
  group.terms["loadcell"].params["ankle"] = False
  edited = replace(cfg, observations={**cfg.observations, "osl": group})
  assert list(edited.observations["osl"].terms) == [
    "actpack",
    "contact",
    "last_action",
  ]
  assert "loadcell" in group.terms  # A new dict: the old group is untouched.
  assert _dim(edited) == 350

  with pytest.raises(ValueError) as error:
    replace(cfg, osl_touch_sites="original")
  assert "--env.osl-touch-sites" in str(error.value)
  assert "--env.observations.osl.terms.contact.params.heel False" in str(error.value)
  toe_only = osl_split_env_cfg(
    "robot", "rigid", "mimic", "train", osl_obs=mdp.OslObsCfg(heel_contact=False)
  )
  original = replace(toe_only, osl_touch_sites="original")
  assert mdp.osl_sensor_set(original.observations["osl"])["toe_contact"]
  replace(no_contact, osl_touch_sites="original")  # No contacts at all: fine.

  unknown = copy.deepcopy(cfg)
  unknown.observations["osl"].terms["phase"] = ObservationTermCfg(func=lambda: None)
  with pytest.raises(ValueError, match="Unknown osl observation terms"):
    sync_osl_group(unknown)
  bare = copy.deepcopy(cfg)
  bare.observations["osl"] = ObservationGroupCfg(
    terms={"last_action": cfg.observations["osl"].terms["last_action"]}
  )
  with pytest.raises(ValueError, match="at least one sensor term"):
    sync_osl_group(bare)


SPLIT_TASK = "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
FLAG = "--env.observations.osl"
FLAG_SETS: tuple[tuple[str, int, dict[str, Any]], ...] = (
  ("", 500, {}),
  (
    f"{FLAG}.terms.loadcell.params.socket True "
    f"{FLAG}.terms.contact.params.heel False {FLAG}.terms.contact.params.toe False",
    600,
    {"socket_load_cell": True, "heel_contact": False, "toe_contact": False},
  ),
  (
    f"{FLAG}.terms.contact.params.heel False {FLAG}.terms.contact.params.toe False",
    450,
    {"heel_contact": False, "toe_contact": False},
  ),
  (f"{FLAG}.terms.loadcell.params.socket True", 650, {"socket_load_cell": True}),
  (f"{FLAG}.history-length 50", 1000, {}),
  (
    f"{FLAG}.terms.loadcell.params.ankle False "
    f"{FLAG}.terms.loadcell.params.socket True "
    f"{FLAG}.terms.contact.params.heel False {FLAG}.terms.contact.params.toe False",
    450,
    {
      "ankle_load_cell": False,
      "socket_load_cell": True,
      "heel_contact": False,
      "toe_contact": False,
    },
  ),
)
"""Train flags, the OSL input size they give, and the set's switches: the
default, R2's two cells, contacts off, two cells plus contacts, a 50-frame
history, and s2_D's socket cell alone."""


@pytest.mark.parametrize("flags, dim, switches", FLAG_SETS)
def test_sensor_flags_parse_and_round_trip(
  flags: str, dim: int, switches: dict[str, Any]
) -> None:
  """The flags parse as ``train`` parses them, give the set, and the flags
  :func:`~mjlab.tasks.osl_split.mdp.osl_sensor_flags` prints for the parsed group
  are the same."""
  args = tyro.cli(
    TrainConfig,
    args=flags.split(),
    default=TrainConfig.from_task(SPLIT_TASK),
    config=mjlab.TYRO_FLAGS,
  )
  group = args.env.observations["osl"]
  assert _dim(args.env) == dim
  expected = {
    key: value
    for key, value in asdict(mdp.OslObsCfg(**switches)).items()
    if key in mdp.SWITCH_FIELDS.values()
  }
  sensor_set = mdp.osl_sensor_set(group)
  for (term, switch), field in mdp.SWITCH_FIELDS.items():
    assert sensor_set[field] == expected[field], (term, switch)
  assert mdp.osl_sensor_flags(group) == flags


def _legacy_record(**edits: Any) -> dict[str, Any]:
  """A legacy record (both cells, no contacts) with its params or histories
  edited: ``ankle=None`` drops the key, as runs before 242de42d8 recorded."""
  record = mdp.legacy_osl_record()
  params = record["terms"]["loadcell"]["params"]
  for key, value in edits.items():
    if key == "history":
      for term in record["terms"].values():
        term["history_length"] = value
    elif value is None:
      del params[key]
    else:
      params[key] = value
  return record


@pytest.mark.parametrize(
  "record, dim, flags",
  [
    (
      _legacy_record(ankle=None),
      600,
      f"{FLAG}.terms.loadcell.params.socket True "
      f"{FLAG}.terms.contact.params.heel False {FLAG}.terms.contact.params.toe False",
    ),
    (
      _legacy_record(ankle=False),
      450,
      f"{FLAG}.terms.loadcell.params.ankle False "
      f"{FLAG}.terms.loadcell.params.socket True "
      f"{FLAG}.terms.contact.params.heel False {FLAG}.terms.contact.params.toe False",
    ),
    (_legacy_record(ankle=None, history=50), 1200, None),
    (mdp.osl_group_record(mdp.osl_observation_group("train")), 500, ""),
  ],
  ids=["r2", "s2_d", "history_50", "default"],
)
def test_record_round_trips(
  record: dict[str, Any], dim: int, flags: str | None
) -> None:
  """A run's record rebuilds its group at its size: R2's (no ``ankle`` key, so both
  cells), s2_D's (the socket cell alone), a 50-frame history and the new default;
  its record is itself, and its flags rebuild it."""
  group = mdp.osl_group_from_record(record)
  layout = mdp.osl_obs_layout(group)
  assert (
    mdp.osl_layout_dim(layout) == dim == mdp.osl_layout_dim(mdp.osl_obs_layout(record))
  )
  assert mdp.osl_group_record(group) == record
  clean = mdp.osl_group_from_record(record, train=False)
  assert not clean.enable_corruption
  assert all(not t.params.get("randomize", False) for t in clean.terms.values())
  assert mdp.osl_obs_layout(clean) == layout
  if flags is not None:
    assert mdp.osl_sensor_flags(record) == flags
  else:  # Per-term histories, as s2_E's flags set them.
    printed = mdp.osl_sensor_flags(record)
    assert f"{FLAG}.history-length 50" in printed
  with pytest.raises(ValueError, match="unknown term"):
    mdp.osl_group_from_record({**record, "terms": {"phase": {"func": "x"}}})
  bad = copy.deepcopy(record)
  bad["terms"]["actpack"]["scale"] = [2.0]
  with pytest.raises(ValueError, match="outside the schema"):
    mdp.osl_group_from_record(bad)


@pytest.mark.parametrize("legacy", [False, True], ids=["default", "legacy"])
def test_env_yaml_is_a_record(tmp_path: Path, legacy: bool) -> None:
  """A run's params/env.yaml, read as data as the exporter reads it, is a record of
  its group: it rebuilds the group exactly."""
  osl_obs = mdp.OslObsCfg(**mdp.LEGACY_OSL_SENSORS) if legacy else None
  cfg = osl_split_env_cfg("robot", "rigid", "mimic", "train", osl_obs=osl_obs)
  dump_yaml(tmp_path / "env.yaml", asdict(cfg))
  saved = read_config(tmp_path / "env.yaml")["observations"]["osl"]
  rebuilt = mdp.osl_group_from_record(saved)
  assert rebuilt == cfg.observations["osl"]
  assert mdp.osl_obs_layout(saved) == mdp.osl_obs_layout(cfg.observations["osl"])


##
# The real env (slow, GPU).
##

UNICYCLE_LIBRARY = os.environ.get("MJLAB_OSL_UNICYCLE_LIBRARY", "")


@pytest.mark.slow
@pytest.mark.skipif(
  not UNICYCLE_LIBRARY, reason="set MJLAB_OSL_UNICYCLE_LIBRARY to a library dir"
)
@pytest.mark.parametrize("model, reward", [("robot", "mimic"), ("msk", "clf")])
def test_split_env_rollout(model: Model, reward: Reward) -> None:
  """The stage-2 env on the real library: both groups' dims, finite readings from
  every OSL sensor through 20 steps, and the IMU within the ActPack's ranges in the
  observation."""
  cfg = osl_split_env_cfg(model, "compliant", reward, "train")
  cfg.scene.num_envs = 8
  motion = cfg.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = UNICYCLE_LIBRARY
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  dims = env.observation_manager.group_obs_dim
  assert dims["osl"] == (mdp.osl_obs_dim(),)
  # Stage 1's compliant actor group; the robot has no 1695 muscle-state dims.
  assert dims["actor"] == ((2338,) if model == "msk" else (643,))
  env.reset()
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
  # The IMU saturates in the observation only (the sensors have no cutoff), after
  # its mount and bias; the training noise rides on the clamp.
  obs_cfg = mdp.OslObsCfg()
  frame = mdp.osl_actpack_dim(obs_cfg)
  imu_bound = torch.tensor(
    [sensors.GYRO_RANGE + 6 * obs_cfg.gyro_noise] * 3
    + [sensors.ACC_RANGE + 6 * obs_cfg.acc_noise] * 3,
    device=device,
  )
  for _ in range(20):
    obs, step_reward, _, _, _ = env.step(zero)
    assert torch.isfinite(step_reward).all()
    osl = obs["osl"]
    assert isinstance(osl, torch.Tensor) and torch.isfinite(osl).all()
    imu = osl[:, : frame * obs_cfg.history_length].reshape(8, -1, frame)[..., 4:10]
    assert bool((imu.abs() <= imu_bound).all())
  for name in sensors.OSL_SENSORS:
    data = env.scene[sensors.scene_sensor_name(name)].data
    assert data.shape[0] == 8 and torch.isfinite(data).all()
  env.close()
