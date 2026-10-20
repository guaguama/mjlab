"""OSL tracking contracts for the four v3 variants.

Ordinary tests generate their own 73-joint reference clip from the rigid robot, which
also exercises the padding every other variant needs. The real clip is checked
separately when it is available.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import SOCKET_JOINTS
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from mjlab.tasks.tracking.config.osl import OslTrackingOnPolicyRunner
from mjlab.tasks.tracking.config.osl.env_cfgs import osl_fullbody_tracking_env_cfg
from mjlab.tasks.tracking.config.osl.mdp import OslMotionCommand, bad_osl_anchor_ori
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg

# task id -> (model, socket, number of non-free joints, action dims by term)
TASKS = {
  "Mjlab-Tracking-Flat-OSL-Fullbody": (
    "robot",
    "rigid",
    73,
    {"joint_pos": 27, "osl_effort": 2},
  ),
  "Mjlab-Tracking-Flat-OSL-Fullbody-Compliant": (
    "robot",
    "compliant",
    77,
    {"joint_pos": 27, "osl_effort": 2},
  ),
  "Mjlab-Tracking-Flat-OSL-Fullbody-MSK": (
    "msk",
    "rigid",
    113,
    {"osl_effort": 2, "muscles": 401},
  ),
  "Mjlab-Tracking-Flat-OSL-Fullbody-MSK-Compliant": (
    "msk",
    "compliant",
    117,
    {"osl_effort": 2, "muscles": 401},
  ),
}
REAL_CLIP = Path(
  os.environ.get(
    "MJLAB_OSL_MOTION_FILE",
    Path.home() / "src/00_AMBER/00_projbackflip/clipper/outputs/mjlab/osl_fullbody/"
    "WalkingStraightForwards03_poses_crop0-521_z-0.0016_ankle-7_bioankle-5.npz",
  )
)


@pytest.fixture(scope="module")
def reference_model() -> mujoco.MjModel:
  return Entity(osl_fullbody_tracking_env_cfg().scene.entities["robot"]).spec.compile()


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory, reference_model: mujoco.MjModel) -> Path:
  """A standing 73-joint clip at 50 Hz, shaped like clipper's output."""
  model = reference_model
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  path = tmp_path_factory.mktemp("osl") / "motion.npz"
  frames = 60
  np.savez(
    path,
    fps=np.array([50.0]),
    joint_pos=np.tile(data.qpos[7:], (frames, 1)),
    joint_vel=np.zeros((frames, 73)),
    body_pos_w=np.tile(data.xpos[1:], (frames, 1, 1)),
    body_quat_w=np.tile(data.xquat[1:], (frames, 1, 1)),
    body_lin_vel_w=np.zeros((frames, model.nbody - 1, 3)),
    body_ang_vel_w=np.zeros((frames, model.nbody - 1, 3)),
  )
  return path


@pytest.fixture(scope="module", params=list(TASKS))
def task(request) -> str:
  return request.param


@pytest.fixture(scope="module")
def env(task: str, synthetic_clip: Path):
  cfg = load_env_cfg(task, play=True)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = str(synthetic_clip)
  cfg.scene.num_envs = 2
  result = ManagerBasedRlEnv(cfg, device=get_test_device())
  yield result
  result.close()


def test_task_contract(task: str) -> None:
  model, _, num_joints, action_dims = TASKS[task]
  cfg = load_env_cfg(task)
  play = load_env_cfg(task, play=True)
  motion = cfg.commands["motion"]
  play_motion = play.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  assert isinstance(play_motion, MotionCommandCfg)
  assert load_runner_cls(task) is OslTrackingOnPolicyRunner
  assert list(cfg.actions) == list(action_dims)
  # The motion NPZ is 50 Hz and MotionLoader steps one frame per env step.
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
  assert motion.body_names[:2] == ("Full Body", "pelvis")
  assert motion.anchor_body_name == "pelvis"
  assert "self_collisions" not in cfg.rewards
  assert play_motion.sampling_mode == "start"
  assert not play_motion.pose_range
  assert not play_motion.velocity_range
  assert "push_robot" not in play.events
  assert not play.observations["actor"].enable_corruption
  compiled = Entity(cfg.scene.entities["robot"]).spec.compile()
  for name in ("imu_lin_vel", "imu_ang_vel"):
    sensor = compiled.sensor(name)
    assert compiled.site_bodyid[sensor.objid[0]] == compiled.body("pelvis").id
  for term in ("joint_pos", "joint_vel"):
    noise = cfg.observations["actor"].terms[term].noise
    assert isinstance(noise, UniformNoiseCfg)
    assert isinstance(noise.n_max, tuple)
    assert len(noise.n_max) == num_joints
  if model == "robot":
    joint_pos = cfg.actions["joint_pos"]
    assert isinstance(joint_pos, JointPositionActionCfg)
    scale = joint_pos.scale
    assert isinstance(scale, dict)
    assert len(scale) == 27
    assert 0.08 < min(scale.values()) and max(scale.values()) < 0.32


def assert_coupled(command: OslMotionCommand) -> None:
  q = command.robot.data.joint_pos
  v = command.robot.data.joint_vel
  for dependent, source, c in command.couplings:
    if source == -1:
      torch.testing.assert_close(
        q[:, dependent], torch.full_like(q[:, dependent], c[0])
      )
      assert torch.count_nonzero(v[:, dependent]) == 0
      continue
    x = q[:, source]
    expected = sum(c[i] * x**i for i in range(5))
    derivative = sum(i * c[i] * x ** (i - 1) for i in range(1, 5))
    torch.testing.assert_close(q[:, dependent], expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(v[:, dependent], derivative * v[:, source])


def test_actions_padding_and_coupled_resets(task: str, env: ManagerBasedRlEnv) -> None:
  _, socket, num_joints, action_dims = TASKS[task]
  env.reset()
  command = env.command_manager.get_term("motion")
  assert isinstance(command, OslMotionCommand)
  for term, dim in action_dims.items():
    assert env.action_manager.get_term(term).action_dim == dim
  assert command.robot.data.joint_pos.shape == (2, num_joints)
  assert len(command.couplings) == 44
  assert len(command.independent) == num_joints - 44

  # The 73-joint clip is padded to the entity, zeros in the columns it lacks.
  names = list(command.robot.joint_names)
  assert command.motion.joint_pos.shape[1] == num_joints
  reference = Entity(osl_fullbody_tracking_env_cfg().scene.entities["robot"])
  extra = [i for i, n in enumerate(names) if n not in reference.joint_names]
  assert len(extra) == num_joints - 73
  assert torch.count_nonzero(command.motion.joint_pos[:, extra]) == 0
  assert len(command.socket_ids) == (4 if socket == "compliant" else 0)

  ids = torch.arange(2, device=env.device)
  original = command.motion.joint_pos.clone()
  command.reset_to_frame(ids, 0)
  assert_coupled(command)
  baseline = command.robot.data.joint_pos.clone()
  command.cfg.joint_position_range = (-0.1, 0.1)
  command._resample_command(ids)
  assert_coupled(command)
  # Socket joints are passive and are reset at rest at neutral, not perturbed.
  if command.socket_ids:
    socket_q = command.robot.data.joint_pos[:, command.socket_ids]
    assert torch.count_nonzero(socket_q) == 0
  delta = (
    command.robot.data.joint_pos[:, command.independent]
    - baseline[:, command.independent]
  )
  assert delta.abs().max() <= 0.10001
  assert delta.abs().max() > 0.001
  command.time_steps[:] = command.motion.time_step_total - 1
  command._update_command()
  assert_coupled(command)
  torch.testing.assert_close(command.motion.joint_pos, original)
  command.cfg.joint_position_range = (0.0, 0.0)

  generator = torch.Generator(device=env.device).manual_seed(0)
  for _ in range(20):
    action = torch.randn(
      2, env.action_manager.total_action_dim, device=env.device, generator=generator
    )
    env.step(action)
  assert torch.isfinite(command.robot.data.joint_pos).all()
  assert torch.isfinite(command.robot.data.root_link_pos_w).all()


def test_padding_rejects_unknown_width(tmp_path: Path, synthetic_clip: Path) -> None:
  with np.load(synthetic_clip) as clip:
    arrays = dict(clip)
  arrays["joint_pos"] = arrays["joint_pos"][:, :70]
  arrays["joint_vel"] = arrays["joint_vel"][:, :70]
  bad = tmp_path / "bad.npz"
  np.savez(bad, **arrays)
  cfg = load_env_cfg("Mjlab-Tracking-Flat-OSL-Fullbody-MSK", play=True)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = str(bad)
  cfg.scene.num_envs = 1
  with pytest.raises(ValueError, match="Motion has 70 joints"):
    ManagerBasedRlEnv(cfg, device=get_test_device())


def test_socket_joints_are_named() -> None:
  model = Entity(
    osl_fullbody_tracking_env_cfg("robot", "compliant").scene.entities["robot"]
  )
  assert set(SOCKET_JOINTS) <= set(model.joint_names)


def test_normalized_tilt() -> None:
  # Same z projection, opposite tilt: a z-only comparison would miss this. Not
  # used by the stock-behaviour tasks; kept for tuning.
  angle = torch.tensor([0.0, 0.8])
  reference = torch.stack(
    (torch.cos(angle / 2), torch.sin(angle / 2), torch.zeros(2), torch.zeros(2)), dim=-1
  )
  robot = reference.clone()
  robot[:, 1] *= -1
  command = SimpleNamespace(anchor_quat_w=reference, robot_anchor_quat_w=robot)
  env = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          gravity_vec_w=torch.tensor([[0.0, 0.0, -9.81]]).repeat(2, 1)
        )
      )
    },
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )
  result = bad_osl_anchor_ori(
    cast(ManagerBasedRlEnv, env), SceneEntityCfg("robot"), "motion", 0.8
  )
  assert result.tolist() == [False, True]


def test_native_checkpoint_save() -> None:
  runner = object.__new__(OslTrackingOnPolicyRunner)
  with patch.object(MjlabOnPolicyRunner, "save") as save:
    runner.save("test.pt", {"test": True})
    save.assert_called_once_with(runner, "test.pt", {"test": True})


def test_real_clip_schema_and_fk(reference_model: mujoco.MjModel) -> None:
  """clipper's NPZ carries mjlab's body ordering for the v3 model."""
  if not REAL_CLIP.exists():
    pytest.skip("Set MJLAB_OSL_MOTION_FILE for real walking-clip integration")
  model = reference_model
  with np.load(REAL_CLIP) as clip:
    assert clip["fps"].item() == 50
    frames = clip["joint_pos"].shape[0]
    assert clip["joint_pos"].shape == (frames, 73)
    assert clip["joint_vel"].shape == (frames, 73)
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
      width = 4 if key == "body_quat_w" else 3
      assert clip[key].shape == (frames, model.nbody - 1, width)
    assert all(np.isfinite(clip[key]).all() for key in clip.files)
    data = mujoco.MjData(model)
    for frame in range(frames):
      data.qpos[:3] = clip["body_pos_w"][frame, 0]
      data.qpos[3:7] = clip["body_quat_w"][frame, 0]
      data.qpos[7:] = clip["joint_pos"][frame]
      mujoco.mj_kinematics(model, data)
      np.testing.assert_allclose(data.xpos[1:], clip["body_pos_w"][frame], atol=2e-6)
      dots = np.sum(data.xquat[1:] * clip["body_quat_w"][frame], axis=-1)
      np.testing.assert_allclose(np.abs(dots), 1, atol=2e-6)
