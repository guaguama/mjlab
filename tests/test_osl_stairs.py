"""Contracts for the stairs task (``tasks/osl_stairs``).

The light tests use a synthetic terrain and a fake env. The clip tests need a
clipper stairs clip with its terrain sidecar (``MJLAB_OSL_STAIRS_MOTION``); the
rollout test also builds a warp env and is slow.
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
import tyro
from conftest import get_test_device

import mjlab
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_mimic.mdp.commands import _clip_joint_names
from mjlab.tasks.osl_stairs.config.env_cfgs import (
  OslStairsEnvCfg,
  osl_stairs_env_cfg,
  sync_terrain,
)
from mjlab.tasks.osl_stairs.mdp import StairsMotionCommand, StairsMotionCommandCfg
from mjlab.tasks.osl_stairs.rl.runner import OslStairsOnPolicyRunner
from mjlab.tasks.osl_stairs.terrain import (
  TerrainBox,
  TerrainBoxes,
  check_scene_terrain,
  load_terrain,
  resolve_terrain_file,
  scene_boxes,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls

TASKS = {
  "Mjlab-Stairs-OSL-Fullbody-ROBOT": ("robot", "rigid", "mimic"),
  "Mjlab-Stairs-OSL-Fullbody-ROBOT-Compliant": ("robot", "compliant", "mimic"),
  "Mjlab-Stairs-OSL-Fullbody-ROBOT-CLF": ("robot", "rigid", "clf"),
  "Mjlab-Stairs-OSL-Fullbody-ROBOT-CLF-Compliant": ("robot", "compliant", "clf"),
  "Mjlab-Stairs-OSL-Fullbody-MSK": ("msk", "rigid", "mimic"),
  "Mjlab-Stairs-OSL-Fullbody-MSK-Compliant": ("msk", "compliant", "mimic"),
  "Mjlab-Stairs-OSL-Fullbody-MSK-CLF": ("msk", "rigid", "clf"),
  "Mjlab-Stairs-OSL-Fullbody-MSK-CLF-Compliant": ("msk", "compliant", "clf"),
}
MOTION = Path(os.environ.get("MJLAB_OSL_STAIRS_MOTION", "/nonexistent.npz"))
needs_clip = pytest.mark.skipif(
  not MOTION.is_file(), reason="set MJLAB_OSL_STAIRS_MOTION to a stairs clip"
)

# Two steps of a TERRA stair flight (0.2 m rise), turned 0.825 rad, as in upstairs04.
BOXES = [
  {
    "pos": [-0.78, 0.05, 0.1],
    "size": [0.2, 0.5, 0.1],
    "yaw": 0.825,
    "pitch": 0.0,
    "name": "terrain_box_0",
  },
  {
    "pos": [-0.56, 0.29, 0.2],
    "size": [0.12, 0.5, 0.2],
    "yaw": 0.825,
    "pitch": 0.1,
    "name": "terrain_box_1",
  },
]


def _write_clip(folder: Path, boxes: list[dict] | None = None) -> Path:
  """An (empty) motion path and its terrain sidecar."""
  motion = folder / "clip.npz"
  motion.write_bytes(b"")
  payload = {"boxes": BOXES if boxes is None else boxes, "provenance": {"n": 1}}
  (folder / "clip_terrain.json").write_text(json.dumps(payload))
  return motion


##
# Terrain file.
##


def test_box_quat_is_terra_rotation() -> None:
  """``quat`` is TERRA's ``Rz(yaw) @ Ry(pitch)`` (clipper's matrix, verbatim)."""
  for yaw, pitch in ((0.825, 0.0), (-2.1, 0.4), (0.3, -1.2)):
    box = TerrainBox(pos=(0, 0, 0), size=(1, 1, 1), yaw=yaw, pitch=pitch)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, box.quat)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    terra = np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0.0, cp]])
    np.testing.assert_allclose(mat.reshape(3, 3), terra, atol=1e-12)


def test_malformed_terrain_raises(tmp_path: Path) -> None:
  path = tmp_path / "t.json"
  for bad in (
    {"walls": []},
    {"boxes": [{"pos": [0, 0, 0], "size": [0.1, -0.1, 0.1]}]},
    {"boxes": [{"pos": [0, 0], "size": [0.1, 0.1, 0.1]}]},
    {"boxes": [{"pos": [0, 0, 0], "size": [0.1, 0.1, 0.1], "colour": 1}]},
  ):
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="invalid terrain"):
      load_terrain(path)
  path.write_text(json.dumps({"boxes": []}))
  assert load_terrain(path) == ()


def test_terrain_file_resolution(tmp_path: Path) -> None:
  motion = _write_clip(tmp_path)
  assert resolve_terrain_file(str(motion)) == tmp_path / "clip_terrain.json"
  other = tmp_path / "other.json"
  other.write_text(json.dumps({"boxes": []}))
  assert resolve_terrain_file(str(motion), str(other)) == other
  with pytest.raises(FileNotFoundError, match="clip2_terrain.json"):
    resolve_terrain_file(str(tmp_path / "clip2.npz"))
  with pytest.raises(ValueError, match="No terrain"):
    resolve_terrain_file("", "")


def test_scene_holds_the_terrain_boxes(tmp_path: Path) -> None:
  """The boxes join the terrain body at TERRA's coordinates, with the floor's
  collision settings; the check rejects a scene built from another file."""
  cfg = osl_stairs_env_cfg("robot", "rigid", "mimic", "play")
  motion = cast(StairsMotionCommandCfg, cfg.commands["motion"])
  motion.motion_file = str(_write_clip(tmp_path))
  sync_terrain(cfg)
  model = Scene(cfg.scene, device="cpu").compile()
  boxes = load_terrain(tmp_path / "clip_terrain.json")
  check_scene_terrain(model, boxes, tmp_path / "clip_terrain.json")
  ids = scene_boxes(model)
  floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  assert len(ids) == 2
  for gid in ids:
    assert model.geom_bodyid[gid] == model.geom_bodyid[floor]
    for field in ("geom_contype", "geom_conaffinity", "geom_condim", "geom_priority"):
      assert getattr(model, field)[gid] == getattr(model, field)[floor], field
    np.testing.assert_allclose(model.geom_friction[gid], model.geom_friction[floor])
  with pytest.raises(ValueError, match="does not match"):
    check_scene_terrain(model, boxes[:1], tmp_path / "clip_terrain.json")
  moved = (TerrainBox(pos=(0, 0, 0.1), size=(0.2, 0.5, 0.1)), boxes[1])
  with pytest.raises(ValueError, match="box 0 differs"):
    check_scene_terrain(model, moved, tmp_path / "clip_terrain.json")


##
# Registration and configs.
##


def test_tasks_and_configs() -> None:
  assert sorted(t for t in list_tasks() if t.startswith("Mjlab-Stairs-")) == sorted(
    TASKS
  )
  for task in TASKS:
    assert load_runner_cls(task) is OslStairsOnPolicyRunner
    for play in (False, True):
      cfg = load_env_cfg(task, play=play)
      assert isinstance(cfg, OslStairsEnvCfg)
      motion = cfg.commands["motion"]
      assert isinstance(motion, StairsMotionCommandCfg)
      assert motion.random_start is not play
      assert cfg.scene.spec_fn == TerrainBoxes(motion.motion_file, motion.terrain_file)
      assert cfg.scene.env_spacing == 0.0
      assert ("push_robot" in cfg.events) is not play
      assert "trajectory_end" in cfg.terminations


def test_domain_randomization_per_mode() -> None:
  """The gait-library set: physical draws in training and play, pushes in training,
  delays everywhere; validation runs the nominal plant at mid-range delays."""
  for mode in ("train", "play", "validation"):
    cfg = osl_stairs_env_cfg("msk", "compliant", "clf", mode)
    physical = {"torso_inertia", "foot_friction", "joint_armature", "muscle_strength"}
    if mode == "validation":
      assert set(cfg.events) == {"reset_scene_to_default"}
    else:
      assert physical <= set(cfg.events)
    robot = cfg.scene.entities["robot"]
    assert isinstance(robot, EntityCfg) and robot.articulation is not None
    lags = {(a.delay_min_lag, a.delay_max_lag) for a in robot.articulation.actuators}
    # 2 ms steps: OSL 0-20 ms, human 20-40 ms; validation at the middle.
    assert lags == ({(5, 5), (15, 15)} if mode == "validation" else {(0, 10), (10, 20)})


def test_cli_motion_moves_the_terrain(tmp_path: Path) -> None:
  """tyro rebuilds the env config after the overrides, so the terrain follows."""
  motion = _write_clip(tmp_path)
  args = tyro.cli(
    TrainConfig,
    args=[
      "--env.commands.motion.motion-file",
      str(motion),
      "--env.commands.motion.terrain-file",
      "custom.json",
    ],
    default=TrainConfig.from_task("Mjlab-Stairs-OSL-Fullbody-ROBOT"),
    config=mjlab.TYRO_FLAGS,
  )
  assert isinstance(args.env, OslStairsEnvCfg)
  assert args.env.scene.spec_fn == TerrainBoxes(str(motion), "custom.json")


##
# Command (fake env).
##


def _fake_command(
  num_envs: int = 3, frames: int = 50
) -> tuple[StairsMotionCommand, dict[str, torch.Tensor]]:
  """A StairsMotionCommand over a straight synthetic clip, and the log of what it
  writes to the robot."""
  cmd = StairsMotionCommand.__new__(StairsMotionCommand)
  cmd.cfg = StairsMotionCommandCfg(motion_file="synthetic", random_start=True)
  cmd._env = cast(
    Any,
    SimpleNamespace(
      num_envs=num_envs,
      device="cpu",
      common_step_counter=7,
      scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
    ),
  )
  writes: dict[str, torch.Tensor] = {}
  cmd.robot = cast(
    Any,
    SimpleNamespace(
      write_joint_state_to_sim=lambda p, v, env_ids: writes.update(joint=p),
      write_root_state_to_sim=lambda s, env_ids: writes.update(root=s),
      reset=lambda env_ids: writes.update(target=torch.zeros(1)),
      set_joint_position_target=lambda p, env_ids: writes.update(target=p),
    ),
  )
  t = torch.arange(frames, dtype=torch.float32)
  cmd.num_frames = frames
  cmd.ref_root_pos = torch.stack([0.5 + 0.01 * t, -0.3 + 0.0 * t, 0.9 + 0.01 * t], -1)
  cmd.ref_root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(frames, 1)
  cmd.ref_root_lin_vel = torch.zeros(frames, 3)
  cmd.ref_root_ang_vel_w = torch.zeros(frames, 3)
  cmd.ref_joint_pos = t[:, None].repeat(1, 4)
  cmd.ref_joint_vel = torch.zeros(frames, 4)
  cmd.frame = torch.zeros(num_envs, dtype=torch.long)
  cmd.start_frame = torch.zeros(num_envs, dtype=torch.long)
  cmd.pos_offset = torch.full((num_envs, 3), 9.0)
  cmd._just_reset = torch.zeros(num_envs, dtype=torch.bool)
  cmd._jump_step = torch.full((num_envs,), -10, dtype=torch.long)
  return cmd, writes


def test_start_keeps_the_reference_world_pose() -> None:
  """No re-centring: the robot starts at the reference's own world pose, servos
  hold the start pose, and the episode clears a pending push flag."""
  cmd, writes = _fake_command()
  ids = torch.arange(3)
  cmd.mark_discontinuity(ids)
  cast(Any, cmd._env).common_step_counter += 1
  assert cmd.reference_jumped.all()
  cmd._resample_command(ids)
  start = cmd.frame
  torch.testing.assert_close(writes["root"][:, :3], cmd.ref_root_pos[start])
  torch.testing.assert_close(writes["joint"], cmd.ref_joint_pos[start])
  torch.testing.assert_close(writes["target"], cmd.ref_joint_pos[start])
  assert torch.equal(cmd.pos_offset, torch.zeros(3, 3))
  assert not cmd.reference_jumped.any()


def test_push_marks_one_reward_as_jumped() -> None:
  cmd, _ = _fake_command()
  env = cast(Any, cmd._env)
  cmd.mark_discontinuity(torch.tensor([1]))
  assert not cmd.reference_jumped.any()  # This step's reward came before the push.
  env.common_step_counter += 1
  assert cmd.reference_jumped.tolist() == [False, True, False]
  env.common_step_counter += 1
  assert not cmd.reference_jumped.any()


##
# Runner.
##


def test_evaluation_envs_play_the_training_clip(tmp_path: Path) -> None:
  """Validation and video envs are the task's own modes on the training env's clip
  and terrain (videos on the nominal plant); stress plays the training config."""
  train = osl_stairs_env_cfg("msk", "rigid", "clf", "train")
  train_motion = cast(StairsMotionCommandCfg, train.commands["motion"])
  train_motion.motion_file = str(_write_clip(tmp_path))
  train_motion.terrain_file = str(tmp_path / "clip_terrain.json")
  sync_terrain(train)
  runner = OslStairsOnPolicyRunner.__new__(OslStairsOnPolicyRunner)
  runner.env = cast(Any, SimpleNamespace(unwrapped=SimpleNamespace(cfg=train)))
  runner.validation_cfg = {"model": "msk", "socket": "rigid", "reward": "clf"}
  for purpose in ("validation", "video", "stress"):
    cfg = runner._eval_env_cfg(purpose)
    motion = cast(StairsMotionCommandCfg, cfg.commands["motion"])
    assert (motion.motion_file, motion.terrain_file) == (
      train_motion.motion_file,
      train_motion.terrain_file,
    )
    assert cfg.scene.spec_fn == train.scene.spec_fn
    assert motion.random_start is (purpose == "stress")
    if purpose != "stress":
      assert set(cfg.events) == {"reset_scene_to_default"}


def test_run_keeps_a_copy_of_its_terrain(tmp_path: Path) -> None:
  _write_clip(tmp_path)
  source = tmp_path / "clip_terrain.json"
  cmd = SimpleNamespace(terrain_path=source)
  runner = OslStairsOnPolicyRunner.__new__(OslStairsOnPolicyRunner)
  runner.env = cast(
    Any,
    SimpleNamespace(
      unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd))
    ),
  )
  runner.logger = cast(
    Any, SimpleNamespace(log_dir=str(tmp_path / "run"), disable_logs=False)
  )
  copy = runner.save_terrain()
  assert copy is not None and copy == tmp_path / "run" / "params" / "terrain.json"
  assert copy.read_text() == source.read_text()
  runner.logger.disable_logs = True  # Other ranks write nothing.
  assert runner.save_terrain() is None


##
# The real clip.
##


@needs_clip
def test_reference_climbs_onto_the_terrain() -> None:
  """The clip and its sidecar share a frame: replayed kinematically on the built
  scene, the feet land on the boxes (shallow penetration, none floating far
  above), and the clip ends standing on the top step at its starting height."""
  cfg = osl_stairs_env_cfg("robot", "rigid", "mimic", "play")
  motion = cast(StairsMotionCommandCfg, cfg.commands["motion"])
  motion.motion_file = str(MOTION)
  sync_terrain(cfg)
  model = Scene(cfg.scene, device="cpu").compile()
  data = mujoco.MjData(model)
  boxes = set(scene_boxes(model))
  clip = np.load(MOTION)
  joints = [
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")]
    for n in _clip_joint_names()
  ]
  root = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/root")
  ]
  deepest, touching = 0.0, 0
  frames = clip["joint_pos"].shape[0]
  for t in range(frames):
    data.qpos[:] = model.qpos0
    data.qpos[root : root + 3] = clip["body_pos_w"][t, 0]
    data.qpos[root + 3 : root + 7] = clip["body_quat_w"][t, 0]
    data.qpos[joints] = clip["joint_pos"][t]
    mujoco.mj_forward(model, data)
    dists = [c.dist for c in data.contact[: data.ncon] if {c.geom1, c.geom2} & boxes]
    touching += bool(dists)
    deepest = min([deepest, *dists])
  assert touching > 0.5 * frames
  assert deepest > -0.08
  terrain = load_terrain(resolve_terrain_file(str(MOTION)))
  top = max(b.pos[2] + b.size[2] for b in terrain)
  start_z, end_z = clip["body_pos_w"][0, 0, 2], clip["body_pos_w"][-1, 0, 2]
  assert abs((end_z - top) - start_z) < 0.03


@pytest.mark.slow
@needs_clip
def test_rollout_on_the_stairs() -> None:
  """A warp env: episodes start at the reference's world pose with the servos
  holding it, the feet meet the boxes, and rewards stay finite."""
  cfg = osl_stairs_env_cfg("robot", "rigid", "mimic", "train")
  motion = cast(StairsMotionCommandCfg, cfg.commands["motion"])
  motion.motion_file = str(MOTION)
  sync_terrain(cfg)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    env.reset()
    cmd = cast(StairsMotionCommand, env.command_manager.get_term("motion"))
    robot = env.scene["robot"]
    start = cmd.frame.clone()
    root = robot.data.data.qpos[:, robot.indexing.free_joint_q_adr[:3]]
    torch.testing.assert_close(root, cmd.ref_root_pos[start])
    torch.testing.assert_close(robot.data.joint_pos_target, cmd.ref_joint_pos[start])
    boxes = torch.tensor(scene_boxes(env.sim.mj_model))
    box_contacts = 0
    actions = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
    for _ in range(20):
      _, reward, *_ = env.step(actions)
      assert torch.isfinite(reward).all()
      ncon = int(env.sim.data.nacon.numpy()[0])
      geom = torch.as_tensor(env.sim.data.contact.geom.numpy()[:ncon])
      box_contacts += int(torch.isin(geom, boxes).any(dim=-1).sum())
    assert box_contacts > 0
  finally:
    env.close()
