"""Tests for the round-1 OSL gait-library task (``mjlab.tasks.osl_gait_library``).

The command and twist reward are exercised on a synthetic library through a fake env
(CPU only): loop unrolling, the phase-preserving switch with re-basing, the blend's
velocity consistency and the stride window. The real library is checked by
``test_library_env_rollout`` (slow; needs ``MJLAB_OSL_GAIT_LIBRARY``).
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_gait_library.config.env_cfgs import (
  TWIST_STD,
  TWIST_WEIGHT,
  osl_gait_library_env_cfg,
)
from mjlab.tasks.osl_gait_library.mdp import (
  GaitLibraryCommand,
  GaitLibraryCommandCfg,
  heading_twist,
  stride_twist,
)
from mjlab.tasks.osl_gait_library.mdp.commands import (
  planar_pose,
  se2_compose,
  se2_inverse,
)
from mjlab.tasks.osl_gait_library.rl.runner import GaitLibraryOnPolicyRunner
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.config.rl_cfg import MuscleMimicPpoCfg, OslMimicRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# task id -> (model, socket, reward)
TASKS: dict[str, tuple[Model, Socket, Reward]] = {
  "Mjlab-Gait-Library-OSL-Fullbody": ("robot", "rigid", "mimic"),
  "Mjlab-Gait-Library-OSL-Fullbody-Compliant": ("robot", "compliant", "mimic"),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF": ("msk", "rigid", "clf"),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF-Compliant": ("msk", "compliant", "clf"),
}
DT = 0.01

##
# Configuration.
##


@pytest.mark.parametrize("task", list(TASKS))
def test_task_contract(task: str) -> None:
  model, socket, reward = TASKS[task]
  cfg = load_env_cfg(task)
  assert load_runner_cls(task) is GaitLibraryOnPolicyRunner

  # The osl_mimic task's simulation and actions.
  mj = cfg.sim.mujoco
  assert mj.timestep == 0.002 and cfg.decimation == 5 and mj.integrator == "euler"
  assert ("eulerdamp" in mj.disableflags) == (socket == "rigid")
  assert cfg.episode_length_s == 10.0
  expected_actions = (
    ["servos", "osl_motors"] if model == "robot" else ["osl_motors", "muscles"]
  )
  assert list(cfg.actions) == expected_actions

  motion = cfg.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  assert motion.resampling_time_range == (3.0, 8.0)
  assert motion.blend_time_s == 0.4 and motion.rel_static_envs == 0.15
  assert motion.rebase_on_resample and not motion.fixed_clips
  for group in ("actor", "critic"):
    terms = list(cfg.observations[group].terms)
    assert terms[-3:] == ["goal", "twist", "phase"]
  assert "trajectory_end" not in cfg.terminations
  assert {"site_deviation", "root_deviation", "root_orientation"} <= set(
    cfg.terminations
  )
  assert (cfg.rewards["mimic"].params["clf"] is not None) == (reward == "clf")
  assert cfg.rewards["twist"].weight == TWIST_WEIGHT
  assert cfg.rewards["twist"].params["std"] == TWIST_STD

  rl = load_rl_cfg(task)
  assert isinstance(rl, OslMimicRunnerCfg)
  assert isinstance(rl.algorithm, MuscleMimicPpoCfg)
  assert rl.algorithm.bootstrap_time_outs
  assert rl.algorithm.actor_bias_init == ("qpos0" if model == "robot" else "none")
  assert rl.actor.distribution_cfg is not None
  assert rl.actor.distribution_cfg["init_std"] == (0.2 if model == "robot" else 3.0)
  v = rl.validation
  assert v.num_steps == 1000 and v.num_envs == 0
  assert (v.model, v.socket, v.reward) == (model, socket, reward)

  validation = osl_gait_library_env_cfg(model, socket, reward, mode="validation")
  v_motion = validation.commands["motion"]
  assert isinstance(v_motion, GaitLibraryCommandCfg) and v_motion.fixed_clips
  assert set(validation.terminations) == {"time_out", "site_deviation"}


##
# Planar poses and the heading-frame twist.
##


def test_se2_inverse_and_planar_pose() -> None:
  g = torch.Generator().manual_seed(0)
  a = torch.rand(16, 3, generator=g) * 4 - 2
  b = torch.rand(16, 3, generator=g) * 4 - 2
  ident = se2_compose(a, se2_inverse(a))
  torch.testing.assert_close(ident, torch.zeros_like(ident), atol=1e-5, rtol=0)
  # (a ∘ b) ∘ b^-1 = a
  back = se2_compose(se2_compose(a, b), se2_inverse(b))
  torch.testing.assert_close(back[:, :2], a[:, :2], atol=1e-5, rtol=0)
  yaw = torch.tensor([0.3, -2.0])
  quat = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], dim=-1)
  pose = planar_pose(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]), quat)
  torch.testing.assert_close(pose[:, 2], yaw)


def test_heading_twist_uses_body_minus_y_forward() -> None:
  yaw = torch.tensor([0.0, 0.7, -2.5])
  quat = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], dim=-1)
  forward = torch.stack([torch.sin(yaw), -torch.cos(yaw), 0 * yaw], dim=-1)
  left = torch.stack([torch.cos(yaw), torch.sin(yaw), 0 * yaw], dim=-1)
  vel = 1.2 * forward + 0.3 * left
  ang = torch.tensor([[0.0, 0.0, 0.4]]).expand(3, 3)
  twist = heading_twist(quat, vel, ang)
  torch.testing.assert_close(
    twist, torch.tensor([[1.2, 0.3, 0.4]]).expand(3, 3), atol=1e-6, rtol=0
  )


##
# The command on a synthetic library.
##


def _synthetic_clip(period: int, speed: float, phase0: float = 0.0) -> dict:
  """A walker moving at ``speed`` along +x whose two joints swing once per stride."""
  t = torch.arange(period, dtype=torch.float32)
  w = 2 * math.pi / (period * DT)
  ang = w * t * DT + phase0
  root_pos = torch.stack([speed * t * DT, 0.01 * torch.sin(ang), 0.9 + 0 * t], -1)
  yaw = 0.05 * torch.sin(ang)
  root_quat = torch.stack([torch.cos(yaw / 2), 0 * t, 0 * t, torch.sin(yaw / 2)], -1)
  root_lin = torch.stack([speed + 0 * t, 0.01 * w * torch.cos(ang), 0 * t], -1)
  root_ang = torch.stack([0 * t, 0 * t, 0.05 * w * torch.cos(ang)], -1)
  joint_pos = torch.stack([torch.sin(ang), 0.5 * torch.cos(ang)], -1)
  joint_vel = torch.stack([w * torch.cos(ang), -0.5 * w * torch.sin(ang)], -1)
  rpos = torch.stack([0.2 * torch.sin(ang), 0 * t, -0.4 + 0 * t], -1)[:, None, :]
  v_rel = torch.stack([0.2 * w * torch.cos(ang), 0 * t, 0 * t], -1)[:, None, :]
  zeros3 = torch.zeros(period, 1, 3)
  site_pos = torch.stack([root_pos, root_pos + rpos[:, 0]], dim=1)
  return {
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "root_pos": root_pos,
    "root_quat": root_quat,
    "root_lin_vel": root_lin,
    "root_ang_vel_w": root_ang,
    "root_ang_vel_b": root_ang,
    "site_pos": site_pos,
    "rpos": rpos,
    "rangles": zeros3,
    "rvel": torch.zeros(period, 1, 6),
    "v_rel": v_rel,
    "omega_rel": zeros3,
  }


def _fake_command(num_envs: int = 4) -> GaitLibraryCommand:
  """A GaitLibraryCommand over two walking clips (100 and 80 frames) and an idle."""
  cmd = GaitLibraryCommand.__new__(GaitLibraryCommand)
  robot_data = SimpleNamespace(
    root_link_pos_w=torch.zeros(num_envs, 3),
    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
  )
  fake_env: Any = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    common_step_counter=0,
    step_dt=DT,
    scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
  )
  fake_robot: Any = SimpleNamespace(data=robot_data)
  cmd._env = fake_env
  cmd.cfg = GaitLibraryCommandCfg(library_dir="synthetic", rel_static_envs=0.0)
  cmd.robot = fake_robot
  cmd._step_dt = DT
  clips = [_synthetic_clip(100, 1.0), _synthetic_clip(80, 1.3, 0.2)]
  idle = {k: v[:1].expand(50, *v.shape[1:]).clone() for k, v in clips[0].items()}
  for k in ("joint_vel", "root_lin_vel", "root_ang_vel_w", "root_ang_vel_b", "v_rel"):
    idle[k] = torch.zeros_like(idle[k])
  clips.append(idle)
  lengths = [100, 80, 50]
  t_max = max(lengths)
  cmd.data = {}
  for key in clips[0]:
    cmd.data[key] = torch.stack(
      [
        torch.cat(
          [c[key], c[key][-1:].expand(t_max - c[key].shape[0], *c[key].shape[1:])]
        )
        for c in clips
      ]
    )
  cmd.data["root_planar"] = planar_pose(cmd.data["root_pos"], cmd.data["root_quat"])
  cmd.num_clips = 3
  cmd.num_frames = torch.tensor(lengths)
  cmd.twists = torch.tensor([[1.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
  cmd.strides = torch.tensor([[1.0, 0.0, 0.0], [1.3 * 0.8, 0.0, 0.0], [0.0, 0.0, 0.0]])
  cmd.idle_clip = 2
  cmd.vx_fwd_range = (1.0, 1.3)
  cmd._metric = torch.ones(3)
  cmd._max_wraps = 4
  n = num_envs
  cmd.clip = torch.zeros(n, dtype=torch.long)
  cmd.frame = torch.zeros(n, dtype=torch.long)
  cmd.twist_command = torch.zeros(n, 3)
  cmd.anchor = torch.zeros(n, 3)
  cmd.z_offset = torch.zeros(n)
  cmd.clip_prev = torch.zeros(n, dtype=torch.long)
  cmd.anchor_prev = torch.zeros(n, 3)
  cmd.blend_count = torch.full((n,), 40, dtype=torch.long)
  cmd.frame_prev = torch.zeros(n)
  cmd.rate_prev = torch.ones(n)
  cmd.catchup = torch.zeros(n, 2)
  cmd.blend_steps = 40
  cmd._alpha_rate = 1.0 / (40 * DT)
  cmd._just_reset = torch.zeros(n, dtype=torch.bool)
  cmd._jump_step = torch.full((n,), -10, dtype=torch.long)
  cmd._episode_reset = False
  cmd._cache = {}
  return cmd


def test_loops_unroll_continuously() -> None:
  cmd = _fake_command()
  cmd.frame[:] = torch.tensor([98, 99, 50, 0])
  cmd.anchor[:] = torch.tensor([0.5, -0.2, 0.3])
  xs = []
  for _ in range(4):
    xs.append(cmd.reference(0).root_pos_w.clone())
    cmd._update_command(None)
  steps = torch.stack(xs).diff(dim=0)  # (3, B, 3)
  # Every step advances the served root by speed * dt along the anchor's heading,
  # across the loop as well (env 1 wraps on the first update).
  heading = torch.tensor([math.cos(0.3), math.sin(0.3)])
  along = (steps[..., :2] * heading).sum(-1)
  torch.testing.assert_close(along, torch.full_like(along, 1.0 * DT), atol=2e-4, rtol=0)
  assert int(cmd.frame[1]) == 3


def test_lookahead_crosses_the_loop() -> None:
  cmd = _fake_command()
  cmd.frame[:] = torch.tensor([95, 0, 60, 99])
  ahead = cmd.lookahead()
  expected = torch.tensor([20, 40, 60, 80]) * DT * 1.0
  torch.testing.assert_close(
    ahead.root_pos_delta[..., 0], expected.expand(4, 4), atol=1e-4, rtol=0
  )


def test_switch_keeps_phase_and_rebases_on_the_robot() -> None:
  cmd = _fake_command()
  cmd.frame[:] = torch.tensor([50, 25, 75, 99])
  robot = torch.tensor(
    [[3.0, -1.0, 0.0], [0.5, 0.5, 0.0], [-2.0, 4.0, 0.0], [1.0, 1.0, 0.0]]
  )
  yaw = torch.tensor([0.4, -1.0, 2.0, 0.0])
  data = cast(Any, cmd.robot.data)
  data.root_link_pos_w = robot
  data.root_link_quat_w = torch.stack(
    [torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1
  )
  cmd._sample_twist = lambda n: torch.tensor([[1.3, 0.0, 0.0]]).repeat(n, 1)  # type: ignore[method-assign]
  fake_env = cast(Any, cmd._env)
  fake_env.common_step_counter = 7
  cmd._switch(torch.arange(4))
  # Phase kept: frame / T is unchanged (80-frame clip), to the nearest frame.
  assert cmd.clip.tolist() == [1, 1, 1, 1]
  assert cmd.frame.tolist() == [40, 20, 60, 79]
  # Both the outgoing and the incoming clip are served at the robot's planar pose.
  ref = cmd.reference(0)
  pose = planar_pose(ref.root_pos_w, ref.root_quat)
  torch.testing.assert_close(pose[:, :2], robot[:, :2], atol=1e-5, rtol=0)
  torch.testing.assert_close(pose[:, 2], yaw, atol=1e-5, rtol=0)
  new_clip = cmd._gather(cmd.clip, cmd.frame, cmd.anchor)
  new_pose = planar_pose(new_clip["root_pos"], new_clip["root_quat"])
  torch.testing.assert_close(new_pose, pose, atol=1e-5, rtol=0)
  assert bool((cmd.alpha == 0).all())
  fake_env.common_step_counter = 8
  assert bool(cmd.reference_jumped.all())


def test_blended_velocities_are_derivatives_of_blended_positions() -> None:
  cmd = _fake_command()
  cmd.frame[:] = torch.tensor([10, 45, 70, 99])
  cmd._sample_twist = lambda n: torch.tensor([[1.3, 0.0, 0.0]]).repeat(n, 1)  # type: ignore[method-assign]
  cmd._switch(torch.arange(4))
  for _ in range(5):
    cmd._update_command(None)
  assert bool(((cmd.alpha > 0) & (cmd.alpha < 1)).all())
  before, mid, after = (cmd.reference(k) for k in (0, 1, 2))
  for pos, vel in (
    ("joint_pos", "joint_vel"),
    ("root_pos_w", "root_lin_vel_w"),
    ("rpos", "v_rel"),
  ):
    fd = (getattr(after, pos) - getattr(before, pos)) / (2 * DT)
    torch.testing.assert_close(getattr(mid, vel), fd, atol=2e-2, rtol=2e-2)


def test_blend_accelerates_without_overshoot() -> None:
  """Slow to fast: the served root's speed moves between the two clips' speeds and
  stays continuous through the end of the blend, when the catch-up is folded."""
  cmd = _fake_command(num_envs=2)
  cmd.frame[:] = torch.tensor([30, 77])
  cmd._sample_twist = lambda n: torch.tensor([[1.3, 0.0, 0.0]]).repeat(n, 1)  # type: ignore[method-assign]
  cmd._switch(torch.arange(2))
  positions = [cmd.reference(0).root_pos_w[:, :2].clone()]
  for _ in range(cmd.blend_steps + 10):
    nxt = cmd.reference(1).root_pos_w[:, :2].clone()
    cmd._update_command(None)
    now = cmd.reference(0).root_pos_w[:, :2].clone()
    torch.testing.assert_close(now, nxt, atol=1e-5, rtol=0)  # offset 1 == next step
    positions.append(now)
  assert bool((cmd.alpha == 1.0).all()) and bool((cmd.catchup == 0).all())
  speed = torch.stack(positions).diff(dim=0)[..., 0] / DT
  # The outgoing clip runs time-warped (1.0 m/s * 100/80); the new one at 1.3 m/s.
  assert float(speed.max()) < 1.3 + 0.05 and float(speed.min()) > 1.0 - 0.05


def test_blend_into_idle_slows_to_a_stand() -> None:
  """Walk to idle: the outgoing walk keeps its own speed (idle has no period to warp
  onto), so the served root decelerates from walking to standing, never faster."""
  cmd = _fake_command(num_envs=2)
  cmd.clip[:] = 1  # 1.3 m/s, 80 frames
  cmd.frame[:] = torch.tensor([10, 79])
  cmd.frame_prev[:] = cmd.frame.float()
  cmd._sample_twist = lambda n: torch.zeros(n, 3)  # type: ignore[method-assign]
  cmd._switch(torch.arange(2))
  assert cmd.clip.tolist() == [2, 2]
  torch.testing.assert_close(cmd.rate_prev, torch.ones(2))
  positions = [cmd.reference(0).root_pos_w[:, :2].clone()]
  for _ in range(cmd.blend_steps + 5):
    nxt = cmd.reference(1).root_pos_w[:, :2].clone()
    cmd._update_command(None)
    now = cmd.reference(0).root_pos_w[:, :2].clone()
    torch.testing.assert_close(now, nxt, atol=1e-5, rtol=0)
    positions.append(now)
  speed = torch.stack(positions).diff(dim=0)[..., 0] / DT
  assert float(speed.max()) < 1.3 + 0.05
  assert float(speed[-5:].abs().max()) < 1e-4  # standing after the blend


def test_idle_snaps_to_zero_twist() -> None:
  cmd = _fake_command()
  clip = cmd._snap(torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [1.2, 0.0, 0.0]]))
  assert clip.tolist() == [2, 2, 1]


##
# The stride-averaged twist reward.
##


def _fake_reward(cmd: GaitLibraryCommand, robot_vx: float) -> stride_twist:
  term = stride_twist.__new__(stride_twist)
  n = cmd.num_envs
  term._env = cmd._env
  term._len = 200
  term._robot = torch.zeros(n, 200, 3)
  term._ref = torch.zeros(n, 200, 3)
  term._count = torch.zeros(n, dtype=torch.long)
  term._slots = torch.arange(200)
  term._rows = torch.arange(n)
  term.components = {}
  data = cast(Any, cmd.robot.data)
  data.root_link_lin_vel_w = torch.zeros(n, 3)
  data.root_link_ang_vel_w = torch.zeros(n, 3)
  data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
  # Body -y forward: with identity orientation, world -y is forward.
  data.root_link_lin_vel_w[:, 1] = -robot_vx
  return term


@pytest.mark.parametrize("offset, expected", [(0.0, 1.0), (0.05, math.exp(-0.25))])
def test_stride_twist_scores_the_stride_mean(offset: float, expected: float) -> None:
  cmd = _fake_command(num_envs=2)
  # Serve a reference whose instantaneous forward speed swings around its 1.0 mean.
  cmd.twist_command[:] = torch.tensor([1.0, 0.0, 0.0])
  term = _fake_reward(cmd, robot_vx=1.0 + offset)
  ref_vx = 1.0 + 0.3 * torch.sin(2 * math.pi * torch.arange(300) / 100)
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: cmd))

  def fake_reference(offset_: int):
    vx = float(ref_vx[step[0]])
    return SimpleNamespace(
      root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
      root_lin_vel_w=torch.tensor([[0.0, -vx, 0.0]]).repeat(2, 1),
      root_ang_vel_b=torch.zeros(2, 3),
    )

  cmd.reference = fake_reference  # type: ignore[method-assign]
  step = [0]
  reward = torch.zeros(2)
  for i in range(300):
    step[0] = i
    reward = term(env, "motion", std=TWIST_STD)  # type: ignore[arg-type]
  torch.testing.assert_close(reward, torch.full((2,), expected), atol=1e-4, rtol=0)


##
# The real library (slow).
##

LIBRARY = os.environ.get("MJLAB_OSL_GAIT_LIBRARY", "")


@pytest.mark.slow
@pytest.mark.skipif(not LIBRARY, reason="set MJLAB_OSL_GAIT_LIBRARY to a library dir")
@pytest.mark.parametrize("model, reward", [("robot", "mimic"), ("msk", "clf")])
def test_library_env_rollout(model: Model, reward: Reward) -> None:
  """A few hundred steps with timer switches: finite rewards, a continuous reference."""
  from mjlab.envs import ManagerBasedRlEnv

  cfg = osl_gait_library_env_cfg(model, "rigid", reward, mode="train")
  cfg.scene.num_envs = 8
  motion = cfg.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = LIBRARY
  motion.resampling_time_range = (0.3, 0.6)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  cmd = cast(GaitLibraryCommand, env.command_manager.get_term("motion"))
  env.reset()
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
  prev = cmd.reference(0).root_pos_w.clone()
  worst = 0.0
  for _ in range(200):
    _, step_reward, terminated, truncated, _ = env.step(zero)
    jumped = cmd._jump_step == env.common_step_counter
    assert torch.isfinite(step_reward).all()
    now = cmd.reference(0).root_pos_w
    moved = (now - prev)[:, :2].norm(dim=-1)
    fresh = terminated | truncated | jumped | (env.episode_length_buf <= 1)
    worst = max(worst, float(moved[~fresh].max()) if bool((~fresh).any()) else 0.0)
    prev = now.clone()
  # Between switches and resets the root target moves no faster than the library's
  # own fastest in-stride root speed (a blend never overshoots it).
  peak = 0.0
  for c in range(cmd.num_clips):
    p = cmd.data["root_pos"][c, : int(cmd.num_frames[c]), :2]
    peak = max(peak, float((p[1:] - p[:-1]).norm(dim=-1).max()))
  assert worst < 1.05 * peak + 1e-4
  env.close()
