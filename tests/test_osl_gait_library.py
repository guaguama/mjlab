"""Tests for the OSL gait-library tasks (``mjlab.tasks.osl_gait_library``).

The command and twist reward are exercised on a synthetic library through a fake env
(CPU only): loop unrolling, the phase-preserving switch with re-basing, the blend's
velocity consistency, the stride window, library and continuous sampling, the
coverage check and the twist cap. The task configs are checked for both variants,
including the unicycle variant's domain randomization. The real libraries are
checked by the slow rollouts (``MJLAB_OSL_GAIT_LIBRARY`` for the forward library,
``MJLAB_OSL_UNICYCLE_LIBRARY`` for the unicycle one).
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  OSL_TORQUE_JOINTS,
  Model,
  Socket,
  get_spec,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scripts.play import strip_disturbances
from mjlab.tasks.osl_gait_library import mdp
from mjlab.tasks.osl_gait_library.config.dr import FOOT_GEOMS, SLIDE_SCALE, OslDrCfg
from mjlab.tasks.osl_gait_library.config.env_cfgs import (
  TWIST_STD,
  TWIST_WEIGHT,
  UNICYCLE_BCK_ARC_REGIONS,
  UNICYCLE_FWD_ARC_REGIONS,
  UNICYCLE_TWIST_METRIC_WEIGHTS,
  UNICYCLE_WZ_RANGE,
  osl_gait_library_env_cfg,
)
from mjlab.tasks.osl_gait_library.config.rl_cfg import GaitLibraryRunnerCfg
from mjlab.tasks.osl_gait_library.mdp import (
  GaitLibraryCommand,
  GaitLibraryCommandCfg,
  heading_twist,
  stride_twist,
  twist_limit_curriculum,
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
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

# task id -> (model, socket, reward)
TASKS: dict[str, tuple[Model, Socket, Reward]] = {
  "Mjlab-Gait-Library-OSL-Fullbody-ROBOT": ("robot", "rigid", "mimic"),
  "Mjlab-Gait-Library-OSL-Fullbody-ROBOT-Compliant": ("robot", "compliant", "mimic"),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF": ("msk", "rigid", "clf"),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF-Compliant": ("msk", "compliant", "clf"),
}
_MODELS: tuple[tuple[Model, str], ...] = (("robot", "-ROBOT"), ("msk", "-MSK"))
_REWARDS: tuple[tuple[Reward, str], ...] = (("mimic", ""), ("clf", "-CLF"))
_SOCKETS: tuple[tuple[Socket, str], ...] = (("rigid", ""), ("compliant", "-Compliant"))
UNICYCLE_TASKS: dict[str, tuple[Model, Socket, Reward]] = {
  f"Mjlab-Gait-Library-Unicycle-OSL-Fullbody{m_tag}{r_tag}{s_tag}": (m, s, r)
  for m, m_tag in _MODELS
  for r, r_tag in _REWARDS
  for s, s_tag in _SOCKETS
}
PHYSICAL_EVENTS = {
  "torso_inertia",
  "foot_friction",
  "joint_armature",
  "osl_motor_strength",
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
  assert isinstance(rl, GaitLibraryRunnerCfg) and rl.validation.variant == "walk_fwd"

  validation = osl_gait_library_env_cfg(model, socket, reward, mode="validation")
  v_motion = validation.commands["motion"]
  assert isinstance(v_motion, GaitLibraryCommandCfg) and v_motion.fixed_clips
  assert set(validation.terminations) == {"time_out", "site_deviation"}

  # Round 1 is unchanged: continuous sampling over the forward span, no DR.
  assert motion.sampling == "continuous" and motion.rel_turn_envs == 0.0
  assert not PHYSICAL_EVENTS & set(cfg.events) and "push_robot" not in cfg.events
  assert set(_delays(cfg).values()) == {(0, 0)}


def _delays(cfg: ManagerBasedRlEnvCfg) -> dict[str, tuple[int, int]]:
  """(min, max) actuator lags in physics steps, OSL motors vs human actuators."""
  entity = cfg.scene.entities["robot"]
  assert entity.articulation is not None
  lags = {}
  for act in entity.articulation.actuators:
    key = "osl" if tuple(act.target_names_expr) == OSL_TORQUE_JOINTS else "human"
    lags[key] = (act.delay_min_lag, act.delay_max_lag)
    if act.delay_max_lag > 0:
      assert act.delay_hold_prob == 1.0  # one lag per episode
  return lags


@pytest.mark.parametrize("task", list(UNICYCLE_TASKS))
def test_unicycle_task_contract(task: str) -> None:
  model, socket, reward = UNICYCLE_TASKS[task]
  cfg = load_env_cfg(task)
  assert load_runner_cls(task) is GaitLibraryOnPolicyRunner
  assert (cfg.rewards["mimic"].params["clf"] is not None) == (reward == "clf")
  assert list(cfg.actions) == (
    ["servos", "osl_motors"] if model == "robot" else ["osl_motors", "muscles"]
  )
  motion = cfg.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  assert motion.sampling == "library" and motion.max_twist is None
  fractions = (motion.rel_static_envs, motion.rel_turn_envs, motion.rel_back_envs)
  assert fractions == (0.15, 0.3, 0.4)
  assert motion.resampling_time_range == (3.0, 8.0) and motion.blend_time_s == 0.4

  # Training: the physical draws (each actuator kind's strength), pushes, and the
  # actuator delays (OSL 0-20 ms, human 20-40 ms at 2 ms steps). Perception DR off.
  strength = {"servo_gains"} if model == "robot" else {"muscle_strength"}
  physical = PHYSICAL_EVENTS | strength
  assert physical | {"push_robot"} <= set(cfg.events)
  assert not {"servo_gains", "muscle_strength"} - strength & set(cfg.events)
  assert "osl_encoder_bias" not in set(cfg.events)
  assert not cfg.observations["actor"].enable_corruption
  assert _delays(cfg) == {"osl": (0, 10), "human": (10, 20)}

  # Play keeps the draws and delays without pushes; validation runs the nominal
  # model with each delay fixed at the middle of its range.
  play = osl_gait_library_env_cfg(model, socket, reward, "play", "unicycle")
  assert physical <= set(play.events) and "push_robot" not in play.events
  assert _delays(play) == _delays(cfg)
  validation = osl_gait_library_env_cfg(model, socket, reward, "validation", "unicycle")
  assert not (physical | {"push_robot"}) & set(validation.events)
  assert _delays(validation) == {"osl": (5, 5), "human": (15, 15)}

  rl = load_rl_cfg(task)
  assert isinstance(rl, GaitLibraryRunnerCfg)
  v = rl.validation
  assert (v.model, v.socket, v.reward, v.variant) == (model, socket, reward, "unicycle")
  assert rl.experiment_name.startswith("osl_fullbody_unicycle")
  assert "unicycle" in rl.wandb_tags


def test_perception_switches() -> None:
  """Encoder bias and observation noise, off by default, reach only the actor."""
  dr_cfg = OslDrCfg(encoder_bias=True, obs_noise=True)
  cfg = osl_gait_library_env_cfg("robot", "rigid", "mimic", "train", "unicycle", dr_cfg)
  bias = cfg.events["osl_encoder_bias"].params
  assert bias["asset_cfg"].joint_names == OSL_TORQUE_JOINTS  # sensing only
  assert bias["bias_range"] == (-0.015, 0.015)
  actor, critic = cfg.observations["actor"], cfg.observations["critic"]
  assert actor.terms["joint_pos"].func is mdp.joint_pos_biased
  assert critic.terms["joint_pos"].func is not mdp.joint_pos_biased
  assert actor.enable_corruption and not critic.enable_corruption
  noise = actor.terms["joint_pos"].noise
  assert noise is not None
  magnitudes = set(cast(Any, noise).n_max)
  assert magnitudes == {0.01, 0.01 * SLIDE_SCALE}  # hinges vs slide joints
  play = osl_gait_library_env_cfg("robot", "rigid", "mimic", "play", "unicycle", dr_cfg)
  assert not play.observations["actor"].enable_corruption


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
  # The clips are 0.3 m/s apart, so the continuous sampler's snap error reaches 0.15.
  cmd.cfg = GaitLibraryCommandCfg(
    library_dir="synthetic", rel_static_envs=0.0, max_snap_error=0.2
  )
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
  cmd.clip_names = ["walk_a", "walk_b", "idle"]
  cmd.num_frames = torch.tensor(lengths)
  cmd.twists = torch.tensor([[1.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
  cmd.strides = torch.tensor([[1.0, 0.0, 0.0], [1.3 * 0.8, 0.0, 0.0], [0.0, 0.0, 0.0]])
  cmd._index_library()
  assert cmd.idle_clip == 2 and cmd.vx_fwd_range == pytest.approx((1.0, 1.3))
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
# Sampling: library draws, continuous boxes, the coverage check and the twist cap.
##

# idle, forward (two gentle, one tight), backward (two), pivots (two).
UNICYCLE_TWISTS = [
  [0.0, 0.0, 0.0],
  [1.0, 0.0, 0.0],
  [1.0, 0.0, 0.5],
  [0.8, 0.0, 1.0],
  [-0.6, 0.0, 0.2],
  [-0.6, 0.0, -0.2],
  [0.0, 0.0, 0.8],
  [0.0, 0.0, -1.5],
]
UNICYCLE_SPLIT = {"rel_static_envs": 0.15, "rel_turn_envs": 0.3, "rel_back_envs": 0.4}


def _indexed_command(twists: list[list[float]], **cfg: Any) -> GaitLibraryCommand:
  """A command over clips that exist only as twist labels (enough to sample)."""
  cmd = GaitLibraryCommand.__new__(GaitLibraryCommand)
  cmd._env = cast(Any, SimpleNamespace(num_envs=1, device="cpu", common_step_counter=0))
  cmd.cfg = GaitLibraryCommandCfg(library_dir="synthetic", **cfg)
  cmd.twists = torch.tensor(twists)
  cmd.num_clips = len(twists)
  cmd.num_frames = torch.full((len(twists),), 100)
  cmd.clip_names = [f"clip{i}" for i in range(len(twists))]
  cmd._index_library()
  return cmd


def _unicycle_grid() -> list[list[float]]:
  """The unicycle_v001 grid: idle, both forward bands, backward arcs, pivots."""

  def steps(lo: float, hi: float, step: float) -> list[float]:
    return [round(lo + i * step, 3) for i in range(round((hi - lo) / step) + 1)]

  grid = [[0.0, 0.0, 0.0]]
  grid += [[x, 0.0, z] for x in steps(0.5, 1.5, 0.05) for z in steps(-0.5, 0.5, 0.05)]
  grid += [
    [x, 0.0, s * z]
    for x in steps(0.6, 1.2, 0.1)
    for z in steps(0.6, 1.5, 0.1)
    for s in (1, -1)
  ]
  grid += [[x, 0.0, z] for x in steps(-0.8, -0.5, 0.05) for z in steps(-0.5, 0.5, 0.05)]
  grid += [[0.0, 0.0, s * z] for z in steps(0.5, 1.7, 0.05) for s in (1, -1)]
  return grid


def test_library_sampling_draws_labels_in_the_mode_split() -> None:
  cmd = _indexed_command(UNICYCLE_TWISTS, sampling="library", **UNICYCLE_SPLIT)
  torch.manual_seed(0)
  n = 40000
  twist, clip = cmd._draw(n)
  torch.testing.assert_close(twist, cmd.twists[clip])  # commanded at the label
  share = {
    g: c / n for g, c in Counter(cmd.clip_groups[c] for c in clip.tolist()).items()
  }
  # sesteban's split: idle 0.15, pivots 0.30, arcs 0.55 of which 40% backward.
  assert share["idle"] == pytest.approx(0.15, abs=0.01)
  assert share["turn"] == pytest.approx(0.30, abs=0.01)
  assert share["bck"] == pytest.approx(0.22, abs=0.01)
  assert share["fwd"] + share["fwd_tight"] == pytest.approx(0.33, abs=0.01)
  # Uniform within a mode: three forward clips, a third each.
  assert share["fwd_tight"] == pytest.approx(0.11, abs=0.01)


def test_library_must_hold_every_drawn_mode() -> None:
  no_pivots = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.6, 0.0, 0.0]]
  with pytest.raises(ValueError, match="pivot"):
    _indexed_command(no_pivots, sampling="library", **UNICYCLE_SPLIT)
  no_backward = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
  with pytest.raises(ValueError, match="backward arc"):
    _indexed_command(no_backward, sampling="library", **UNICYCLE_SPLIT)
  # Continuous pivots over 0.5-1.7 rad/s are far from the two pivot clips.
  with pytest.raises(ValueError, match="does not cover"):
    _indexed_command(
      UNICYCLE_TWISTS, sampling="continuous", wz_range=(0.5, 1.7), **UNICYCLE_SPLIT
    )


def test_unicycle_regions_cover_the_library_grid() -> None:
  """The unicycle config's continuous sampler fits the unicycle_v001 grid (the load
  check passes), and draws its forward arcs from the two bands by their shares."""
  cmd = _indexed_command(
    _unicycle_grid(),
    sampling="continuous",
    fwd_arc_regions=UNICYCLE_FWD_ARC_REGIONS,
    bck_arc_regions=UNICYCLE_BCK_ARC_REGIONS,
    wz_range=UNICYCLE_WZ_RANGE,
    twist_metric_weights=UNICYCLE_TWIST_METRIC_WEIGHTS,
    **UNICYCLE_SPLIT,
  )
  torch.manual_seed(0)
  vx, wz, available = cmd._sample_boxes("fwd", 20000)
  assert available
  gentle = (vx >= 0.5) & (vx <= 1.5) & (wz <= 0.5)
  tight = (vx >= 0.6) & (vx <= 1.2) & (wz >= 0.5) & (wz <= 1.5)
  assert bool((gentle | tight).all())
  assert float((wz > 0.5).float().mean()) == pytest.approx(0.25, abs=0.015)
  twist, clip = cmd._draw(2000)
  error = (twist - cmd.twists[clip]).abs()
  idle = clip == cmd.idle_clip
  assert float(error[~idle].max()) <= cmd.cfg.max_snap_error


def test_twist_limit_caps_draws() -> None:
  cmd = _indexed_command(UNICYCLE_TWISTS, sampling="library", **UNICYCLE_SPLIT)
  cmd.set_twist_limit((1.0, 0.6, 0.6))
  torch.manual_seed(0)
  twist, clip = cmd._draw(5000)
  assert bool((twist[:, 0] <= 1.0).all() and (twist[:, 0] >= -0.6).all())
  assert bool((twist[:, 2].abs() <= 0.6).all())
  # No pivot is under the cap (|wz| >= 0.8), so pivot draws idle: 0.15 + 0.30.
  assert cmd.idle_clip is not None
  idle = float(clip.eq(cmd.idle_clip).float().mean())
  assert idle == pytest.approx(0.45, abs=0.03)
  cmd.set_twist_limit(None)
  _, clip = cmd._draw(5000)
  assert set(clip.tolist()) == set(range(cmd.num_clips))
  # Continuous draws are clamped to the cap and snap among the clips under it.
  fake = _fake_command()
  fake.set_twist_limit((1.1, 0.0, 0.0))
  twist, clip = fake._draw(500)
  assert bool((twist[:, 0] <= 1.1).all()) and set(clip.tolist()) == {0}


def test_continuous_cap_truncates_the_sampling_boxes() -> None:
  """Under a cap, continuous draws come from the boxes cut down to it: an emptied
  mode idles instead of snapping into another, and nothing piles up at the edge."""
  cmd = _indexed_command(
    _unicycle_grid(),
    sampling="continuous",
    fwd_arc_regions=UNICYCLE_FWD_ARC_REGIONS,
    bck_arc_regions=UNICYCLE_BCK_ARC_REGIONS,
    wz_range=UNICYCLE_WZ_RANGE,
    twist_metric_weights=UNICYCLE_TWIST_METRIC_WEIGHTS,
    **UNICYCLE_SPLIT,
  )
  torch.manual_seed(0)
  n = 40000
  cmd.set_twist_limit((1.5, 0.0, 1.7))  # no backward walking
  twist, clip = cmd._draw(n)
  share = Counter(cmd.clip_groups[c] for c in clip.tolist())
  assert share["bck"] == 0
  assert share["idle"] / n == pytest.approx(0.15 + 0.55 * 0.4, abs=0.01)
  assert share["turn"] / n == pytest.approx(0.30, abs=0.01)
  moving = clip != cmd.idle_clip
  error = (twist - cmd.twists[clip]).abs()[moving]
  assert float(error.max()) <= cmd.cfg.max_snap_error
  cmd.set_twist_limit((0.8, 0.6, 0.5))
  twist, clip = cmd._draw(n)
  fwd = twist[:, 0] > 0
  assert float(twist[fwd, 0].max()) <= 0.8 and float(twist[:, 2].abs().max()) <= 0.5
  # Uniform over vx 0.5-0.8: the top 0.01 m/s holds about 1/30 of the draws.
  assert float((twist[fwd, 0] > 0.79).float().mean()) < 0.06


def test_twist_limit_curriculum_stages() -> None:
  cmd = _indexed_command(UNICYCLE_TWISTS, sampling="library", **UNICYCLE_SPLIT)
  env = SimpleNamespace(
    common_step_counter=0, command_manager=SimpleNamespace(get_term=lambda _: cmd)
  )
  stages = [
    {"step": 0, "max_twist": (1.0, 0.6, 0.6)},
    {"step": 800, "max_twist": None},
  ]
  term_cfg = SimpleNamespace(params={"command_name": "motion", "stages": stages})
  term = twist_limit_curriculum(cast(Any, term_cfg), cast(Any, env))
  log = term(cast(Any, env), torch.arange(2), "motion", cast(Any, stages))
  assert cmd.twist_limit == (1.0, 0.6, 0.6)
  assert float(log["max_wz"]) == pytest.approx(0.6)
  env.common_step_counter = 799
  term(cast(Any, env), torch.arange(2), "motion", cast(Any, stages))
  assert cmd.twist_limit == (1.0, 0.6, 0.6)
  env.common_step_counter = 800
  log = term(cast(Any, env), torch.arange(2), "motion", cast(Any, stages))
  assert cmd.twist_limit is None and math.isinf(float(log["max_wz"]))


##
# Play: the twist pin, the reference ghost, the nominal plant.
##


def test_pinned_twist_replaces_the_sampler() -> None:
  cmd = _indexed_command(
    _unicycle_grid(),
    sampling="library",
    twist_metric_weights=UNICYCLE_TWIST_METRIC_WEIGHTS,
    fixed_twist=(0.53, 0.0, 0.22),
    **UNICYCLE_SPLIT,
  )
  # Library sampling commands the nearest clip's label, as in training.
  twist, clip = cmd._draw(64)
  assert set(clip.tolist()) == {cmd.nearest_clip((0.53, 0.0, 0.22))}
  torch.testing.assert_close(twist, torch.tensor([[0.55, 0.0, 0.2]]).repeat(64, 1))
  # The pin ignores the twist cap: it is an explicit choice.
  cmd.set_twist_limit((0.6, 0.6, 0.1))
  cmd._fixed_twist = (1.0, 0.0, 0.4)
  twist, _ = cmd._draw(4)
  torch.testing.assert_close(twist, torch.tensor([[1.0, 0.0, 0.4]]).repeat(4, 1))


def test_set_fixed_twist_switches_every_env_at_the_next_step() -> None:
  cmd = _fake_command()
  cmd.frame[:] = torch.tensor([50, 25, 75, 99])
  # Continuous sampling commands the pinned twist itself; walk_b (1.3) is nearest.
  cmd.set_fixed_twist((1.2, 0.0, 0.0))
  assert cmd.fixed_twist is None and cmd.clip.tolist() == [0, 0, 0, 0]  # posted only
  cmd._update_command(None)
  assert cmd.fixed_twist == (1.2, 0.0, 0.0)
  assert cmd.clip.tolist() == [1, 1, 1, 1]
  torch.testing.assert_close(cmd.twist_command, torch.tensor([[1.2, 0.0, 0.0]] * 4))
  # The timer-resample path: phase kept (then one step on), blending in.
  assert cmd.frame.tolist() == [41, 21, 61, 0]
  assert bool((cmd.alpha < 1.0).all())
  # Later timer resamples keep the pin; releasing it returns to the sampler.
  cmd._sample_twist = lambda n: torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1)  # type: ignore[method-assign]
  cmd._switch(torch.arange(4))
  assert cmd.clip.tolist() == [1, 1, 1, 1]
  cmd.set_fixed_twist(None)
  cmd._update_command(None)
  assert cmd.fixed_twist is None and cmd.clip.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
  "model, socket", [("robot", "rigid"), ("msk", "rigid"), ("msk", "compliant")]
)
def test_reference_ghost_is_the_robot_model_with_socket_and_osl(
  model: Model, socket: Socket
) -> None:
  spec = get_spec(model, socket)
  task_model = spec.compile()
  free = mujoco.mjtJoint.mjJNT_FREE
  joints = [
    task_model.joint(j).name
    for j in range(task_model.njnt)
    if task_model.jnt_type[j] != free
  ]
  cmd = GaitLibraryCommand.__new__(GaitLibraryCommand)
  cmd._env = cast(Any, SimpleNamespace(device="cpu"))
  cmd.cfg = GaitLibraryCommandCfg(library_dir="synthetic")
  cmd.robot = cast(Any, SimpleNamespace(joint_names=joints))
  ghost = cmd._build_ghost()
  g = ghost.model
  assert g.nu == 29 and g.nq == task_model.nq  # the robot model, at the socket

  # Shown: geom group 0 alone, the socket and OSL included, whether they collide or
  # not (the stock tracking ghost shows only non-colliding geoms).
  shown = g.geom_rgba[:, 3] > 0
  assert np.array_equal(shown, g.geom_group == 0)
  osl = [i for i in np.flatnonzero(shown) if g.geom(i).name.startswith("OSL_KA_L1_")]
  assert len(osl) == 7
  # The socket mesh collides and the foot shell does not: neither decides visibility.
  assert g.geom("OSL_KA_L1_osl_generic_socket").contype[0] == 1
  assert g.geom("OSL_KA_L1_osl_foot_assembly_geom_1").contype[0] == 0
  assert not g.site_rgba[:, 3].any() and not g.tendon_rgba[:, 3].any()

  # The ghost puts every body where the task's model has it, at any pose.
  rng = np.random.default_rng(0)
  task_data = mujoco.MjData(task_model)
  root = np.r_[0.3, -0.2, 0.95, 0.9, 0.1, -0.2, 0.3]
  root[3:] /= np.linalg.norm(root[3:])
  task_data.qpos[:7] = root
  joint_pos = rng.uniform(-0.3, 0.3, len(joints))
  for name, q in zip(joints, joint_pos, strict=True):
    task_data.qpos[task_model.joint(name).qposadr[0]] = q
  ghost_data = mujoco.MjData(g)
  ghost_data.qpos[ghost.root_adr : ghost.root_adr + 7] = root
  ghost_data.qpos[ghost.qpos_adr] = joint_pos[ghost.joint_cols.numpy()]
  mujoco.mj_kinematics(task_model, task_data)
  mujoco.mj_kinematics(g, ghost_data)
  for b in range(1, g.nbody):
    task_body = task_model.body(g.body(b).name).id
    np.testing.assert_allclose(ghost_data.xpos[b], task_data.xpos[task_body], atol=1e-9)

  # The native viewer draws it on any task's scene, muscles or not.
  scene = mujoco.MjvScene(task_model, maxgeom=1000)
  MujocoNativeDebugVisualizer(scene, task_model, env_idx=0).add_ghost_mesh(
    ghost_data.qpos.copy(), model=g
  )
  drawn = [geom for geom in scene.geoms[: scene.ngeom]]
  geoms = {geom.objid for geom in drawn if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM}
  assert geoms == set(np.flatnonzero(shown).tolist())


def test_no_disturbances_runs_the_validation_plant() -> None:
  play = osl_gait_library_env_cfg("robot", "compliant", "mimic", "play", "unicycle")
  motion = play.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg) and motion.debug_vis  # ghost
  assert (
    not load_env_cfg("Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT")
    .commands["motion"]
    .debug_vis
  )
  removed = strip_disturbances(play)
  assert set(removed) == PHYSICAL_EVENTS | {"servo_gains"}
  validation = osl_gait_library_env_cfg(
    "robot", "compliant", "mimic", "validation", "unicycle"
  )
  assert set(play.events) == set(validation.events)
  assert _delays(play) == _delays(validation) == {"osl": (5, 5), "human": (15, 15)}


def test_validation_groups_follow_clip_families() -> None:
  cmd = _indexed_command(UNICYCLE_TWISTS, sampling="library", **UNICYCLE_SPLIT)
  env = SimpleNamespace(
    num_envs=10, device="cpu", command_manager=SimpleNamespace(get_term=lambda _: cmd)
  )
  runner = GaitLibraryOnPolicyRunner.__new__(GaitLibraryOnPolicyRunner)
  groups = runner._validation_groups(cast(Any, env))
  assert set(groups) == {"idle", "fwd", "fwd_tight", "bck", "turn"}
  # Validation env i plays clip i % 8.
  assert groups["idle"].nonzero().flatten().tolist() == [0, 8]
  assert groups["fwd"].nonzero().flatten().tolist() == [1, 2, 9]
  assert groups["turn"].nonzero().flatten().tolist() == [6, 7]


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


UNICYCLE_LIBRARY = os.environ.get("MJLAB_OSL_UNICYCLE_LIBRARY", "")


@pytest.mark.slow
@pytest.mark.skipif(
  not UNICYCLE_LIBRARY, reason="set MJLAB_OSL_UNICYCLE_LIBRARY to a library dir"
)
@pytest.mark.parametrize("model, reward", [("robot", "mimic"), ("msk", "clf")])
def test_unicycle_env_rollout(model: Model, reward: Reward) -> None:
  """The unicycle task on the real library: every draw is in range per env, the
  actuator lags are held per episode, pushes reach the CLF, rewards stay finite."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.osl_gait_library.mdp.events import actuator_ctrl_ids

  cfg = osl_gait_library_env_cfg(model, "rigid", reward, "train", "unicycle")
  cfg.scene.num_envs = 8
  motion = cfg.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = UNICYCLE_LIBRARY
  motion.resampling_time_range = (0.3, 0.6)
  cfg.events["push_robot"].interval_range_s = (0.05, 0.1)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  cmd = cast(GaitLibraryCommand, env.command_manager.get_term("motion"))
  assert cmd.num_clips == 779 and set(cmd.clip_groups) == {
    "idle",
    "fwd",
    "fwd_tight",
    "bck",
    "turn",
  }
  dr_cfg = OslDrCfg()
  robot = env.scene["robot"]

  def ratio(field: str, ids: torch.Tensor, slot: int) -> torch.Tensor:
    value = getattr(env.sim.model, field)[:, ids, slot]
    return value / env.sim.get_default_field(field)[ids, slot]

  def within(x: torch.Tensor, bounds: tuple[float, float]) -> bool:
    return bool((x >= bounds[0] - 1e-5).all() and (x <= bounds[1] + 1e-5).all())

  motors = actuator_ctrl_ids(env, robot, "motor")
  assert motors.numel() == 2
  assert within(ratio("actuator_gainprm", motors, 0), dr_cfg.osl_motor_strength)
  servos = actuator_ctrl_ids(env, robot, "servo")
  if model == "robot":
    assert servos.numel() == 27
    assert within(ratio("actuator_gainprm", servos, 0), dr_cfg.servo_kp)
    assert within(ratio("actuator_biasprm", servos, 2), dr_cfg.servo_kd)
  else:
    muscles = actuator_ctrl_ids(env, robot, "muscle")
    assert muscles.numel() == 339
    force = ratio("actuator_gainprm", muscles, 2)
    assert within(force, dr_cfg.muscle_strength)
    torch.testing.assert_close(force, ratio("actuator_biasprm", muscles, 2))
  foot = [re.fullmatch(FOOT_GEOMS, name) is not None for name in robot.geom_names]
  ids = robot.indexing.geom_ids.long()
  friction = env.sim.model.geom_friction[:, ids, 0]
  feet = friction[:, torch.tensor(foot, device=device)]
  assert sum(foot) == 8 and within(feet, dr_cfg.foot_friction)
  assert bool((feet == feet[:, :1]).all())  # one draw per env, shared by the feet
  assert float(feet[:, 0].std()) > 0.0
  default = env.sim.get_default_field("geom_friction")[ids, 0]
  others = ~torch.tensor(foot, device=device)
  torch.testing.assert_close(friction[:, others], default[others].expand(8, -1))
  priority = env.sim.mj_model.geom_priority[robot.indexing.geom_ids.cpu().numpy()]
  foot_mask = np.array(foot)
  assert set(priority[foot_mask].tolist()) == {1}
  assert set(priority[~foot_mask].tolist()) == {0}
  torso = int(robot.indexing.body_ids[robot.body_names.index("torso")])
  com = (
    env.sim.model.body_ipos[:, torso] - env.sim.get_default_field("body_ipos")[torso]
  )
  assert within(com, dr_cfg.torso_com) and float(com.abs().max()) > 1e-3
  mass = (
    env.sim.model.body_mass[:, torso] / env.sim.get_default_field("body_mass")[torso]
  )
  assert within(mass, dr_cfg.torso_mass)

  env.reset()
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
  # The delayed commands start from the start pose, not from zero: every human lag
  # (>= 10 substeps) outlasts the first step's 5, so the servos still hold it.
  if model == "robot":
    start = env.sim.data.qpos.clone()
    joints = env.sim.mj_model.actuator_trnid[servos.cpu().numpy(), 0]
    adr = torch.as_tensor(env.sim.mj_model.jnt_qposadr[joints], device=device)
    env.step(zero)
    torch.testing.assert_close(
      env.sim.data.ctrl[:, servos], start[:, adr], atol=1e-5, rtol=0
    )
  pushed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
  for _ in range(150):
    _, reward_t, _, _, _ = env.step(zero)
    assert torch.isfinite(reward_t).all()
    pushed |= cmd._jump_step == env.common_step_counter
  assert bool(pushed.all())
  # The feet's own friction reaches their floor contacts (foot priority 1).
  data = env.sim.wp_data
  n = int(data.nacon.numpy()[0])
  pair = wp.to_torch(data.contact.geom)[:n].long()
  world = wp.to_torch(data.contact.worldid)[:n].long()
  contact_mu = wp.to_torch(data.contact.friction)[:n, 0]
  foot_ids = ids[torch.tensor(foot, device=device)]
  on_robot = torch.isin(pair, ids)
  on_foot = torch.isin(pair, foot_ids)
  floor = (on_foot[:, 0] & ~on_robot[:, 1]) | (on_foot[:, 1] & ~on_robot[:, 0])
  assert bool(floor.any())
  torch.testing.assert_close(contact_mu[floor], feet[world[floor], 0])
  for act in robot.actuators:
    buffer = act._delay_buffer
    assert buffer is not None
    lags = buffer.current_lags
    osl = tuple(act.cfg.target_names_expr) == OSL_TORQUE_JOINTS
    assert within(lags.float(), (0, 10) if osl else (10, 20))
  env.close()


def test_episodes_start_at_a_random_heading_on_the_env_origin() -> None:
  """With ``start_yaw_range`` the start frame's root lands on the env origin turned
  by a random heading (the robot is written there), and at 0 every episode starts
  on the clip's native heading exactly as before."""
  writes: dict[str, torch.Tensor] = {}
  cmd = _fake_command()
  robot = cast(Any, cmd.robot)
  robot.write_joint_state_to_sim = lambda pos, vel, env_ids: writes.update(q=pos)
  robot.write_root_state_to_sim = lambda state, env_ids: writes.update(root=state)
  robot.reset = lambda env_ids: None
  robot.set_joint_position_target = lambda target, env_ids: None
  ids = torch.arange(4)
  cmd._draw = lambda n: (  # type: ignore[method-assign]
    torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1),
    torch.zeros(n, dtype=torch.long),
  )
  native = planar_pose(cmd.data["root_pos"][0], cmd.data["root_quat"][0])

  torch.manual_seed(0)
  cmd._start_episode(ids)
  cmd._cache.clear()  # As _resample_command does after an episode start.
  frame = cmd.frame.clone()
  torch.testing.assert_close(cmd.anchor[:, 2], torch.zeros(4))
  ref = cmd.reference(0)
  torch.testing.assert_close(
    ref.root_pos_w[:, :2], torch.zeros(4, 2), atol=1e-6, rtol=0
  )

  cmd.cfg.start_yaw_range = math.pi
  torch.manual_seed(0)
  cmd._start_episode(ids)
  cmd._cache.clear()  # As _resample_command does after an episode start.
  yaw = cmd.anchor[:, 2]
  assert bool((yaw.abs() <= math.pi).all()) and float(yaw.std()) > 0.1
  assert torch.equal(cmd.frame, frame)  # The yaw draw comes after the phase draw.
  ref = cmd.reference(0)
  pose = planar_pose(ref.root_pos_w, ref.root_quat)
  torch.testing.assert_close(pose[:, :2], torch.zeros(4, 2), atol=1e-6, rtol=0)
  turned = torch.atan2(
    torch.sin(pose[:, 2] - native[frame, 2]), torch.cos(pose[:, 2] - native[frame, 2])
  )
  torch.testing.assert_close(turned, yaw, atol=1e-5, rtol=0)
  # The robot starts on the turned reference, velocities turned with it.
  torch.testing.assert_close(writes["root"][:, :3], ref.root_pos_w)
  torch.testing.assert_close(writes["root"][:, 3:7], ref.root_quat)
  torch.testing.assert_close(writes["root"][:, 7:10], ref.root_lin_vel_w)
  speed = ref.root_lin_vel_w[:, :2].norm(dim=-1)
  native_speed = cmd.data["root_lin_vel"][0, frame, :2].norm(dim=-1)
  torch.testing.assert_close(speed, native_speed, atol=1e-5, rtol=0)
  # Validation's fixed clips keep the native heading.
  cmd.cfg.fixed_clips = True
  cmd._start_episode(ids)
  cmd._cache.clear()  # As _resample_command does after an episode start.
  torch.testing.assert_close(cmd.anchor[:, 2], torch.zeros(4))
