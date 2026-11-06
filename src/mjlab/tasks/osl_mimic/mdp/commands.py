"""The reference-trajectory command for musclemimic-style imitation.

It reproduces musclemimic's trajectory handling (``loco_mujoco/trajectory/handler.py``
and ``musclemimic/environments/base.py``) on mjlab's 100 Hz motion NPZ:

* **Starts.** Every episode starts from the exact reference state at a frame drawn
  uniformly from ``[0, T)`` (``random_start``), or at frame 0 (validation and
  play). No noise is added. Muscle activations restart at zero, as MuJoCo's reset
  leaves them.
* **Re-centring.** The reference's root xy at the start frame is moved onto the
  env origin, and every reference position is shifted by the same offset for the
  rest of the episode. This is musclemimic's ``xy_offset``.
* **Timing.** musclemimic advances the trajectory index *before* it computes the
  observation, termination and reward of a step. mjlab advances commands after
  rewards. So :attr:`frame` is the frame the last observation was built at, and
  rewards and terminations compare the post-step state with :attr:`next_frame`.
  A freshly reset env is not advanced in the step it was reset in.
* **End.** The index saturates at the last frame. The episode ends when the next
  frame is the last one (see ``terminations.trajectory_end``).

Sites are musclemimic's 17 mimic sites. Each sits at its body's origin with the
body's orientation, so their reference comes straight from the NPZ body arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import get_spec
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.config.osl.mdp import check_motion_fps
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

from .kinematics import relative_site_quantities, relative_site_velocities
from .reference import MimicLookahead, MimicReference

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# musclemimic's ``body2sites_for_mimic`` for OSLFullBody, in ``sites_for_mimic``
# order. The first is the main site every relative quantity is taken against.
MIMIC_BODIES: tuple[str, ...] = (
  "pelvis",
  "lumbar1",
  "head",
  "humerus_l",
  "ulna_l",
  "lunate_l",
  "humerus_r",
  "ulna_r",
  "lunate_r",
  "femur_l",
  "tibia_l",
  "talus_l",
  "toes_l",
  "femur_r",
  "OSL_KA_L1_osl_knee_assembly",
  "OSL_KA_L1_osl_ankle_assembly",
  "OSL_KA_L1_osl_foot_assembly",
)

ROOT_BODY = "Full Body"


def _clip_joint_names() -> list[str]:
  """The joints a clip names: the rigid robot's 73, in model order."""
  spec = get_spec(
    "robot", "rigid"
  )  # Bind: iterating a temporary spec reads freed memory.
  return [j.name for j in spec.joints if j.name != "root"]


def pad_clip_joints(
  joint_pos: torch.Tensor, joint_vel: torch.Tensor, joint_names: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
  """A clip's (T, 73) joint arrays on the entity's joints; absent joints are zero.

  Clips name the rigid robot's joints. The compliant socket's four joints (and the
  musculoskeletal model's fingers, when kept) are absent and held at zero, the
  neutral socket. A clip that already matches the entity passes through.
  """
  width = joint_pos.shape[-1]
  if width == len(joint_names):
    return joint_pos, joint_vel
  clip_names = _clip_joint_names()
  if width != len(clip_names):
    raise ValueError(
      f"Motion has {width} joints; expected {len(joint_names)} (this entity) or "
      f"{len(clip_names)} (the rigid robot)."
    )
  columns = [joint_names.index(name) for name in clip_names]
  pos = joint_pos.new_zeros(*joint_pos.shape[:-1], len(joint_names))
  vel = torch.zeros_like(pos)
  pos[..., columns] = joint_pos
  vel[..., columns] = joint_vel
  return pos, vel


@dataclass(kw_only=True)
class MimicMotionCommandCfg(CommandTermCfg):
  """Configuration for :class:`MimicMotionCommand`."""

  motion_file: str = ""
  """mjlab motion NPZ written by clipper at the control rate (100 Hz here)."""
  entity_name: str = "robot"
  site_bodies: tuple[str, ...] = MIMIC_BODIES
  random_start: bool = True
  """Start at a uniformly random frame (musclemimic's ``start_from_random_step``).
  False starts every episode at frame 0."""
  lookahead_steps: int = 5
  """``GoalTrajMimic.n_step_lookahead``: frames in the goal, the current one included."""
  lookahead_stride: int = 20
  """``GoalTrajMimic.n_step_stride``, in frames."""
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  """Never resample mid-episode; starts happen only on reset."""

  def build(self, env: ManagerBasedRlEnv) -> MimicMotionCommand:
    return MimicMotionCommand(self, env)


class MimicMotionCommand(CommandTerm):
  cfg: MimicMotionCommandCfg

  def __init__(self, cfg: MimicMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if not cfg.motion_file:
      raise ValueError(
        "No motion file: pass --env.commands.motion.motion-file <clip.npz> "
        "(a clipper --format mjlab clip at the task's control rate)."
      )
    check_motion_fps(cfg.motion_file, env.step_dt)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._load_motion(cfg.motion_file)

    n = self.num_envs
    self.frame = torch.zeros(n, dtype=torch.long, device=self.device)
    self.start_frame = torch.zeros(n, dtype=torch.long, device=self.device)
    self.pos_offset = torch.zeros(n, 3, device=self.device)
    self._just_reset = torch.zeros(n, dtype=torch.bool, device=self.device)

  ##
  # Loading.
  ##

  def _load_motion(self, path: str) -> None:
    body_names = list(self.robot.body_names)
    joint_names = list(self.robot.joint_names)
    with np.load(path) as data:
      arrays = {
        k: torch.tensor(data[k], dtype=torch.float32, device=self.device)
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        )
      }
    if arrays["body_pos_w"].shape[1] != len(body_names):
      raise ValueError(
        f"Motion has {arrays['body_pos_w'].shape[1]} bodies; the entity has "
        f"{len(body_names)}. Body arrays must follow the entity's body order."
      )
    self.num_frames = int(arrays["joint_pos"].shape[0])
    if self.num_frames < 2:
      raise ValueError("A motion needs at least two frames.")

    self.ref_joint_pos, self.ref_joint_vel = pad_clip_joints(
      arrays["joint_pos"], arrays["joint_vel"], joint_names
    )

    root = body_names.index(ROOT_BODY)
    self.ref_root_pos = arrays["body_pos_w"][:, root]
    self.ref_root_quat = arrays["body_quat_w"][:, root]
    self.ref_root_lin_vel = arrays["body_lin_vel_w"][:, root]
    # A free joint's angular qvel is in the body frame.
    self.ref_root_ang_vel_b = quat_apply_inverse(
      self.ref_root_quat, arrays["body_ang_vel_w"][:, root]
    )
    self.ref_root_ang_vel_w = arrays["body_ang_vel_w"][:, root]

    self.site_body_ids = [body_names.index(b) for b in self.cfg.site_bodies]
    self.ref_site_pos = arrays["body_pos_w"][:, self.site_body_ids]
    self.ref_site_mat = matrix_from_quat(arrays["body_quat_w"][:, self.site_body_ids])
    ang = arrays["body_ang_vel_w"][:, self.site_body_ids]
    lin = arrays["body_lin_vel_w"][:, self.site_body_ids]
    # Relative quantities are differences, so the per-env offset never enters them;
    # precompute every frame once.
    self.ref_rpos, self.ref_rangles, self.ref_rvel = relative_site_quantities(
      self.ref_site_pos, self.ref_site_mat, ang, lin
    )
    self.ref_v_rel, self.ref_omega_rel = relative_site_velocities(
      self.ref_site_mat, ang, lin
    )

  ##
  # Frames.
  ##

  @property
  def next_frame(self) -> torch.Tensor:
    """The frame rewards and terminations compare the post-step state with."""
    return torch.clamp(self.frame + 1, max=self.num_frames - 1)

  def lookahead_frames(self) -> torch.Tensor:
    """(B, L) goal frames ``frame + i * stride``, clipped to the clip."""
    offsets = torch.arange(self.cfg.lookahead_steps, device=self.device)
    frames = self.frame[:, None] + offsets[None, :] * self.cfg.lookahead_stride
    return torch.clamp(frames, max=self.num_frames - 1)

  @property
  def phase(self) -> torch.Tensor:
    """``GoalTrajMimic`` motion phase: frame / T."""
    return self.frame.float() / float(self.num_frames)

  ##
  # Reference accessors. Positions carry the per-env offset (sim coordinates).
  ##

  def ref_root_pos_w(self, frames: torch.Tensor) -> torch.Tensor:
    return self.ref_root_pos[frames] + self.pos_offset

  def ref_site_pos_w(self, frames: torch.Tensor) -> torch.Tensor:
    return self.ref_site_pos[frames] + self.pos_offset[:, None, :]

  def ref_root_qvel(self, frames: torch.Tensor) -> torch.Tensor:
    """Free-joint qvel: world linear velocity, body-frame angular velocity."""
    return torch.cat(
      [self.ref_root_lin_vel[frames], self.ref_root_ang_vel_b[frames]], -1
    )

  ##
  # The imitation terms' interface (see ``reference.py``).
  ##

  def reference(self, offset: int) -> MimicReference:
    f = torch.clamp(self.frame + offset, max=self.num_frames - 1)
    return MimicReference(
      root_pos_w=self.ref_root_pos_w(f),
      root_quat=self.ref_root_quat[f],
      root_lin_vel_w=self.ref_root_lin_vel[f],
      root_ang_vel_b=self.ref_root_ang_vel_b[f],
      joint_pos=self.ref_joint_pos[f],
      joint_vel=self.ref_joint_vel[f],
      site_pos_w=self.ref_site_pos_w(f),
      rpos=self.ref_rpos[f],
      rangles=self.ref_rangles[f],
      rvel=self.ref_rvel[f],
      v_rel=self.ref_v_rel[f],
      omega_rel=self.ref_omega_rel[f],
    )

  def lookahead(self) -> MimicLookahead:
    frames = self.lookahead_frames()
    now, future = frames[:, :1], frames[:, 1:]
    return MimicLookahead(
      root_pos_delta=self.ref_root_pos[future] - self.ref_root_pos[now],
      root_qvel_delta=self.ref_root_qvel(future) - self.ref_root_qvel(now),
      rpos=self.ref_rpos[future],
    )

  @property
  def reference_jumped(self) -> torch.Tensor:
    """A single clip only jumps at episode starts, which the CLF handles itself."""
    return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  ##
  # Simulation state, in the same conventions.
  ##

  @property
  def root_qpos(self) -> torch.Tensor:
    """(B, 7) free-joint qpos read straight from qpos, as musclemimic does."""
    return self.robot.data.data.qpos[:, self.robot.indexing.free_joint_q_adr]

  @property
  def root_qvel(self) -> torch.Tensor:
    """(B, 6) free-joint qvel: world linear, body-frame angular."""
    return self.robot.data.data.qvel[:, self.robot.indexing.free_joint_v_adr]

  def site_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World pose and velocity of the mimic sites (their bodies' frames)."""
    data = self.robot.data
    ids = self.site_body_ids
    return (
      data.body_link_pos_w[:, ids],
      matrix_from_quat(data.body_link_quat_w[:, ids]),
      data.body_link_ang_vel_w[:, ids],
      data.body_link_lin_vel_w[:, ids],
    )

  ##
  # CommandTerm interface.
  ##

  @property
  def command(self) -> torch.Tensor:
    return self.ref_joint_pos[self.next_frame]

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    if self.cfg.random_start:
      start = torch.randint(0, self.num_frames, (len(env_ids),), device=self.device)
    else:
      start = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
    self.start_frame[env_ids] = start
    self.frame[env_ids] = start
    self._just_reset[env_ids] = True

    origins = self._env.scene.env_origins[env_ids]
    offset = origins.clone()
    offset[:, :2] -= self.ref_root_pos[start, :2]
    self.pos_offset[env_ids] = offset

    root_state = torch.cat(
      [
        self.ref_root_pos[start] + offset,
        self.ref_root_quat[start],
        self.ref_root_lin_vel[start],
        self.ref_root_ang_vel_w[start],
      ],
      dim=-1,
    )
    self.robot.write_joint_state_to_sim(
      self.ref_joint_pos[start], self.ref_joint_vel[start], env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      advance = ~self._just_reset
      self.frame = torch.where(advance, self.next_frame, self.frame)
      self._just_reset.zero_()
    else:
      self._just_reset[env_ids] = False
