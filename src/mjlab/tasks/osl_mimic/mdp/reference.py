"""The interface between a reference command and the imitation terms.

The reward, observation and termination terms read the reference only through
:class:`MimicReferenceSource`. :class:`~.commands.MimicMotionCommand` implements it for
one clip; the gait-library command implements it for looping, blended clips. Every
tensor is per env and in simulation coordinates (world frame, env offset applied).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass
class MimicReference:
  """The reference at one control step."""

  root_pos_w: torch.Tensor
  """(B, 3) free-joint position."""
  root_quat: torch.Tensor
  """(B, 4) free-joint orientation, (w, x, y, z)."""
  root_lin_vel_w: torch.Tensor
  """(B, 3) free-joint linear velocity, world frame."""
  root_ang_vel_b: torch.Tensor
  """(B, 3) free-joint angular velocity, body frame (MuJoCo's qvel convention)."""
  joint_pos: torch.Tensor
  """(B, J) every non-root joint, entity order."""
  joint_vel: torch.Tensor
  """(B, J)"""
  site_pos_w: torch.Tensor
  """(B, S, 3) mimic site positions."""
  rpos: torch.Tensor
  """(B, K, 3) site positions relative to the main site, world frame."""
  rangles: torch.Tensor
  """(B, K, 3) site orientations relative to the main site, as rotation vectors."""
  rvel: torch.Tensor
  """(B, K, 6) musclemimic's relative site velocities (see ``kinematics``)."""
  v_rel: torch.Tensor
  """(B, K, 3) ``d/dt rpos``, world frame (the CLF's pairing)."""
  omega_rel: torch.Tensor
  """(B, K, 3) relative angular velocity in the main-site frame (the CLF's)."""

  @property
  def root_qvel(self) -> torch.Tensor:
    """(B, 6) free-joint qvel: world linear, body-frame angular."""
    return torch.cat([self.root_lin_vel_w, self.root_ang_vel_b], dim=-1)


@dataclass
class MimicLookahead:
  """``GoalTrajMimic``'s concise lookahead: the reference ahead of the current step."""

  root_pos_delta: torch.Tensor
  """(B, L, 3) root position at each lookahead step minus the current one."""
  root_qvel_delta: torch.Tensor
  """(B, L, 6) root qvel at each lookahead step minus the current one."""
  rpos: torch.Tensor
  """(B, L, K, 3) relative site positions at each lookahead step."""


class MimicReferenceSource(Protocol):
  """What the imitation terms need from a reference command."""

  robot: Entity
  site_body_ids: list[int]

  @property
  def root_qpos(self) -> torch.Tensor:
    """(B, 7) the robot's free-joint qpos."""
    ...

  @property
  def root_qvel(self) -> torch.Tensor:
    """(B, 6) the robot's free-joint qvel."""
    ...

  @property
  def phase(self) -> torch.Tensor:
    """(B,) the goal's motion phase in [0, 1)."""
    ...

  @property
  def reference_jumped(self) -> torch.Tensor:
    """(B,) bool: the reference was moved discontinuously since the last reward, so
    a finite difference across the jump is meaningless (the CLF restarts there)."""
    ...

  def site_state(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The robot's mimic sites: world positions, orientations, angular and linear
    velocities."""
    ...

  def reference(self, offset: int) -> MimicReference:
    """The reference ``offset`` control steps after the current one. Rewards and
    terminations read ``offset=1`` (musclemimic advances before it scores);
    observations read ``offset=0``."""
    ...

  def lookahead(self) -> MimicLookahead:
    """The goal observation's lookahead from the current step."""
    ...
