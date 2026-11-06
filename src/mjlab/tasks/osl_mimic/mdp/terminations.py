"""musclemimic's terminal-state checks, against the reference one step ahead.

``MeanRelativeSiteDeviationWithRootTerminalStateHandler`` is three of these OR-ed
together (training); ``MeanRelativeSiteDeviationTerminalStateHandler`` is the first
alone (validation). Source: ``musclemimic/core/terminal_state_handler/
enhanced_fullbody.py``. The trajectory end is ``reached_trajectory_end``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import MimicMotionCommand
from .kinematics import quat_angle, relative_site_quantities
from .reference import MimicReferenceSource

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> MimicReferenceSource:
  return cast(MimicReferenceSource, env.command_manager.get_term(command_name))


def mean_site_deviation(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """Mean over the 16 sites of ``||rpos - rpos_ref||`` above ``threshold`` (m)."""
  cmd = _command(env, command_name)
  pos, mat, ang, lin = cmd.site_state()
  rpos, _, _ = relative_site_quantities(pos, mat, ang, lin)
  deviation = (rpos - cmd.reference(1).rpos).norm(dim=-1).mean(-1)
  return deviation > threshold


def root_site_deviation(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """Absolute distance of the pelvis site from its re-centred reference (m)."""
  cmd = _command(env, command_name)
  pelvis = cmd.robot.data.body_link_pos_w[:, cmd.site_body_ids[0]]
  reference = cmd.reference(1).site_pos_w[:, 0]
  return (pelvis - reference).norm(dim=-1) > threshold


def root_orientation_deviation(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """Geodesic angle between the root quaternion and its reference (rad)."""
  cmd = _command(env, command_name)
  angle = quat_angle(cmd.root_qpos[:, 3:7], cmd.reference(1).root_quat)
  return angle > threshold


def trajectory_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The next frame is the clip's last: musclemimic's episode end (one clip only)."""
  cmd = cast(MimicMotionCommand, env.command_manager.get_term(command_name))
  return cmd.next_frame >= cmd.num_frames - 1
