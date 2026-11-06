"""The gait-library terms added to the musclemimic observation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from .commands import GaitLibraryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> GaitLibraryCommand:
  return cast(GaitLibraryCommand, env.command_manager.get_term(command_name))


def twist_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(B, 3) the commanded twist [vx, vy, wz]: zero on the idle clip."""
  return _command(env, command_name).twist_command


def stride_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(B, 2) the stride clock ``(sin, cos)(2 pi phase)``, blanked to (0, 0) while idle,
  as in sesteban's ``motion_phase``."""
  cmd = _command(env, command_name)
  angle = 2.0 * math.pi * cmd.phase
  clock = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
  return clock * (~cmd.is_idle).float()[:, None]
