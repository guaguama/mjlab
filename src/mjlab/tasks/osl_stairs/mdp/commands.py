"""The single-clip reference command on a clip's own terrain.

:class:`~mjlab.tasks.osl_mimic.mdp.MimicMotionCommand` (musclemimic's trajectory
handling: random start frame, exact reference state, end at the last frame, the
``frame``/``next_frame`` timing), with three changes:

* **No re-centring.** The clip's root and its terrain share TERRA's world frame, and
  the terrain is static scene geometry, the same in every env. So every episode
  starts at the reference's own world position and heading, and the reference is
  never shifted: the env origin, zero in this task's scene, is the only offset.
* **Start pose held by the servos.** As in the gait-library command: the reset zeroes
  the joint position targets, and the env writes them to the actuators once more
  before the policy acts, which fills a freshly reset actuator delay history. The
  start frame's joint positions are written as the targets instead.
* **Pushes are discontinuities.** The randomization's pushes call
  :meth:`StairsMotionCommand.mark_discontinuity`, and the next reward sees
  ``reference_jumped``, so the CLF skips its finite difference across the kick.

At construction the command checks that the scene's terrain boxes are the ones its
``terrain_file`` (or the motion's ``<clip>_terrain.json``) describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.osl_mimic.mdp import MimicMotionCommand, MimicMotionCommandCfg
from mjlab.tasks.osl_stairs.terrain import (
  check_scene_terrain,
  load_terrain,
  resolve_terrain_file,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class StairsMotionCommandCfg(MimicMotionCommandCfg):
  """Configuration for :class:`StairsMotionCommand`."""

  terrain_file: str = ""
  """TERRA TerrainSpec JSON of the clip's static boxes. Empty reads the motion's
  sidecar, ``<clip>_terrain.json`` beside ``<clip>.npz``."""

  def build(self, env: ManagerBasedRlEnv) -> StairsMotionCommand:
    return StairsMotionCommand(self, env)


class StairsMotionCommand(MimicMotionCommand):
  def __init__(self, cfg: StairsMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.terrain_path = resolve_terrain_file(cfg.motion_file, cfg.terrain_file)
    self.terrain = load_terrain(self.terrain_path)
    check_scene_terrain(env.sim.mj_model, self.terrain, self.terrain_path)
    self._jump_step = torch.full(
      (self.num_envs,), -10, dtype=torch.long, device=self.device
    )

  @property
  def reference_jumped(self) -> torch.Tensor:
    return (self._env.common_step_counter - self._jump_step) == 1

  def mark_discontinuity(self, env_ids: torch.Tensor | slice) -> None:
    """Flag a jump of the robot's state, such as a push, made after this step's
    reward. The next reward then sees ``reference_jumped``."""
    self._jump_step[env_ids] = self._env.common_step_counter

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
    self._jump_step[env_ids] = -10

    offset = self._env.scene.env_origins[env_ids].clone()
    self.pos_offset[env_ids] = offset

    joint_pos = self.ref_joint_pos[start]
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
      joint_pos, self.ref_joint_vel[start], env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
