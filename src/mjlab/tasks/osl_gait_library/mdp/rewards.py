"""The twist-following reward that rides next to the CLF imitation reward."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply

from .commands import GaitLibraryCommand, heading_twist

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class stride_twist(ManagerTermBase):
  """``exp(-sum(((mean twist - target) / std)^2))`` over the last stride.

  The library's twist labels are stride means: a clip's root velocity swings through
  every stride, and only its average equals the label. So an instantaneous twist
  reward (sesteban's ``twist_tracking``) cannot score even a perfect replay near 1,
  nor tell neighbouring commands apart. Here the robot's planar twist [forward, left,
  yaw rate] (heading frame; forward is the root's body -y) is averaged over the
  snapped clip's own period of N frames:

  * a box window of N frames on vx and vy;
  * a triangular one (a box of boxes, 2N - 1 frames) on wz, whose in-stride swing is
    the largest and which a slightly different cadence would otherwise leak.

  The target is the same window over the served reference's twist, plus the
  command's offset from the clip's label (the snap error). That is exactly the
  command in steady state, and it follows the reference through clip switches. A
  perfect tracker scores 1 at any commanded speed; one 0.05 m/s grid step off scores
  0.78 at ``std`` 0.1.

  Diagnostics for the last step are kept in :attr:`components`, (B,) each.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    cmd = cast(
      GaitLibraryCommand, env.command_manager.get_term(cfg.params["command_name"])
    )
    self._len = 2 * int(cmd.num_frames.max())
    n = env.num_envs
    self._robot = torch.zeros(n, self._len, 3, device=env.device)
    self._ref = torch.zeros(n, self._len, 3, device=env.device)
    self._count = torch.zeros(n, dtype=torch.long, device=env.device)
    self._slots = torch.arange(self._len, device=env.device)
    self._rows = torch.arange(n, device=env.device)
    self.components: dict[str, torch.Tensor] = {}

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._count[slice(None) if env_ids is None else env_ids] = 0

  def __call__(
    self, env: ManagerBasedRlEnv, command_name: str, std: float
  ) -> torch.Tensor:
    cmd = cast(GaitLibraryCommand, env.command_manager.get_term(command_name))
    data = cmd.robot.data
    robot = heading_twist(
      data.root_link_quat_w, data.root_link_lin_vel_w, data.root_link_ang_vel_w
    )
    ref = cmd.reference(1)
    ref_twist = heading_twist(
      ref.root_quat, ref.root_lin_vel_w, quat_apply(ref.root_quat, ref.root_ang_vel_b)
    )

    slot = self._count % self._len
    self._robot[self._rows, slot] = robot
    self._ref[self._rows, slot] = ref_twist
    self._count += 1

    age = (slot[:, None] - self._slots[None, :]) % self._len  # 0 = this step
    valid = (age < self._count[:, None]).float()
    period = cmd.num_frames[cmd.clip].float()[:, None]
    lag = age.float()
    box = (lag < period).float() * valid
    tri = (period - (lag - (period - 1.0)).abs()).clamp(min=0.0) * valid

    def window(buf: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
      return (buf * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-6)

    robot_mean = torch.cat(
      [window(self._robot, box)[:, :2], window(self._robot, tri)[:, 2:]], dim=-1
    )
    ref_mean = torch.cat(
      [window(self._ref, box)[:, :2], window(self._ref, tri)[:, 2:]], dim=-1
    )
    target = ref_mean + (cmd.twist_command - cmd.twists[cmd.clip])
    err = robot_mean - target
    reward = torch.exp(-(err / std).square().sum(-1))

    self.components = {
      "twist_reward": reward,
      "twist_err_vx": err[:, 0].abs(),
      "twist_err_vy": err[:, 1].abs(),
      "twist_err_wz": err[:, 2].abs(),
      "twist_vx_cmd": cmd.twist_command[:, 0],
      "twist_vx_robot": robot_mean[:, 0],
    }
    return reward
