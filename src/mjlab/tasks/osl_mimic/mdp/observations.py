"""musclemimic's OSLFullBody observation, term by term.

``MyoFullBody._get_observation_specification`` followed by ``GoalTrajMimic``'s goal
(``use_concise_lookahead=True``, ``enable_motion_phase=True``). Actor and critic
see the same vector; nothing is corrupted.

One timing difference remains. musclemimic builds the observation straight after
the physics substeps, so its site positions, site velocities and touch readings lag
``qpos`` by one 2 ms substep. mjlab refreshes kinematics before observing, so they
are current here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mujoco
import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationTermCfg

from .kinematics import relative_site_quantities
from .reference import MimicReferenceSource

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

TOUCH_SENSORS: tuple[str, ...] = ("r_foot", "r_toes", "l_foot", "l_toes")


def _command(env: ManagerBasedRlEnv, command_name: str) -> MimicReferenceSource:
  return cast(MimicReferenceSource, env.command_manager.get_term(command_name))


def root_pos_no_xy(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``FreeJointPosNoXY``: root height and quaternion, (B, 5)."""
  return _command(env, command_name).root_qpos[:, 2:7]


def root_qvel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``FreeJointVel``: world linear and body-frame angular velocity, (B, 6)."""
  return _command(env, command_name).root_qvel


def joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``JointPosArray``: absolute positions of every non-root joint, model order."""
  return _command(env, command_name).robot.data.joint_pos


def joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``JointVelArray``: velocities of every non-root joint, model order."""
  return _command(env, command_name).robot.data.joint_vel


def touch(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  """The four foot touch sensors, in musclemimic's order, (B, 4)."""
  return torch.cat(
    [env.scene[f"{entity_name}/{name}"].data for name in TOUCH_SENSORS], dim=-1
  )


class muscle_state(ManagerTermBase):
  """Length, velocity, force, excitation (``ctrl``) and activation of every muscle.

  musclemimic interleaves these per muscle; here they come in five blocks of 339.
  A fixed permutation of input features changes nothing the network can learn.
  Excitation is the clipped command, which can be negative: musclemimic's muscle
  ctrlrange is [-1, 1], and MuJoCo clamps it to [0, 1] only inside the activation
  dynamics. The muscle action term here writes the same clipped value to ``ctrl``.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    model = env.sim.mj_model
    prefix = cfg.params.get("entity_name", "robot") + "/"
    ids = [
      i
      for i in range(model.nu)
      if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
      and model.actuator(i).name.startswith(prefix)
    ]
    if not ids:
      raise ValueError("muscle_state needs an entity with muscle actuators.")
    self._ctrl_ids = torch.tensor(ids, device=env.device)
    self._act_ids = torch.tensor(
      [int(model.actuator_actadr[i]) for i in ids], device=env.device
    )

  def __call__(
    self, env: ManagerBasedRlEnv, entity_name: str = "robot"
  ) -> torch.Tensor:
    del entity_name
    data = env.sim.data
    ids = self._ctrl_ids
    return torch.cat(
      [
        data.actuator_length[:, ids],
        data.actuator_velocity[:, ids],
        data.actuator_force[:, ids],
        data.ctrl[:, ids],
        data.act[:, self._act_ids],
      ],
      dim=-1,
    )


def mimic_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``GoalTrajMimic`` with the concise lookahead, (B, 469) for 17 sites.

  1. The current site rpos, rangles and rvel (16 x 12), from the simulation.
  2. The reference site rpos at the current frame (48).
  3. For each of the next four lookahead frames (every 20 frames, 0.2 s): the root
     position and root qvel deltas from the current reference frame (3 + 6, raw
     qpos/qvel differences) and that frame's site rpos (48).
  4. The motion phase, frame / T.
  """
  cmd = _command(env, command_name)
  pos, mat, ang, lin = cmd.site_state()
  rpos, rangles, rvel = relative_site_quantities(pos, mat, ang, lin)
  batch = rpos.shape[0]

  ahead = cmd.lookahead()
  parts = [
    rpos.reshape(batch, -1),
    rangles.reshape(batch, -1),
    rvel.reshape(batch, -1),
    cmd.reference(0).rpos.reshape(batch, -1),
  ]
  for i in range(ahead.rpos.shape[1]):
    parts.append(ahead.root_pos_delta[:, i])
    parts.append(ahead.root_qvel_delta[:, i])
    parts.append(ahead.rpos[:, i].reshape(batch, -1))
  parts.append(cmd.phase[:, None])
  return torch.cat(parts, dim=-1)
