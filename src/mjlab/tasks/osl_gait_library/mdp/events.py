"""Domain-randomization events for the OSL models, and a push the CLF accounts for.

The OSL models mix three actuator kinds behind ``XmlActuator`` groups: position servos
(the robot model's human joints), torque motors (the OSL knee and ankle) and muscles
(the musculoskeletal model's human). mjlab's ``dr.pd_gains`` covers only the first and
raises on the others, and its actuator selection indexes actuator groups, so each kind
gets its own strength term here, selected by kind rather than by name:

* servos: stiffness and damping, scaled independently (G1's ``pd_gains``);
* motors: the gain, i.e. the torque per unit command;
* muscles: the peak isometric force, which scales the active and passive force alike.

Every draw is per actuator and per env, from the model's default values, so repeated
draws never compound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import GaitLibraryCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")

ActuatorKind = Literal["servo", "motor", "muscle"]


def actuator_ctrl_ids(
  env: ManagerBasedRlEnv, entity: Entity, kind: ActuatorKind
) -> torch.Tensor:
  """Global ctrl indices of the entity's actuators of one kind."""
  model = env.sim.mj_model
  ids: list[int] = []
  for act in entity.actuators:
    if not isinstance(act, XmlActuator):
      continue
    for i in act.global_ctrl_ids.tolist():
      muscle = int(model.actuator_dyntype[i]) == mujoco.mjtDyn.mjDYN_MUSCLE
      if kind == "servo":
        keep = act.command_field == "position"
      elif kind == "motor":
        keep = act.command_field == "effort" and not muscle
      else:
        keep = muscle
      if keep:
        ids.append(int(i))
  return torch.tensor(ids, dtype=torch.long, device=env.device)


def _selection(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  kind: ActuatorKind,
) -> tuple[torch.Tensor, torch.Tensor]:
  rows = (
    torch.arange(env.num_envs, device=env.device)
    if env_ids is None
    else env_ids.to(env.device, dtype=torch.long)
  )
  return rows, actuator_ctrl_ids(env, env.scene[asset_cfg.name], kind)


def _uniform(bounds: tuple[float, float], shape: tuple[int, int], device: str):
  return torch.empty(shape, device=device).uniform_(bounds[0], bounds[1])


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def servo_pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Scale every position servo's stiffness and damping, as G1's ``pd_gains``."""
  rows, ctrl = _selection(env, env_ids, asset_cfg, "servo")
  if ctrl.numel() == 0:
    return
  shape = (len(rows), len(ctrl))
  kp = _uniform(kp_range, shape, env.device)
  kd = _uniform(kd_range, shape, env.device)
  gain = env.sim.get_default_field("actuator_gainprm")
  bias = env.sim.get_default_field("actuator_biasprm")
  model = env.sim.model
  model.actuator_gainprm[rows[:, None], ctrl, 0] = gain[ctrl, 0] * kp
  model.actuator_biasprm[rows[:, None], ctrl, 1] = bias[ctrl, 1] * kp
  model.actuator_biasprm[rows[:, None], ctrl, 2] = bias[ctrl, 2] * kd


@requires_model_fields("actuator_gainprm")
def motor_strength(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Scale every torque motor's gain: its torque constant and drivetrain losses."""
  rows, ctrl = _selection(env, env_ids, asset_cfg, "motor")
  if ctrl.numel() == 0:
    return
  scale = _uniform(scale_range, (len(rows), len(ctrl)), env.device)
  gain = env.sim.get_default_field("actuator_gainprm")
  env.sim.model.actuator_gainprm[rows[:, None], ctrl, 0] = gain[ctrl, 0] * scale


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def muscle_strength(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Scale every muscle's peak isometric force.

  MuJoCo's muscle gain and bias both read the force from slot 2 (compiled from
  ``scale`` when the XML leaves it negative), so both are scaled: a stronger muscle
  is also stiffer passively, as a larger cross-section is.
  """
  rows, ctrl = _selection(env, env_ids, asset_cfg, "muscle")
  if ctrl.numel() == 0:
    return
  scale = _uniform(scale_range, (len(rows), len(ctrl)), env.device)
  gain = env.sim.get_default_field("actuator_gainprm")
  bias = env.sim.get_default_field("actuator_biasprm")
  model = env.sim.model
  model.actuator_gainprm[rows[:, None], ctrl, 2] = gain[ctrl, 2] * scale
  model.actuator_biasprm[rows[:, None], ctrl, 2] = bias[ctrl, 2] * scale


def push_by_setting_velocity_marked(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  command_name: str = "motion",
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """mjlab's root-velocity kick, flagged to the library command as a discontinuity
  so the CLF does not difference its error rate across the kick."""
  push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
  cmd = cast(GaitLibraryCommand, env.command_manager.get_term(command_name))
  cmd.mark_discontinuity(slice(None) if env_ids is None else env_ids)
