"""OSL motion padding, reset constraints and gravity-relative tilt termination."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import SOCKET_JOINTS, get_spec
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


def _clip_joint_names() -> list[str]:
  """The joints a motion clip names: the rigid robot's 73, in spec order.

  Clips are generated from the rigid model, so they do not carry the compliant
  socket's four joints.
  """
  # Bind the spec: iterating the joints of a temporary spec reads freed memory.
  spec = get_spec("robot", "rigid")
  return [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]


def check_motion_fps(motion_file: str | Path, step_dt: float) -> float:
  """Raise unless a motion NPZ was written at the env's control rate.

  mjlab's motion loaders advance one frame per env step and never read ``fps``, so a
  clip at any other rate silently plays at the wrong speed. clipper writes ``fps``
  into every NPZ (``--output-fps`` sets it). Returns the clip's rate.
  """
  with np.load(motion_file) as data:
    if "fps" not in data:
      raise ValueError(
        f"{motion_file} has no 'fps' key, so its frame rate cannot be checked. "
        "Regenerate it with clipper (--format mjlab)."
      )
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  expected = 1.0 / step_dt
  if abs(fps - expected) > 1e-6 * expected:
    raise ValueError(
      f"{motion_file} is a {fps:g} Hz clip, but this task steps at {expected:g} Hz "
      f"(step_dt {step_dt:g} s). Re-export it with clipper's --output-fps "
      f"{expected:g}."
    )
  return fps


class OslMotionCommand(MotionCommand):
  """Pad the motion to the entity's joints and keep resets on the manifold.

  One 73-joint clip serves every variant: all have the same 105 bodies in the same
  order, and the clip's joints appear in each in the same relative order. The joints
  a clip does not name -- the compliant socket's -- are padded with zeros, their
  neutral position. The clip's frame rate must equal the control rate.

  Stock sampling perturbs all coordinates; dependent samples are discarded here.
  This shared write hook also covers wraparound and deterministic viewer resets.
  Reference arrays are never modified after padding, and dependent limits remain
  soft.
  """

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    check_motion_fps(cfg.motion_file, env.step_dt)
    super().__init__(cfg, env)
    self._pad_motion_joints()
    model = env.sim.mj_model
    names = self.robot.joint_names
    # Socket joints are passive, and stock RSI's joint noise would otherwise slide
    # socket_piston by up to 10 cm. Resets put them at rest at neutral instead.
    self.socket_ids = [names.index(n) for n in SOCKET_JOINTS if n in names]
    pending = {}
    prefix = cfg.entity_name + "/"
    for i in range(model.neq):
      if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
        continue
      dependent_name = model.joint(model.eq_obj1id[i]).name
      if not dependent_name.startswith(prefix):
        continue
      dependent = names.index(dependent_name.removeprefix(prefix))
      source = (
        names.index(model.joint(model.eq_obj2id[i]).name.removeprefix(prefix))
        if model.eq_obj2id[i] >= 0
        else -1
      )
      pending[dependent] = (source, tuple(model.eq_data[i, :5]))
    self.independent = [i for i in range(len(names)) if i not in pending]
    self.couplings = []
    resolved = set(self.independent) | {-1}
    while pending:
      ready = [i for i, (source, _) in pending.items() if source in resolved]
      if not ready:
        raise ValueError("Cyclic OSL joint equality constraints")
      for dependent in ready:
        source, coefficients = pending.pop(dependent)
        self.couplings.append((dependent, source, coefficients))
        resolved.add(dependent)

  def _pad_motion_joints(self) -> None:
    names = list(self.robot.joint_names)
    width = self.motion.joint_pos.shape[1]
    if width == len(names):
      return
    clip_names = _clip_joint_names()
    if width != len(clip_names):
      raise ValueError(
        f"Motion has {width} joints; expected {len(names)} (this entity) or "
        f"{len(clip_names)} (the rigid robot)."
      )
    columns = [names.index(n) for n in clip_names]
    for attr in ("joint_pos", "joint_vel"):
      clip = getattr(self.motion, attr)
      padded = clip.new_zeros(clip.shape[0], len(names))
      padded[:, columns] = clip
      setattr(self.motion, attr, padded)

  def _write_reference_state_to_sim(
    self,
    env_ids,
    root_pos,
    root_ori,
    root_lin_vel,
    root_ang_vel,
    joint_pos,
    joint_vel,
  ) -> None:
    joint_pos = joint_pos.clone()
    joint_vel = joint_vel.clone()
    ids = self.independent
    limits = self.robot.data.soft_joint_pos_limits[env_ids][:, ids]
    joint_pos[:, ids] = torch.clip(joint_pos[:, ids], limits[..., 0], limits[..., 1])
    joint_pos[:, self.socket_ids] = 0.0
    joint_vel[:, self.socket_ids] = 0.0
    for dependent, source, c in self.couplings:
      if source == -1:
        # MuJoCo omits joint2 for a coordinate constrained to a constant.
        joint_pos[:, dependent] = c[0]
        joint_vel[:, dependent] = 0.0
        continue
      q = joint_pos[:, source]
      joint_pos[:, dependent] = c[0] + q * (c[1] + q * (c[2] + q * (c[3] + q * c[4])))
      joint_vel[:, dependent] = joint_vel[:, source] * (
        c[1] + q * (2 * c[2] + q * (3 * c[3] + q * 4 * c[4]))
      )
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)


@dataclass(kw_only=True)
class OslMotionCommandCfg(MotionCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> OslMotionCommand:
    return OslMotionCommand(self, env)


def bad_osl_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  gravity = torch.nn.functional.normalize(
    env.scene[asset_cfg.name].data.gravity_vec_w, dim=-1
  )
  reference = quat_apply_inverse(command.anchor_quat_w, gravity)
  robot = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
  return 1.0 - (reference * robot).sum(dim=-1) > threshold
