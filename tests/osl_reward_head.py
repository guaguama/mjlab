"""Frozen copies of the OSL imitation reward terms at commit 2d1e12a89, before the
CLF-RL decrease condition and the regularizers (2026-11-18): ``mimic_reward``
(stage 1) as :class:`HeadMimicReward` and ``split_mimic_reward`` (stage 2) as
:class:`HeadSplitMimicReward`, with HEAD's ``relative_site_quantities``.

Verbatim but for the renames, the module header and these classes' shared config
dataclasses (``ClfCfg``, ``SplitRewardCfg``), imported from the current code: they
only hold the values, and HEAD reads only the fields it had. Do not edit: the
bit-identity tests (``test_osl_reward_clfrl.py``) hold the current terms, run
with the old flags, to these.
"""
# pyright: reportPossiblyUnboundVariable=false
# ruff: noqa: N801

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OSL_TORQUE_JOINTS
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.osl_mimic.mdp.kinematics import (
  di_care_block,
  exact_decay_rate,
  quat_angle,
  relative_site_velocities,
  rotvec_from_matrix,
  v_max_unit,
)
from mjlab.tasks.osl_mimic.mdp.reference import MimicReference, MimicReferenceSource
from mjlab.tasks.osl_mimic.mdp.rewards import ClfCfg
from mjlab.tasks.osl_split.config.sensors import LEG_CONTACT_SENSOR
from mjlab.tasks.osl_split.mdp.rewards import SplitRewardCfg
from mjlab.tasks.osl_split.rl.layout import AGENTS, ActionLayout
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _head_relative_site_quantities(
  pos: torch.Tensor,
  mat: torch.Tensor,
  ang_vel: torch.Tensor,
  lin_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """``calculate_relative_site_quantities`` (``loco_mujoco/core/utils/math.py``).

  Args:
    pos: (..., S, 3) world site positions. Site 0 is the main site (the pelvis).
    mat: (..., S, 3, 3) world site orientations.
    ang_vel: (..., S, 3) world angular velocity of each site.
    lin_vel: (..., S, 3) world linear velocity of each site's origin. Together these
      are ``calc_site_velocities(..., flg_local=False)``.

  Returns:
    rpos: (..., K, 3) ``p_i - p_main``, in the world frame (K = S - 1).
    rangles: (..., K, 3) ``rotvec(R_main^T R_i)``.
    rvel: (..., K, 6) ``[R_rel_i^T w_i - w_main, R_main (v_main - v_i)]``, from
      ``calculate_relative_velocity_in_local_frame``. This is not the time
      derivative of ``rpos`` (it is negated and rotated by ``R_main``, not its
      transpose). MimicReward and the goal observation use it as is; the CLF does
      not (see :func:`relative_site_velocities`).
  """
  rpos = pos[..., 1:, :] - pos[..., :1, :]
  main_mat = mat[..., :1, :, :]
  rel_rot = main_mat.transpose(-1, -2) @ mat[..., 1:, :, :]
  rangles = rotvec_from_matrix(rel_rot)
  ang = (rel_rot.transpose(-1, -2) @ ang_vel[..., 1:, :, None]).squeeze(-1)
  ang = ang - ang_vel[..., :1, :]
  lin = (main_mat @ (lin_vel[..., :1, :] - lin_vel[..., 1:, :])[..., None]).squeeze(-1)
  return rpos, rangles, torch.cat([ang, lin], dim=-1)


# MimicReward's kernel sharpness (``*_w_exp``) defaults; no OSL config overrides them.
MIMIC_EXPS: dict[str, float] = {
  "qpos": 10.0,
  "qvel": 2.0,
  "root_pos": 10.0,
  "root_vel": 10.0,
  "rpos": 100.0,
  "rquat": 10.0,
  "rvel": 0.1,
}

MIMIC_TERMS = ("qpos", "qvel", "root_pos", "root_vel", "rpos", "rquat", "rvel")

CLF_GROUPS = ("rpos", "rangles", "root_pos", "root_ori")


def _head_yaw(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class HeadMimicReward(ManagerTermBase):
  """musclemimic's imitation reward, clipped to be non-negative.

  Params:
    command_name: The reference command (a :class:`MimicReferenceSource`).
    weights: The seven ``*_w_sum`` weights, all required. ``rvel`` weighs the
      rotational and the linear relative-velocity kernels alike.
    action_out_of_bounds_coeff: musclemimic's default 0.01, active in every config.
    joint_torque_weights: ``{joint: w}`` adding ``-w * qfrc_actuator**2``.
    clf: A :class:`ClfCfg` to add the CLF terms, or None for plain MimicReward.

  Diagnostics for the last step are kept in :attr:`components`, (B,) each.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    params = cfg.params
    missing = set(MIMIC_TERMS) - set(params["weights"])
    if missing:
      raise ValueError(f"mimic_reward needs every weight; missing {sorted(missing)}")
    self._weights = {k: float(params["weights"][k]) for k in MIMIC_TERMS}
    self._oob_coeff = float(params.get("action_out_of_bounds_coeff", 0.01))

    cmd = cast(
      MimicReferenceSource, env.command_manager.get_term(params["command_name"])
    )
    names = list(cmd.robot.joint_names)
    torque = params.get("joint_torque_weights") or {}
    v_adr = cmd.robot.indexing.joint_v_adr
    self._torque_dofs = torch.tensor(
      [int(v_adr[names.index(j)]) for j in torque], dtype=torch.long, device=env.device
    )
    self._torque_weights = torch.tensor(list(torque.values()), device=env.device)

    self.components: dict[str, torch.Tensor] = {}
    # The last call's intermediates, for subclasses that recombine them (the stage-2
    # split reward, ``osl_split.mdp.rewards``). Read-only, and valid only until the
    # env steps or resets again: "action" is the action manager's live buffer.
    self._cache: dict[str, torch.Tensor | None] = {}
    self._clf_cache: dict[str, torch.Tensor] = {}

    self._clf: ClfCfg | None = params.get("clf")
    if self._clf is not None:
      self._setup_clf(self._clf, cmd, env)

  def _setup_clf(
    self, clf: ClfCfg, cmd: MimicReferenceSource, env: ManagerBasedRlEnv
  ) -> None:
    unknown = set(clf.scales) - set(CLF_GROUPS)
    if unknown:
      raise ValueError(f"Unknown CLF groups {sorted(unknown)}; valid: {CLF_GROUPS}")
    if clf.tau <= 0.0 or clf.c_viol <= 0.0:
      raise ValueError("CLF tau and c_viol must be positive.")
    self._p11, self._p12, self._p22 = di_care_block(clf.q_pos, clf.q_vel, clf.r)
    self._v_max = v_max_unit(self._p11, self._p12, self._p22) + 1e-8
    self._eps_dead = clf.eps_dead * self._v_max
    self._viol_norm = clf.c_viol * self._v_max
    self._alpha = (
      clf.alpha_safety * exact_decay_rate(clf.q_pos, clf.q_vel, clf.r)
      if clf.alpha is None
      else float(clf.alpha)
    )
    self._clf_dt = float(env.step_dt)
    n_rel = len(cmd.site_body_ids) - 1
    sizes = {"rpos": 3 * n_rel, "rangles": 3 * n_rel, "root_pos": 3, "root_ori": 3}
    self._clf_groups = [g for g in CLF_GROUPS if g in clf.scales]
    scales = [torch.full((sizes[g],), clf.scales[g]) for g in self._clf_groups]
    self._clf_scales = torch.cat(scales).to(env.device)
    n = self._clf_scales.numel()
    self._clf_n = n
    self._ed_prev = torch.zeros(env.num_envs, n, device=env.device)
    self._clf_valid = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    weights: dict[str, float],
    action_out_of_bounds_coeff: float = 0.01,
    joint_torque_weights: dict[str, float] | None = None,
    clf: ClfCfg | None = None,
  ) -> torch.Tensor:
    del weights, action_out_of_bounds_coeff, joint_torque_weights, clf
    cmd = cast(MimicReferenceSource, env.command_manager.get_term(command_name))
    robot = cmd.robot
    ref = cmd.reference(1)
    w = self._weights

    root_q = cmd.root_qpos
    root_v = cmd.root_qvel
    joint_q = robot.data.joint_pos
    joint_v = robot.data.joint_vel
    ref_root_pos = ref.root_pos_w
    ref_root_quat = ref.root_quat
    ref_joint_q = ref.joint_pos
    ref_joint_v = ref.joint_vel
    ref_root_v = ref.root_qvel

    # qpos: every non-quaternion coordinate (root xyz and all joints), plus the root
    # quaternion's geodesic angle -- added unsquared, as musclemimic does.
    qpos_err = torch.cat([root_q[:, :3] - ref_root_pos, joint_q - ref_joint_q], -1)
    root_angle = quat_angle(root_q[:, 3:7], ref_root_quat)
    qpos_dist = qpos_err.square().mean(-1) + root_angle
    qvel_err = torch.cat([root_v - ref_root_v, joint_v - ref_joint_v], -1)
    qvel_dist = qvel_err.square().mean(-1)
    root_pos_err = root_q[:, :3] - ref_root_pos
    root_pos_dist = root_pos_err.square().mean(-1)

    # Root velocity: linear in the root frame, angular as stored (body frame).
    sim_root_vel = torch.cat(
      [quat_apply_inverse(root_q[:, 3:7], root_v[:, :3]), root_v[:, 3:]], -1
    )
    ref_root_vel = torch.cat(
      [quat_apply_inverse(ref_root_quat, ref_root_v[:, :3]), ref_root_v[:, 3:]], -1
    )
    root_vel_err = sim_root_vel - ref_root_vel
    root_vel_dist = root_vel_err.square().mean(-1)

    site_pos, site_mat, site_ang, site_lin = cmd.site_state()
    rpos, rangles, rvel = _head_relative_site_quantities(
      site_pos, site_mat, site_ang, site_lin
    )
    ref_rpos, ref_rangles, ref_rvel = ref.rpos, ref.rangles, ref.rvel
    rpos_err = rpos - ref_rpos
    rangles_err = rangles - ref_rangles
    rpos_dist = rpos_err.square().mean((-1, -2))
    rangles_dist = rangles_err.square().mean((-1, -2))
    rvel_rot_dist = (rvel[..., :3] - ref_rvel[..., :3]).square().mean((-1, -2))
    rvel_lin_dist = (rvel[..., 3:] - ref_rvel[..., 3:]).square().mean((-1, -2))

    k = MIMIC_EXPS
    r_qpos = torch.exp(-k["qpos"] * qpos_dist)
    r_qvel = torch.exp(-k["qvel"] * qvel_dist)
    r_root_pos = torch.exp(-k["root_pos"] * root_pos_dist)
    r_root_vel = torch.exp(-k["root_vel"] * root_vel_dist)
    r_rpos = torch.exp(-k["rpos"] * rpos_dist)
    r_rquat = torch.exp(-k["rquat"] * rangles_dist)
    r_rvel_rot = torch.exp(-k["rvel"] * rvel_rot_dist)
    r_rvel_lin = torch.exp(-k["rvel"] * rvel_lin_dist)
    total = (
      w["qpos"] * r_qpos
      + w["qvel"] * r_qvel
      + w["root_pos"] * r_root_pos
      + w["root_vel"] * r_root_vel
      + w["rpos"] * r_rpos
      + w["rquat"] * r_rquat
      + w["rvel"] * (r_rvel_rot + r_rvel_lin)
    )

    # Penalties, floored at -1 together (``out_of_bounds_action_cost``: squared
    # excess over [-1, 1], summed and divided by the action dimension).
    action = env.action_manager.action
    excess = (-1.0 - action).clamp(min=0.0) + (action - 1.0).clamp(min=0.0)
    penalty = -self._oob_coeff * excess.square().sum(-1) / action.shape[-1]
    qfrc = None
    if self._torque_dofs.numel():
      qfrc = env.sim.data.qfrc_actuator[:, self._torque_dofs]
      penalty = penalty - (self._torque_weights * qfrc.square()).sum(-1)
    penalty = penalty.clamp(min=-1.0)
    floored_penalty = penalty

    clf_terms: dict[str, torch.Tensor] = {}
    if self._clf is not None:
      clf_terms = self._clf_terms(env, cmd, ref, root_q, root_v, rpos, rangles)
      total = total + self._clf.w * clf_terms["level"]
      # Outside the -1 floor, so a saturated penalty cannot hide it.
      penalty = penalty - self._clf.pen_w * clf_terms["violation"]

    preclip = total + penalty
    reward = torch.nan_to_num(preclip.clamp(min=0.0), nan=0.0)

    joint_err = joint_q - ref_joint_q
    self.components = {
      "reward_qpos": r_qpos,
      "reward_qvel": r_qvel,
      "reward_root_pos": r_root_pos,
      "reward_root_vel": r_root_vel,
      "reward_rpos": r_rpos,
      "reward_rquat": r_rquat,
      "reward_rvel_rot": r_rvel_rot,
      "reward_rvel_lin": r_rvel_lin,
      "penalty_total": penalty,
      "reward_preclip": preclip,
      "reward_clipped": (preclip < 0.0).float(),
      "err_root_xyz": root_pos_dist.sqrt(),
      "err_root_yaw": torch.atan2(
        torch.sin(_head_yaw(root_q[:, 3:7]) - _head_yaw(ref_root_quat)),
        torch.cos(_head_yaw(root_q[:, 3:7]) - _head_yaw(ref_root_quat)),
      ).abs(),
      "err_joint_pos": joint_err.square().mean(-1).sqrt(),
      "err_joint_vel": (joint_v - ref_joint_v).square().mean(-1).sqrt(),
      "err_site_abs": (site_pos - ref.site_pos_w).norm(dim=-1).mean(-1),
      "err_rpos": rpos_dist.sqrt(),
      "dist_joint_pos": joint_err.abs().mean(-1),
      "dist_joint_vel": (joint_v - ref_joint_v).abs().mean(-1),
      "dist_rel_site_pos": (rpos - ref_rpos).norm(dim=-1).mean(-1),
      "dist_rel_site_ori": (rangles - ref_rangles).norm(dim=-1).mean(-1),
      "dist_rel_site_vel": (rvel - ref_rvel).norm(dim=-1).mean(-1),
    }
    self.components.update({f"clf_{k}": v for k, v in clf_terms.items()})
    self._cache = {
      "qpos_err": qpos_err,
      "quat_angle": root_angle,
      "qvel_err": qvel_err,
      "root_pos_err": root_pos_err,
      "root_vel_err": root_vel_err,
      "rpos_err": rpos_err,
      "rangles_err": rangles_err,
      "rvel_err": rvel - ref_rvel,
      "action": action,
      "qfrc": qfrc,
      "penalty": floored_penalty,  # Before the CLF violation is subtracted.
      "joint_q_err": joint_err,
      "joint_v_err": qvel_err[:, 6:],
      "r_qpos": r_qpos,
      "r_qvel": r_qvel,
      "r_root_pos": r_root_pos,
      "r_root_vel": r_root_vel,
      "r_rpos": r_rpos,
      "r_rquat": r_rquat,
      "r_rvel_rot": r_rvel_rot,
      "r_rvel_lin": r_rvel_lin,
    }
    return reward

  def _clf_terms(
    self,
    env: ManagerBasedRlEnv,
    cmd: MimicReferenceSource,
    ref: MimicReference,
    root_q: torch.Tensor,
    root_v: torch.Tensor,
    rpos: torch.Tensor,
    rangles: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    assert self._clf is not None
    batch = rpos.shape[0]
    es, eds = [], []
    if "rpos" in self._clf_groups or "rangles" in self._clf_groups:
      _, site_mat, site_ang, site_lin = cmd.site_state()
      v_rel, omega_rel = relative_site_velocities(site_mat, site_ang, site_lin)
    for group in self._clf_groups:
      if group == "rpos":
        es.append((rpos - ref.rpos).reshape(batch, -1))
        eds.append((v_rel - ref.v_rel).reshape(batch, -1))
      elif group == "rangles":
        es.append((rangles - ref.rangles).reshape(batch, -1))
        eds.append((omega_rel - ref.omega_rel).reshape(batch, -1))
      elif group == "root_pos":
        # Free-joint linear qvel and the reference's are both world-frame, so the
        # error and its rate pair exactly.
        es.append(root_q[:, :3] - ref.root_pos_w)
        eds.append(root_v[:, :3] - ref.root_lin_vel_w)
      elif group == "root_ori":
        m_sim = matrix_from_quat(root_q[:, 3:7])
        m_ref = matrix_from_quat(ref.root_quat)
        m_rel = m_ref.transpose(-1, -2) @ m_sim
        es.append(rotvec_from_matrix(m_rel))
        eds.append((m_rel @ root_v[:, 3:, None]).squeeze(-1) - ref.root_ang_vel_b)
    tau = self._clf.tau
    e = torch.cat(es, -1) / self._clf_scales
    ed = torch.cat(eds, -1) * (tau / self._clf_scales)

    p11, p12, p22, inv_n = self._p11, self._p12, self._p22, 1.0 / self._clf_n
    cross = (2.0 * p12 * inv_n) * (e * ed).sum(-1)
    v = (p11 * inv_n) * e.square().sum(-1) + cross + (p22 * inv_n) * ed.square().sum(-1)
    v = v.clamp(min=0.0)

    # First step of an episode, or of a reference that just jumped: no previous error
    # velocity to difference against.
    first = (env.episode_length_buf <= 1) | cmd.reference_jumped
    edd = torch.where(
      first[:, None], torch.zeros_like(ed), (ed - self._ed_prev) * (tau / self._clf_dt)
    )
    vdot = (2.0 * inv_n) * (
      (ed * (p11 * e + p12 * ed)).sum(-1) + (edd * (p12 * e + p22 * ed)).sum(-1)
    )
    violation = ((vdot + self._alpha * v - self._eps_dead) / self._viol_norm).clamp(
      0, 1
    )
    violation = violation * (~first).float() * self._clf_valid

    # A non-finite V maps to the worst level term, never the best.
    v_norm = v / self._v_max
    level = torch.where(
      torch.isfinite(v_norm), torch.exp(-v_norm.clamp(max=5.0)), torch.zeros_like(v)
    )

    self._clf_cache = {
      "e": e,
      "ed": ed,
      "edd": edd,
      "first": first,
      "valid": self._clf_valid.clone(),
    }
    self._ed_prev = ed.detach().clone()
    self._clf_valid.fill_(1.0)
    return {
      "V": v,
      "V_norm": v_norm,
      "vdot": vdot,
      "cross": cross,
      "violation": violation,
      "level": level,
    }


CHANNELS: tuple[str, str] = ("human", "osl")
"""The reward channels, in :attr:`split_mimic_reward.channel_rewards` column order:
the agents' (``rl.layout.AGENTS``), so the runner can index rewards by agent."""
if CHANNELS != AGENTS:
  raise ImportError(f"Reward channels {CHANNELS} must follow the agents {AGENTS}.")
GROUPS: tuple[str, str, str] = ("human", "osl_sites", "osl_joints")
"""The coordinate groups a channel weighs."""
OSL_SITE_PREFIX = "OSL_KA_L1_"
"""Mimic sites on bodies with this prefix belong to ``osl_sites``."""

SITE_KERNELS = ("rpos", "rquat", "rvel_rot", "rvel_lin")
"""MimicReward's site kernels: per-site means over the axes, then the weighted mean
over the sites."""
JOINT_KERNELS = ("qpos", "qvel")
"""MimicReward's joint kernels: the root's coordinates (human), then every joint."""
FIXED_KERNELS = ("root_pos", "root_vel")
"""MimicReward's human-only kernels: they keep their stage-1 values in both
channels, whatever the weights."""

SPLIT_CLF_DIAGNOSTICS: tuple[str, ...] = (
  "split_V_norm_human",
  "split_V_norm_osl",
  "split_level_human",
  "split_level_osl",
  "split_violation_human",
  "split_violation_osl",
  "split_qform_human",
  "split_qform_osl_sites",
  "split_qform_osl_joints",
)
"""The CLF-only diagnostics. ``split_qform_<group>`` is the group's mean quadratic
form per coordinate in V_max units: the V_norm it would have alone."""
SPLIT_DIAGNOSTICS: tuple[str, ...] = (
  "split_reward_human",
  "split_reward_osl",
  "split_preclip_human",
  "split_preclip_osl",
  "split_clipped_human",
  "split_clipped_osl",
  "split_penalty_human",
  "split_penalty_osl",
  "split_err_osl_knee",
  "split_err_osl_ankle",
  *SPLIT_CLF_DIAGNOSTICS,
)
"""Every key :class:`split_mimic_reward` adds to ``components``, (B,) each, read by
``osl_mimic.mdp.mimic_component``. The penalties include the CLF violation, as
stage 1's ``penalty_total`` does; ``split_err_osl_*`` are ``|q - q_ref|`` in rad."""
SPLIT_CONTACT_DIAGNOSTICS: tuple[str, ...] = (
  "split_leg_contact",
  "split_leg_contact_force",
)
"""The leg-crossing diagnostics, added to ``components`` when the scene has the
:data:`~..config.sensors.LEG_CONTACT_SENSOR`: ``split_leg_contact`` is 1 where an OSL
collision geom touched the biological left leg in any physics substep of the step
(its mean is a contact rate), ``split_leg_contact_force`` the largest net contact
force over those substeps (N)."""

Weights = dict[str, dict[str, float]]


def stage1_channel_weights(clf: bool) -> Weights:
  """Both channels at the weights that reproduce the stage-1 imitation reward.

  Stage 1's CLF has no joint coordinates (``osl_joints`` 0); MimicReward's joint
  kernels hold the OSL joints like every other joint (1). A reward with both the
  CLF and non-zero MimicReward weights has no exact stage-1 equivalent, since one
  ``osl_joints`` weight serves both; ``clf=True`` keeps its CLF exact.
  """
  group = {"human": 1.0, "osl_sites": 1.0, "osl_joints": 0.0 if clf else 1.0}
  return {channel: dict(group) for channel in CHANNELS}


def _index(values: list[int], device: str) -> torch.Tensor:
  return torch.tensor(values, dtype=torch.long, device=device)


def _group_sums(
  values: torch.Tensor, index: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
  """Per group, the sum of ``values`` (B, n) over the group's coordinates.

  Index-select, not a masked product (``0 * inf`` is NaN) or a matmul (mjlab
  enables TF32).
  """
  return {group: values.index_select(-1, idx).sum(-1) for group, idx in index.items()}


class HeadSplitMimicReward(HeadMimicReward):
  """:class:`~mjlab.tasks.osl_mimic.mdp.rewards.mimic_reward` plus two reward
  channels.

  Returns the stage-1 reward unchanged, so it needs ``weight=1`` and
  ``scale_rewards_by_dt=False`` as the stage-1 term does.

  Params: every ``mimic_reward`` param, plus
    channels: A :class:`SplitRewardCfg` (or its fields as a dict); None takes the
      defaults.

  After every call:
    channel_rewards: (B, 2) the imitation reward of each channel (``CHANNELS``
      order), >= 0, a new tensor per call. A reset never touches it: the runner
      reads it after the env has auto-reset the envs that finished.
    components: the stage-1 diagnostics plus ``SPLIT_DIAGNOSTICS``.
  """

  channel_rewards: torch.Tensor

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    params = cfg.params
    channels = params.get("channels")
    if channels is None:
      channels = SplitRewardCfg()
    elif isinstance(channels, Mapping):
      channels = SplitRewardCfg(**channels)
    if not isinstance(channels, SplitRewardCfg):
      raise TypeError(f"channels must be a SplitRewardCfg, got {type(channels)}")
    if not channels.osl_joint_scale > 0.0:
      raise ValueError(f"osl_joint_scale must be > 0, got {channels.osl_joint_scale}")
    self._split = channels
    device = env.device
    cmd = cast(
      MimicReferenceSource, env.command_manager.get_term(params["command_name"])
    )

    # Mimic sites: the first is the main site; the relative sites follow, in order.
    site_bodies = tuple(getattr(getattr(cmd, "cfg", None), "site_bodies", ()))
    if len(site_bodies) != len(cmd.site_body_ids) or len(site_bodies) < 2:
      raise ValueError(
        "The split reward needs the command's cfg.site_bodies, one per site id; "
        f"got {len(site_bodies)} names for {len(cmd.site_body_ids)} sites."
      )
    relative = site_bodies[1:]
    osl_sites = [i for i, b in enumerate(relative) if b.startswith(OSL_SITE_PREFIX)]
    if len(osl_sites) != 3:
      raise ValueError(f"Expected 3 OSL mimic sites, got {osl_sites} of {relative}")
    human_sites = [i for i in range(len(relative)) if i not in osl_sites]
    self._site_index = {
      "human": _index(human_sites, device),
      "osl_sites": _index(osl_sites, device),
    }

    # Joints: MimicReward's qpos coordinates are root xyz then the joints; its qvel
    # coordinates are the root's six then the joints.
    names = list(cmd.robot.joint_names)
    missing = [j for j in OSL_TORQUE_JOINTS if j not in names]
    if missing:
      raise ValueError(f"The robot has no OSL joints {missing}.")
    osl_joints = [names.index(j) for j in OSL_TORQUE_JOINTS]
    human_joints = [j for j in range(len(names)) if j not in osl_joints]
    self._osl_joint_ids = _index(osl_joints, device)
    self._qpos_index = {
      "human": _index([0, 1, 2] + [3 + j for j in human_joints], device),
      "osl_joints": _index([3 + j for j in osl_joints], device),
    }
    self._qvel_index = {
      "human": _index(list(range(6)) + [6 + j for j in human_joints], device),
      "osl_joints": _index([6 + j for j in osl_joints], device),
    }

    # Penalty owners: action rows by term, torque-penalized joints by name.
    layout = ActionLayout.from_action_manager(env.action_manager)
    self._action_rows = {agent: layout.index(agent, device) for agent in AGENTS}
    self._action_dims = {agent: layout.dim(agent) for agent in AGENTS}
    torque = list(params.get("joint_torque_weights") or {})
    self._torque_index = {
      "human": _index(
        [i for i, j in enumerate(torque) if j not in OSL_TORQUE_JOINTS], device
      ),
      "osl": _index(
        [i for i, j in enumerate(torque) if j in OSL_TORQUE_JOINTS], device
      ),
    }

    # Coordinates per group in each kernel that mixes groups.
    n_sites = {"human": len(human_sites), "osl_sites": len(osl_sites), "osl_joints": 0}
    n_human, n_osl = len(human_joints), len(osl_joints)
    self._kernel_counts: dict[str, dict[str, int]] = {
      "qpos": {"human": 3 + n_human, "osl_sites": 0, "osl_joints": n_osl},
      "qvel": {"human": 6 + n_human, "osl_sites": 0, "osl_joints": n_osl},
      **{kernel: dict(n_sites) for kernel in SITE_KERNELS},
    }
    w = self._weights
    self._kernel_weights: dict[str, float] = {
      "qpos": w["qpos"],
      "qvel": w["qvel"],
      "root_pos": w["root_pos"],
      "root_vel": w["root_vel"],
      "rpos": w["rpos"],
      "rquat": w["rquat"],
      "rvel_rot": w["rvel"],
      "rvel_lin": w["rvel"],
    }
    self._mimic_active = any(v != 0.0 for v in w.values())

    if self._clf is not None:
      self._setup_clf_split(self._clf, len(relative), osl_sites, env)

    self.channel_rewards = torch.zeros(env.num_envs, len(CHANNELS), device=device)
    self._leg_contact = _scene_sensor(env, LEG_CONTACT_SENSOR)
    self._group_weights: Weights = {}
    initial = channels.initial_weights
    self.set_group_weights(
      initial if initial is not None else stage1_channel_weights(self._clf is not None)
    )

  def _setup_clf_split(
    self, clf: ClfCfg, n_rel: int, osl_sites: list[int], env: ManagerBasedRlEnv
  ) -> None:
    # The CLF's coordinates are its groups' errors in CLF_GROUPS order: per relative
    # site three (site-major) for rpos and rangles, then three each for the root.
    owner: list[str] = []
    for group in self._clf_groups:
      if group in ("rpos", "rangles"):
        for site in range(n_rel):
          owner += ["osl_sites" if site in osl_sites else "human"] * 3
      else:
        owner += ["human"] * 3
    assert len(owner) == self._clf_n and set(self._clf_groups) <= set(CLF_GROUPS)
    self._clf_index = {
      group: _index([i for i, o in enumerate(owner) if o == group], env.device)
      for group in ("human", "osl_sites")
    }
    self._kernel_counts["clf"] = {
      "human": owner.count("human"),
      "osl_sites": owner.count("osl_sites"),
      "osl_joints": len(OSL_TORQUE_JOINTS),
    }
    self._kernel_weights["clf"] = clf.w
    self._edj_prev = torch.zeros(
      env.num_envs, len(OSL_TORQUE_JOINTS), device=env.device
    )
    self._clf_joint: dict[str, torch.Tensor] = {}

  ##
  # Group weights.
  ##

  @property
  def group_weights(self) -> Weights:
    """``{channel: {group: weight}}``, a copy."""
    return copy.deepcopy(self._group_weights)

  def set_group_weights(self, weights: Mapping[str, Mapping[str, float]]) -> None:
    """Replace both channels' group weights (the runner does every iteration).

    Every channel and every group must be given, each weight finite and >= 0, and
    no kernel in use may be left with all-zero weights (:meth:`check_group_weights`).
    A refused update keeps the current weights.
    """
    self._group_weights = self.check_group_weights(weights)

  def check_group_weights(self, weights: Mapping[str, Mapping[str, float]]) -> Weights:
    """``weights`` as :meth:`set_group_weights` would store them, without storing
    them; raises ``ValueError`` where it would.

    Which kernels are in use depends on the reward's own weights, so only the term
    can tell whether a channel leaves one of them without coordinates (MimicReward's
    site kernels, say, with only ``osl_joints`` weighted). The runner checks the
    channel schedule's ends with it before training, not when the anneal gets there.
    """
    if set(weights) != set(CHANNELS):
      raise ValueError(f"Need weights for channels {CHANNELS}, got {sorted(weights)}")
    checked: Weights = {}
    for channel in CHANNELS:
      groups = weights[channel]
      if set(groups) != set(GROUPS):
        raise ValueError(
          f"Channel {channel!r} needs weights for groups {GROUPS}, got {sorted(groups)}"
        )
      values = {group: float(groups[group]) for group in GROUPS}
      bad = {g: v for g, v in values.items() if not (math.isfinite(v) and v >= 0.0)}
      if bad:
        raise ValueError(f"Channel {channel!r}: weights must be finite and >= 0: {bad}")
      for kernel in self._mixed_kernels():
        counts = self._kernel_counts[kernel]
        if sum(values[g] * counts[g] for g in GROUPS) <= 0.0:
          raise ValueError(
            f"Channel {channel!r} weighs every coordinate of the {kernel!r} kernel "
            f"at 0 ({values}); its weighted mean is undefined."
          )
      checked[channel] = values
    return checked

  def _mixed_kernels(self) -> list[str]:
    """The kernels in use that take a weighted mean over groups."""
    kernels = ["clf"] if self._clf is not None else []
    kernels += [
      k for k in (*JOINT_KERNELS, *SITE_KERNELS) if self._kernel_weights[k] != 0.0
    ]
    return kernels

  def kernel_group_counts(self) -> dict[str, dict[str, int]]:
    """Per kernel in use that mixes groups, its coordinates per group. A site counts
    once in the site kernels, which average its three axes first."""
    return {k: dict(self._kernel_counts[k]) for k in self._mixed_kernels()}

  def group_coordinate_counts(self) -> dict[str, int]:
    """The coordinates each group's weight multiplies: under the CLF alone, those of
    the channel V (84 human, 18 OSL-site, 2 OSL-joint on the OSL models). With
    MimicReward weights, summed over every kernel in use."""
    counts = dict.fromkeys(GROUPS, 0)
    for per_group in self.kernel_group_counts().values():
      for group, n in per_group.items():
        counts[group] += n
    return counts

  def effective_shares(self) -> Weights:
    """Per channel, each group's share of the channel's imitation reward.

    Each kernel's weight is split among the groups by ``weight x coordinates /
    sum`` inside it, and the shares are averaged over the kernels by their weights.
    Under the CLF alone this is exactly ``weight x count / sum`` over V's
    coordinates. The human-only kernels (root position and velocity) count fully
    as human; the qpos kernel's root-angle term is not counted.
    """
    kernels = {k: self._kernel_weights[k] for k in self._mixed_kernels()}
    if self._clf is not None and self._clf.w == 0.0:
      kernels.pop("clf")
    fixed = {k: self._kernel_weights[k] for k in FIXED_KERNELS}
    total = sum(kernels.values()) + sum(fixed.values())
    shares: Weights = {}
    for channel in CHANNELS:
      w = self._group_weights[channel]
      share = dict.fromkeys(GROUPS, 0.0)
      share["human"] += sum(fixed.values())
      for kernel, kernel_weight in kernels.items():
        counts = self._kernel_counts[kernel]
        den = sum(w[g] * counts[g] for g in GROUPS)
        for group in GROUPS:
          share[group] += kernel_weight * w[group] * counts[group] / den
      shares[channel] = {
        g: (s / total if total > 0.0 else 0.0) for g, s in share.items()
      }
    return shares

  ##
  # The reward.
  ##

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    weights: dict[str, float],
    action_out_of_bounds_coeff: float = 0.01,
    joint_torque_weights: dict[str, float] | None = None,
    clf: ClfCfg | None = None,
    channels: SplitRewardCfg | dict[str, Any] | None = None,
  ) -> torch.Tensor:
    del channels  # Read once, in __init__.
    team = super().__call__(
      env,
      command_name,
      weights,
      action_out_of_bounds_coeff,
      joint_torque_weights,
      clf,
    )
    self._compute_channels(team)
    return team

  def _stash(self, key: str) -> torch.Tensor:
    value = self._cache[key]
    if value is None:
      raise KeyError(f"mimic_reward stashed no {key!r} this step")
    return value

  def _weighted_mean(
    self, kernel: str, sums: Mapping[str, torch.Tensor], weights: Mapping[str, float]
  ) -> torch.Tensor:
    """``sum_g w_g S_g / sum_g w_g n_g`` over the groups the channel weighs. Groups
    at weight 0 are left out entirely, so a non-finite error there cannot leak."""
    counts = self._kernel_counts[kernel]
    num: torch.Tensor | None = None
    den = 0.0
    for group, total in sums.items():
      w, n = weights[group], counts[group]
      if w == 0.0 or n == 0:
        continue
      term = total * w
      num = term if num is None else num + term
      den += w * n
    if num is None or den <= 0.0:  # Prevented by set_group_weights.
      raise RuntimeError(f"No weighted coordinates in the {kernel!r} kernel.")
    return num / den

  def _compute_channels(self, team: torch.Tensor) -> None:
    totals = {c: torch.zeros_like(team) for c in CHANNELS}
    if self._mimic_active:
      sums = self._mimic_group_sums()
      for channel in CHANNELS:
        totals[channel] = self._mimic_total(sums, self._group_weights[channel])

    penalties = self._penalties()
    diagnostics: dict[str, torch.Tensor] = {}
    if self._clf is not None:
      clf_channels = self._clf_channels(diagnostics)
      for channel in CHANNELS:
        terms = clf_channels[channel]
        totals[channel] = totals[channel] + self._clf.w * terms["level"]
        # Outside the -1 floor, as in stage 1.
        penalties[channel] = penalties[channel] - self._clf.pen_w * terms["violation"]

    rewards = []
    for channel in CHANNELS:
      preclip = totals[channel] + penalties[channel]
      rewards.append(torch.nan_to_num(preclip.clamp(min=0.0), nan=0.0))
      diagnostics[f"split_reward_{channel}"] = rewards[-1]
      diagnostics[f"split_preclip_{channel}"] = preclip
      diagnostics[f"split_clipped_{channel}"] = (preclip < 0.0).float()
      diagnostics[f"split_penalty_{channel}"] = penalties[channel]
    self.channel_rewards = torch.stack(rewards, dim=-1)

    osl_err = self._stash("joint_q_err").index_select(-1, self._osl_joint_ids).abs()
    diagnostics["split_err_osl_knee"] = osl_err[:, 0]
    diagnostics["split_err_osl_ankle"] = osl_err[:, 1]
    if self._leg_contact is not None:
      diagnostics.update(leg_contact_diagnostics(self._leg_contact.data))
    self.components.update(diagnostics)

  ##
  # MimicReward kernels.
  ##

  def _mimic_group_sums(self) -> dict[str, dict[str, torch.Tensor]]:
    """Per active kernel that mixes groups, the group sums of its squared errors
    (per-site axis means for the site kernels), shared by both channels."""
    kw = self._kernel_weights
    sums: dict[str, dict[str, torch.Tensor]] = {}
    if kw["qpos"] != 0.0:
      sums["qpos"] = _group_sums(self._stash("qpos_err").square(), self._qpos_index)
    if kw["qvel"] != 0.0:
      sums["qvel"] = _group_sums(self._stash("qvel_err").square(), self._qvel_index)
    if kw["rpos"] != 0.0:
      per_site = self._stash("rpos_err").square().mean(-1)
      sums["rpos"] = _group_sums(per_site, self._site_index)
    if kw["rquat"] != 0.0:
      per_site = self._stash("rangles_err").square().mean(-1)
      sums["rquat"] = _group_sums(per_site, self._site_index)
    if kw["rvel_rot"] != 0.0:
      rvel_sq = self._stash("rvel_err").square()
      sums["rvel_rot"] = _group_sums(rvel_sq[..., :3].mean(-1), self._site_index)
      sums["rvel_lin"] = _group_sums(rvel_sq[..., 3:].mean(-1), self._site_index)
    return sums

  def _mimic_total(
    self, sums: Mapping[str, Mapping[str, torch.Tensor]], weights: Mapping[str, float]
  ) -> torch.Tensor:
    """MimicReward's weighted kernel sum for one channel.

    The qpos kernel's root-angle term (added unsquared in stage 1) scales with the
    human weight relative to the kernel's mean coordinate weight, which is 1 at
    uniform weights; the root position and velocity kernels are human-only and
    keep their stage-1 values.
    """
    kw, k = self._kernel_weights, MIMIC_EXPS
    terms: list[torch.Tensor] = []
    if "qpos" in sums:
      counts = self._kernel_counts["qpos"]
      w_sum = sum(weights[g] * counts[g] for g in GROUPS)
      angle_weight = weights["human"] * sum(counts.values()) / w_sum
      dist = self._weighted_mean("qpos", sums["qpos"], weights)
      dist = dist + angle_weight * self._stash("quat_angle")
      terms.append(kw["qpos"] * torch.exp(-k["qpos"] * dist))
    if "qvel" in sums:
      dist = self._weighted_mean("qvel", sums["qvel"], weights)
      terms.append(kw["qvel"] * torch.exp(-k["qvel"] * dist))
    if kw["root_pos"] != 0.0:
      terms.append(kw["root_pos"] * self._stash("r_root_pos"))
    if kw["root_vel"] != 0.0:
      terms.append(kw["root_vel"] * self._stash("r_root_vel"))
    for kernel in ("rpos", "rquat"):
      if kernel in sums:
        dist = self._weighted_mean(kernel, sums[kernel], weights)
        terms.append(kw[kernel] * torch.exp(-k[kernel] * dist))
    if "rvel_rot" in sums:
      rot = self._weighted_mean("rvel_rot", sums["rvel_rot"], weights)
      lin = self._weighted_mean("rvel_lin", sums["rvel_lin"], weights)
      kernels = torch.exp(-k["rvel"] * rot) + torch.exp(-k["rvel"] * lin)
      terms.append(self._weights["rvel"] * kernels)
    total = terms[0]
    for term in terms[1:]:
      total = total + term
    return total

  ##
  # The CLF.
  ##

  def _clf_channels(
    self, diagnostics: dict[str, torch.Tensor]
  ) -> dict[str, dict[str, torch.Tensor]]:
    """Per channel, the CLF's level and violation over the weighted coordinates:
    ``V = sum_i w_i qf_i / sum_i w_i`` with ``qf = p11 e^2 + 2 p12 e ed + p22
    ed^2``, and V-dot likewise, the OSL joint group included."""
    assert self._clf is not None
    cc = self._clf_cache
    e, ed, edd, first, valid = cc["e"], cc["ed"], cc["edd"], cc["first"], cc["valid"]
    tau, scale = self._clf.tau, self._split.osl_joint_scale

    # The OSL joint group, paired as the CLF's own groups: ed = tau * d/dt e.
    e_j = self._stash("joint_q_err").index_select(-1, self._osl_joint_ids) / scale
    ed_j = self._stash("joint_v_err").index_select(-1, self._osl_joint_ids)
    ed_j = ed_j * (tau / scale)
    edd_j = torch.where(
      first[:, None],
      torch.zeros_like(ed_j),
      (ed_j - self._edj_prev) * (tau / self._clf_dt),
    )
    self._edj_prev = ed_j.detach().clone()
    self._clf_joint = {"e": e_j, "ed": ed_j, "edd": edd_j}

    p11, p12, p22 = self._p11, self._p12, self._p22

    def qform(e_: torch.Tensor, ed_: torch.Tensor) -> torch.Tensor:
      return p11 * e_.square() + (2.0 * p12) * e_ * ed_ + p22 * ed_.square()

    def qform_rate(
      e_: torch.Tensor, ed_: torch.Tensor, edd_: torch.Tensor
    ) -> torch.Tensor:
      return ed_ * (p11 * e_ + p12 * ed_) + edd_ * (p12 * e_ + p22 * ed_)

    q_sums = _group_sums(qform(e, ed), self._clf_index)
    q_sums["osl_joints"] = qform(e_j, ed_j).sum(-1)
    d_sums = _group_sums(qform_rate(e, ed, edd), self._clf_index)
    d_sums["osl_joints"] = qform_rate(e_j, ed_j, edd_j).sum(-1)

    out: dict[str, dict[str, torch.Tensor]] = {}
    for channel in CHANNELS:
      weights = self._group_weights[channel]
      v = self._weighted_mean("clf", q_sums, weights).clamp(min=0.0)
      vdot = 2.0 * self._weighted_mean("clf", d_sums, weights)
      violation = ((vdot + self._alpha * v - self._eps_dead) / self._viol_norm).clamp(
        0, 1
      )
      violation = violation * (~first).float() * valid
      v_norm = v / self._v_max
      level = torch.where(
        torch.isfinite(v_norm), torch.exp(-v_norm.clamp(max=5.0)), torch.zeros_like(v)
      )
      out[channel] = {"level": level, "violation": violation}
      diagnostics[f"split_V_norm_{channel}"] = v_norm
      diagnostics[f"split_level_{channel}"] = level
      diagnostics[f"split_violation_{channel}"] = violation

    counts = self._kernel_counts["clf"]
    for group in GROUPS:
      n = counts[group]
      diagnostics[f"split_qform_{group}"] = (
        q_sums[group] / (n * self._v_max)
        if n
        else torch.zeros_like(q_sums["osl_joints"])
      )
    return out

  ##
  # Penalties.
  ##

  def _penalties(self) -> dict[str, torch.Tensor]:
    """Per channel, the action penalties before the CLF violation, floored at -1."""
    if not self._split.owner_penalties:
      team = self._stash("penalty")
      return {channel: team for channel in CHANNELS}
    action = self._stash("action")
    excess = (-1.0 - action).clamp(min=0.0) + (action - 1.0).clamp(min=0.0)
    squared = excess.square()
    qfrc = self._cache["qfrc"]
    torque = None if qfrc is None else self._torque_weights * qfrc.square()
    out: dict[str, torch.Tensor] = {}
    for agent in AGENTS:
      rows = squared.index_select(-1, self._action_rows[agent]).sum(-1)
      penalty = -self._oob_coeff * rows / self._action_dims[agent]
      index = self._torque_index[agent]
      if torque is not None and index.numel():
        penalty = penalty - torque.index_select(-1, index).sum(-1)
      out[agent] = penalty.clamp(min=-1.0)
    return out


##
# Leg contact.
##


def _scene_sensor(env: Any, name: str) -> Any | None:
  """The scene's sensor ``name``, or None (a scene without it, or a fake env)."""
  sensors = getattr(getattr(env, "scene", None), "sensors", None)
  return sensors.get(name) if isinstance(sensors, Mapping) else None


def leg_contact_diagnostics(data: Any) -> dict[str, torch.Tensor]:
  """:data:`SPLIT_CONTACT_DIAGNOSTICS` from the leg-contact sensor's
  ``ContactData``: the largest net-force magnitude over its substep history (the
  current force without one) and over its primaries, and whether it is non-zero."""
  history = getattr(data, "force_history", None)  # (B, P, H, 3)
  if history is not None:
    force = history.norm(dim=-1).amax(dim=-1)
  else:
    force = data.force.norm(dim=-1)  # (B, P, 3) -> (B, P)
  force = force.amax(dim=-1)
  return {"split_leg_contact": (force > 0.0).float(), "split_leg_contact_force": force}
