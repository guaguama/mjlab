"""musclemimic's ``MimicReward`` and ``CLFReward`` as one mjlab reward term.

The term returns musclemimic's per-step total exactly. It needs ``weight=1`` and
``scale_rewards_by_dt=False``, because the total is clipped at zero after the
weighted sum and so cannot be split into separately weighted mjlab terms.

Sources: ``musclemimic/core/reward/trajectory_based.py`` (``MimicReward.__call__``,
non-``use_mean_exp_reward`` branch) and ``musclemimic/core/reward/clf.py``.
Everything is compared with the command's reference one step ahead,
``reference(1)`` (see ``commands.py``); any :class:`~.reference.MimicReferenceSource`
works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

from .kinematics import (
  di_care_block,
  exact_decay_rate,
  quat_angle,
  relative_site_quantities,
  relative_site_velocities,
  rotvec_from_matrix,
  v_max_unit,
)
from .reference import MimicReference, MimicReferenceSource

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

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


@dataclass
class ClfCfg:
  """``CLFReward``'s constants, with ``conf_osl_fullbody_clf.yaml``'s values."""

  w: float = 1.0
  """Weight of the level term ``exp(-min(V / V_max, 5))``."""
  pen_w: float = 0.25
  """Weight of the decrease-condition violation."""
  tau: float = 0.10
  q_pos: float = 1.0
  q_vel: float = 1.0
  r: float = 0.1
  alpha: float | None = None
  """None: ``alpha_safety`` times the exact achievable decay rate."""
  alpha_safety: float = 0.5
  c_viol: float = 0.10
  eps_dead: float = 0.01
  scales: dict[str, float] = field(
    default_factory=lambda: {
      "rpos": 0.05,
      "rangles": 0.20,
      "root_pos": 0.10,
      "root_ori": 0.20,
    }
  )
  """Per-group error scales, in metres or radians. A group absent here is off.
  musclemimic's ``qpos_joint`` group is off in every OSL config and not ported."""


def _yaw(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class mimic_reward(ManagerTermBase):
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
    rpos, rangles, rvel = relative_site_quantities(
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
        torch.sin(_yaw(root_q[:, 3:7]) - _yaw(ref_root_quat)),
        torch.cos(_yaw(root_q[:, 3:7]) - _yaw(ref_root_quat)),
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


def mimic_component(
  env: ManagerBasedRlEnv, name: str, reward_name: str = "mimic"
) -> torch.Tensor:
  """One per-env diagnostic of :class:`mimic_reward` from the current step."""
  term = env.reward_manager.get_term_cfg(reward_name).func
  components = cast(mimic_reward, term).components
  if name in components:
    return components[name]
  return torch.zeros(env.num_envs, device=env.device)
