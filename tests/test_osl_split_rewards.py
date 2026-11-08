"""Tests for the stage-2 split reward (``mjlab.tasks.osl_split.mdp.rewards``).

Everything runs on CPU against a fake env and command, with no simulation: the reward
reads the reference, the robot's joint and site state, the action and the actuator
forces through a few attributes, which the fakes fill with a random reference and a
robot state at a per-env distance from it.

The first test pins the stash the split reward reads. ``osl_mimic``'s ``mimic_reward``
keeps its intermediates in ``_cache`` and ``_clf_cache``, and stage-1 training runs
use that code, so it is compared bit for bit with a frozen copy of the class as it
was before the stash (``_HeadMimicReward`` below, from commit c97e778b1), over first
steps, normal steps, resets, reference jumps and a non-finite state.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OSL_TORQUE_JOINTS
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.osl_mimic.config.env_cfgs import (
  CLF_WEIGHTS,
  MIMIC_WEIGHTS,
  OSL_TORQUE_WEIGHTS,
)
from mjlab.tasks.osl_mimic.mdp.commands import MIMIC_BODIES
from mjlab.tasks.osl_mimic.mdp.kinematics import (
  di_care_block,
  exact_decay_rate,
  quat_angle,
  relative_site_quantities,
  relative_site_velocities,
  rotvec_from_matrix,
  v_max_unit,
)
from mjlab.tasks.osl_mimic.mdp.reference import MimicReference, MimicReferenceSource
from mjlab.tasks.osl_mimic.mdp.rewards import (
  CLF_GROUPS,
  MIMIC_EXPS,
  MIMIC_TERMS,
  ClfCfg,
  mimic_reward,
)
from mjlab.tasks.osl_split.config.rl_cfg import osl_split_runner_cfg
from mjlab.tasks.osl_split.mdp.rewards import (
  CHANNELS,
  GROUPS,
  SPLIT_CLF_DIAGNOSTICS,
  SPLIT_DIAGNOSTICS,
  SplitRewardCfg,
  split_mimic_reward,
  stage1_channel_weights,
)
from mjlab.tasks.osl_split.rl.layout import ActionLayout
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

NUM_ENVS = 6
DT = 0.01
JOINT_NAMES = (
  "hip_flexion_r",
  "OSL_KA_L1_osl_ankle_angle_r",
  "socket_tx",
  "OSL_KA_L1_osl_knee_angle_r",
  "hip_flexion_l",
  "knee_angle_l",
  "ankle_angle_l",
  "lumbar_extension",
)
"""The OSL's ankle comes before its knee and apart from it, unlike the models (knee
then ankle, adjacent), so every OSL joint index is exercised."""
OSL_IDS = [JOINT_NAMES.index(j) for j in OSL_TORQUE_JOINTS]
ERRORS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.2)
"""Per-env distance of the robot from the reference: rewards from ~1 to clipped."""
LAYOUTS: dict[str, tuple[tuple[str, int], ...]] = {
  "msk": (("osl_motors", 2), ("muscles", 9)),
  "robot": (("servos", 7), ("osl_motors", 2)),
}
"""The stage-1 action orders (OSL rows 0:2 and 7:9), with few human rows."""
CLF = ClfCfg()
OSL_END = {"human": 0.5, "osl_sites": 1.0, "osl_joints": 10.0}


##
# The frozen stage-1 class.
##

# Verbatim from commit c97e778b1 (src/mjlab/tasks/osl_mimic/mdp/rewards.py, the
# last version without the stash), renamed: mimic_reward -> _HeadMimicReward and
# _yaw -> _head_yaw. Do not edit: it is the reference the stash is held to. Its
# _clf_terms binds v_rel and omega_rel conditionally, which pyright flags (in the
# source too), hence the file-level switch below.
# pyright: reportPossiblyUnboundVariable=false


def _head_yaw(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _HeadMimicReward(ManagerTermBase):
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
    qpos_dist = qpos_err.square().mean(-1) + quat_angle(root_q[:, 3:7], ref_root_quat)
    qvel_err = torch.cat([root_v - ref_root_v, joint_v - ref_joint_v], -1)
    qvel_dist = qvel_err.square().mean(-1)
    root_pos_dist = (root_q[:, :3] - ref_root_pos).square().mean(-1)

    # Root velocity: linear in the root frame, angular as stored (body frame).
    sim_root_vel = torch.cat(
      [quat_apply_inverse(root_q[:, 3:7], root_v[:, :3]), root_v[:, 3:]], -1
    )
    ref_root_vel = torch.cat(
      [quat_apply_inverse(ref_root_quat, ref_root_v[:, :3]), ref_root_v[:, 3:]], -1
    )
    root_vel_dist = (sim_root_vel - ref_root_vel).square().mean(-1)

    site_pos, site_mat, site_ang, site_lin = cmd.site_state()
    rpos, rangles, rvel = relative_site_quantities(
      site_pos, site_mat, site_ang, site_lin
    )
    ref_rpos, ref_rangles, ref_rvel = ref.rpos, ref.rangles, ref.rvel
    rpos_dist = (rpos - ref_rpos).square().mean((-1, -2))
    rangles_dist = (rangles - ref_rangles).square().mean((-1, -2))
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
    if self._torque_dofs.numel():
      qfrc = env.sim.data.qfrc_actuator[:, self._torque_dofs]
      penalty = penalty - (self._torque_weights * qfrc.square()).sum(-1)
    penalty = penalty.clamp(min=-1.0)

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


##
# Fakes.
##


def _unit(x: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.normalize(x, dim=-1)


class _FakeCommand:
  """A :class:`MimicReferenceSource` over random states: a random reference, and a
  robot ``error`` (per env) away from it."""

  def __init__(self, seed: int = 0) -> None:
    self.gen = torch.Generator().manual_seed(seed)
    self.cfg = SimpleNamespace(site_bodies=MIMIC_BODIES)
    self.site_body_ids = list(range(len(MIMIC_BODIES)))
    v_adr = torch.arange(6, 6 + len(JOINT_NAMES), dtype=torch.int)
    self.robot: Any = SimpleNamespace(
      joint_names=JOINT_NAMES,
      indexing=SimpleNamespace(joint_v_adr=v_adr),
      data=SimpleNamespace(),
    )
    self.jumped = torch.zeros(NUM_ENVS, dtype=torch.bool)
    self.sample()

  def randn(self, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=self.gen)

  def sample(self, error: float | tuple[float, ...] = ERRORS) -> None:
    """A new reference, and the robot ``error`` from it (0: exactly on it)."""
    b, s, j = NUM_ENVS, len(MIMIC_BODIES), len(JOINT_NAMES)
    scale = torch.as_tensor(error, dtype=torch.float32).expand(b)

    def near(x: torch.Tensor) -> torch.Tensor:
      shape = (b,) + (1,) * (x.dim() - 1)
      return x + scale.reshape(shape) * self.randn(*x.shape)

    def near_quat(q: torch.Tensor) -> torch.Tensor:
      shape = (b,) + (1,) * (q.dim() - 1)
      moved = _unit(q + scale.reshape(shape) * self.randn(*q.shape))
      return torch.where(scale.reshape(shape) > 0.0, moved, q)

    site_pos, site_ang, site_lin = (self.randn(b, s, 3) for _ in range(3))
    site_quat = _unit(self.randn(b, s, 4))
    site_mat = matrix_from_quat(site_quat)
    rpos, rangles, rvel = relative_site_quantities(
      site_pos, site_mat, site_ang, site_lin
    )
    v_rel, omega_rel = relative_site_velocities(site_mat, site_ang, site_lin)
    self.ref = MimicReference(
      root_pos_w=self.randn(b, 3),
      root_quat=_unit(self.randn(b, 4)),
      root_lin_vel_w=self.randn(b, 3),
      root_ang_vel_b=self.randn(b, 3),
      joint_pos=self.randn(b, j),
      joint_vel=self.randn(b, j),
      site_pos_w=site_pos,
      rpos=rpos,
      rangles=rangles,
      rvel=rvel,
      v_rel=v_rel,
      omega_rel=omega_rel,
    )
    self._root_q = torch.cat(
      [near(self.ref.root_pos_w), near_quat(self.ref.root_quat)], dim=-1
    )
    self._root_v = near(self.ref.root_qvel)
    self.robot.data.joint_pos = near(self.ref.joint_pos)
    self.robot.data.joint_vel = near(self.ref.joint_vel)
    self._sites = (
      near(site_pos),
      matrix_from_quat(near_quat(site_quat)),
      near(site_ang),
      near(site_lin),
    )

  @property
  def root_qpos(self) -> torch.Tensor:
    return self._root_q

  @property
  def root_qvel(self) -> torch.Tensor:
    return self._root_v

  @property
  def reference_jumped(self) -> torch.Tensor:
    return self.jumped

  def site_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return self._sites

  def reference(self, offset: int) -> MimicReference:
    assert offset == 1  # Rewards compare the post-step state with the next frame.
    return self.ref


def _fake_env(cmd: _FakeCommand, model: str = "msk") -> Any:
  dims = dict(LAYOUTS[model])
  terms = {name: SimpleNamespace(action_dim=dim) for name, dim in dims.items()}
  return SimpleNamespace(
    num_envs=NUM_ENVS,
    device="cpu",
    step_dt=DT,
    episode_length_buf=torch.zeros(NUM_ENVS, dtype=torch.long),
    command_manager=SimpleNamespace(get_term=lambda name: cmd),
    action_manager=SimpleNamespace(
      active_terms=list(dims),
      get_term=lambda name: terms[name],
      action=torch.zeros(NUM_ENVS, sum(dims.values())),
    ),
    sim=SimpleNamespace(
      data=SimpleNamespace(qfrc_actuator=torch.zeros(NUM_ENVS, 6 + len(JOINT_NAMES)))
    ),
  )


def _advance(
  env: Any,
  cmd: _FakeCommand,
  *,
  error: float | tuple[float, ...] = ERRORS,
  reset: tuple[int, ...] = (),
  jumped: tuple[int, ...] = (),
) -> None:
  """One env step: a new state, action and actuator force. ``reset`` envs are on
  their first episode step and ``jumped`` envs see a reference jump."""
  cmd.sample(error)
  env.episode_length_buf += 1
  env.episode_length_buf[list(reset)] = 1
  cmd.jumped = torch.zeros(NUM_ENVS, dtype=torch.bool)
  cmd.jumped[list(jumped)] = True
  num_actions = env.action_manager.action.shape[-1]
  env.action_manager.action = 1.2 * cmd.randn(NUM_ENVS, num_actions)
  env.sim.data.qfrc_actuator = 40.0 * cmd.randn(NUM_ENVS, 6 + len(JOINT_NAMES))


def _params(reward: str) -> dict[str, Any]:
  """The stage-1 term's params (``osl_mimic_env_cfg``)."""
  clf = reward == "clf"
  return {
    "command_name": "motion",
    "weights": dict(CLF_WEIGHTS if clf else MIMIC_WEIGHTS),
    "action_out_of_bounds_coeff": 0.01,
    "joint_torque_weights": dict(OSL_TORQUE_WEIGHTS),
    "clf": ClfCfg() if clf else None,
  }


def _split(
  reward: str, model: str = "msk", seed: int = 0, **channels: Any
) -> tuple[split_mimic_reward, Any, _FakeCommand, dict[str, Any]]:
  cmd = _FakeCommand(seed)
  env = _fake_env(cmd, model)
  params = {**_params(reward), "channels": SplitRewardCfg(**channels)}
  cfg = RewardTermCfg(func=split_mimic_reward, weight=1.0, params=params)
  return split_mimic_reward(cfg, env), env, cmd, params


def _expected_penalties(
  env: Any, model: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """The owner split's human and OSL penalties, and the team's, by hand."""
  layout = ActionLayout(LAYOUTS[model])
  action = env.action_manager.action
  squared = (action.abs() - 1.0).clamp(min=0.0).square()
  qfrc = env.sim.data.qfrc_actuator[:, [6 + i for i in OSL_IDS]]
  torque = (1e-5 * qfrc.square()).sum(-1)
  human, osl = layout.rows("human"), layout.rows("osl")
  pen_h = (-0.01 * squared[:, human].sum(-1) / len(human)).clamp(min=-1.0)
  pen_o = (-0.01 * squared[:, osl].sum(-1) / len(osl) - torque).clamp(min=-1.0)
  team = (-0.01 * squared.sum(-1) / action.shape[-1] - torque).clamp(min=-1.0)
  return pen_h, pen_o, team


def _clf_owners() -> list[str]:
  """Each CLF coordinate's group: rpos and rangles per relative site (site-major,
  three each), then root position and orientation (three each)."""
  per_site = [
    "osl_sites" if body.startswith("OSL_KA_L1_") else "human"
    for body in MIMIC_BODIES[1:]
  ]
  sites = [owner for owner in per_site for _ in range(3)]
  return sites + sites + ["human"] * 6


def _clf_constants() -> dict[str, float]:
  p11, p12, p22 = di_care_block(CLF.q_pos, CLF.q_vel, CLF.r)
  v_max = v_max_unit(p11, p12, p22) + 1e-8
  return {
    "p11": p11,
    "p12": p12,
    "p22": p22,
    "v_max": v_max,
    "alpha": CLF.alpha_safety * exact_decay_rate(CLF.q_pos, CLF.q_vel, CLF.r),
    "eps": CLF.eps_dead * v_max,
    "viol_norm": CLF.c_viol * v_max,
  }


def _assert_identical(got: torch.Tensor, want: torch.Tensor) -> None:
  assert got.dtype == want.dtype and got.shape == want.shape
  torch.testing.assert_close(got, want, rtol=0.0, atol=0.0, equal_nan=True)


##
# The stash in osl_mimic's mimic_reward.
##

STASH_KEYS = {
  "qpos_err",
  "quat_angle",
  "qvel_err",
  "root_pos_err",
  "root_vel_err",
  "rpos_err",
  "rangles_err",
  "rvel_err",
  "action",
  "qfrc",
  "penalty",
  "joint_q_err",
  "joint_v_err",
  "r_qpos",
  "r_qvel",
  "r_root_pos",
  "r_root_vel",
  "r_rpos",
  "r_rquat",
  "r_rvel_rot",
  "r_rvel_lin",
}


def _check_stash(term: mimic_reward, env: Any, cmd: _FakeCommand, call: int) -> None:
  """The stash holds this step's intermediates (equal_nan: one step is not finite)."""
  cache = term._cache
  assert set(cache) == STASH_KEYS

  def check(key: str, want: torch.Tensor) -> None:
    got = cache[key]
    assert got is not None
    torch.testing.assert_close(got, want, rtol=0.0, atol=0.0, equal_nan=True)

  ref, data = cmd.ref, cmd.robot.data
  root_q, root_v = cmd.root_qpos, cmd.root_qvel
  joint_q_err = data.joint_pos - ref.joint_pos
  joint_v_err = data.joint_vel - ref.joint_vel
  check("qpos_err", torch.cat([root_q[:, :3] - ref.root_pos_w, joint_q_err], -1))
  check("quat_angle", quat_angle(root_q[:, 3:7], ref.root_quat))
  check("qvel_err", torch.cat([root_v - ref.root_qvel, joint_v_err], -1))
  check("root_pos_err", root_q[:, :3] - ref.root_pos_w)
  sim_vel = torch.cat(
    [quat_apply_inverse(root_q[:, 3:7], root_v[:, :3]), root_v[:, 3:]], -1
  )
  ref_vel = torch.cat(
    [quat_apply_inverse(ref.root_quat, ref.root_lin_vel_w), ref.root_ang_vel_b], -1
  )
  check("root_vel_err", sim_vel - ref_vel)
  rpos, rangles, rvel = relative_site_quantities(*cmd.site_state())
  check("rpos_err", rpos - ref.rpos)
  check("rangles_err", rangles - ref.rangles)
  check("rvel_err", rvel - ref.rvel)
  assert cache["action"] is env.action_manager.action
  check("qfrc", env.sim.data.qfrc_actuator[:, [6 + i for i in OSL_IDS]])
  check("joint_q_err", joint_q_err)
  check("joint_v_err", joint_v_err)
  for name in ("qpos", "qvel", "root_pos", "root_vel", "rpos", "rquat"):
    check(f"r_{name}", term.components[f"reward_{name}"])
  check("r_rvel_rot", term.components["reward_rvel_rot"])
  check("r_rvel_lin", term.components["reward_rvel_lin"])
  # The penalty before the CLF violation, after the -1 floor.
  penalty = term.components["penalty_total"]
  if term._clf is not None:
    penalty = penalty + term._clf.pen_w * term.components["clf_violation"]
    clf_cache = term._clf_cache
    assert set(clf_cache) == {"e", "ed", "edd", "first", "valid"}
    assert clf_cache["e"].shape == clf_cache["ed"].shape == (NUM_ENVS, 102)
    first = (env.episode_length_buf <= 1) | cmd.jumped
    assert torch.equal(clf_cache["first"], first)
    # Cloned before the fill: zero on the term's first call only.
    assert torch.equal(clf_cache["valid"], torch.full((NUM_ENVS,), float(call > 0)))
    assert torch.equal(term._clf_valid, torch.ones(NUM_ENVS))
  stashed = cache["penalty"]
  assert stashed is not None
  torch.testing.assert_close(stashed, penalty, equal_nan=True)


@pytest.mark.parametrize("reward", ["mimic", "clf"])
def test_stash_leaves_stage1_bit_identical(reward: str) -> None:
  cmd = _FakeCommand(seed=1)
  env = _fake_env(cmd)
  params = _params(reward)
  cfg = RewardTermCfg(func=mimic_reward, weight=1.0, params=params)
  new, head = mimic_reward(cfg, env), _HeadMimicReward(cfg, env)
  steps: list[dict[str, Any]] = [
    {},  # Every env on its first episode step, and the terms' first call.
    {},
    {"jumped": (1, 4)},
    {"reset": (2,), "jumped": (5,)},
    {"error": 0.0},  # Exactly on the reference.
    {},  # Env 3 not finite.
    {},
  ]
  rewards = []
  for call, step in enumerate(steps):
    _advance(env, cmd, **step)
    if call == 5:
      cmd.robot.data.joint_pos[3, 0] = float("nan")
    got = new(env, **params)
    want = head(env, **params)
    _assert_identical(got, want)
    assert list(new.components) == list(head.components)
    for key, value in head.components.items():
      _assert_identical(new.components[key], value)
    if reward == "clf":
      _assert_identical(new._ed_prev, head._ed_prev)
      _assert_identical(new._clf_valid, head._clf_valid)
    _check_stash(new, env, cmd, call)
    rewards.append(want)
  # Not vacuous: rewards both clear of the clip and clipped (or zeroed when NaN).
  every = torch.cat(rewards)
  assert int((every > 0.05).sum()) >= 10 and bool((every == 0.0).any())


##
# The split reward.
##


@pytest.mark.parametrize("reward", ["mimic", "clf"])
def test_stage1_weights_reproduce_the_team_reward(reward: str) -> None:
  term, env, cmd, params = _split(reward, owner_penalties=False)
  assert term.group_weights == stage1_channel_weights(reward == "clf")
  positive = 0
  for step in ({}, {}, {"jumped": (1,)}, {"reset": (0,)}, {}):
    _advance(env, cmd, **step)
    team = term(env, **params)
    assert term.channel_rewards.shape == (NUM_ENVS, 2)
    for column in range(2):
      torch.testing.assert_close(
        term.channel_rewards[:, column], team, rtol=1e-5, atol=1e-6
      )
    if reward == "clf":
      for channel in CHANNELS:
        for key in ("V_norm", "level", "violation"):
          torch.testing.assert_close(
            term.components[f"split_{key}_{channel}"],
            term.components[f"clf_{key}"],
            rtol=1e-5,
            atol=1e-6,
          )
    positive += int((team > 0.05).sum())
  assert positive >= 5  # Not vacuous: most rewards are not clipped.


@pytest.mark.parametrize("model", ["msk", "robot"])
@pytest.mark.parametrize("reward", ["mimic", "clf"])
def test_owner_penalties_split_the_action_costs(model: str, reward: str) -> None:
  term, env, cmd, params = _split(reward, model=model)
  _advance(env, cmd)
  layout = ActionLayout(LAYOUTS[model])
  human, osl = layout.rows("human"), layout.rows("osl")
  action = torch.zeros(NUM_ENVS, layout.total_dim)
  action[0, human[0]] = 1.5  # Human rows only.
  action[1, osl[1]] = -2.0  # OSL rows only.
  action[2, human] = 3.0
  action[2, osl] = 1.25
  action[3, human] = 40.0  # Floors the human's penalty, not the OSL's.
  env.action_manager.action = action
  qfrc = torch.zeros(NUM_ENVS, 6 + len(JOINT_NAMES))
  qfrc[:, [6 + i for i in OSL_IDS]] = torch.tensor([100.0, -50.0])
  env.sim.data.qfrc_actuator = qfrc
  term(env, **params)

  pen_h, pen_o, team = _expected_penalties(env, model)
  assert pen_h[3] == -1.0 and pen_o[3] == pytest.approx(-0.125)
  assert pen_o[0] == pytest.approx(-0.125) and pen_h[1] == 0.0
  stashed = term._cache["penalty"]
  assert stashed is not None
  torch.testing.assert_close(stashed, team)
  for channel, expected in (("human", pen_h), ("osl", pen_o)):
    penalty = term.components[f"split_penalty_{channel}"]
    if reward == "clf":  # The violation is subtracted outside the floor.
      penalty = penalty + CLF.pen_w * term.components[f"split_violation_{channel}"]
    torch.testing.assert_close(penalty, expected)
  # At the stage-1 weights the channels differ from the team in the penalty only.
  imitation = term.components["reward_preclip"] - term.components["penalty_total"]
  for channel in CHANNELS:
    total = (
      term.components[f"split_preclip_{channel}"]
      - term.components[f"split_penalty_{channel}"]
    )
    torch.testing.assert_close(total, imitation, rtol=1e-5, atol=1e-6)


def test_weighted_clf_matches_an_explicit_computation() -> None:
  weights = {
    "human": {"human": 1.0, "osl_sites": 0.7, "osl_joints": 2.0},
    "osl": dict(OSL_END),
  }
  term, env, cmd, params = _split("clf", initial_weights=weights)
  owners = _clf_owners()
  assert len(owners) == 102
  assert (
    term.group_coordinate_counts()
    == {
      "human": owners.count("human"),
      "osl_sites": owners.count("osl_sites"),
      "osl_joints": 2,
    }
    == {"human": 84, "osl_sites": 18, "osl_joints": 2}
  )
  k = _clf_constants()

  def qform(e: torch.Tensor, ed: torch.Tensor) -> torch.Tensor:
    return k["p11"] * e.square() + 2.0 * k["p12"] * e * ed + k["p22"] * ed.square()

  def rate(e: torch.Tensor, ed: torch.Tensor, edd: torch.Tensor) -> torch.Tensor:
    return ed * (k["p11"] * e + k["p12"] * ed) + edd * (k["p12"] * e + k["p22"] * ed)

  ed_j_prev: torch.Tensor | None = None
  for step in ({}, {}, {"jumped": (2,)}, {"reset": (4,)}, {}):
    _advance(env, cmd, error=tuple(e / 4 for e in ERRORS), **step)
    term(env, **params)
    cache = term._clf_cache
    e, ed, edd = (cache[key].double() for key in ("e", "ed", "edd"))
    first, valid = cache["first"], cache["valid"].double()
    data, ref = cmd.robot.data, cmd.ref
    e_j = (data.joint_pos - ref.joint_pos)[:, OSL_IDS].double() / 0.2
    ed_j = (data.joint_vel - ref.joint_vel)[:, OSL_IDS].double() * (CLF.tau / 0.2)
    if ed_j_prev is None:
      edd_j = torch.zeros_like(ed_j)
    else:
      edd_j = (ed_j - ed_j_prev) * (CLF.tau / DT)
    edd_j[first] = 0.0
    ed_j_prev = ed_j
    penalties = _expected_penalties(env, "msk")
    for column, channel in enumerate(CHANNELS):
      w = torch.tensor([weights[channel][o] for o in owners], dtype=torch.float64)
      w_j = weights[channel]["osl_joints"]
      total_w = float(w.sum()) + 2.0 * w_j
      v = (w * qform(e, ed)).sum(-1) + w_j * qform(e_j, ed_j).sum(-1)
      v = (v / total_w).clamp(min=0.0)
      vdot = (w * rate(e, ed, edd)).sum(-1) + w_j * rate(e_j, ed_j, edd_j).sum(-1)
      vdot = 2.0 * vdot / total_w
      violation = ((vdot + k["alpha"] * v - k["eps"]) / k["viol_norm"]).clamp(0, 1)
      violation = violation * (~first).double() * valid
      level = torch.exp(-(v / k["v_max"]).clamp(max=5.0))
      penalty = penalties[column].double() - CLF.pen_w * violation
      reward = (level + penalty).clamp(min=0.0)
      components = term.components
      torch.testing.assert_close(
        components[f"split_V_norm_{channel}"].double(),
        v / k["v_max"],
        rtol=1e-5,
        atol=1e-6,
      )
      torch.testing.assert_close(
        components[f"split_violation_{channel}"].double(),
        violation,
        rtol=1e-4,
        atol=1e-4,
      )
      torch.testing.assert_close(
        term.channel_rewards[:, column].double(), reward, rtol=1e-4, atol=1e-4
      )
    # Each group's mean quadratic form, in V_max units.
    qf = qform(e, ed)
    for group in ("human", "osl_sites"):
      mask = torch.tensor([o == group for o in owners])
      torch.testing.assert_close(
        term.components[f"split_qform_{group}"].double(),
        qf[:, mask].mean(-1) / k["v_max"],
        rtol=1e-5,
        atol=1e-6,
      )
    torch.testing.assert_close(
      term.components["split_qform_osl_joints"].double(),
      qform(e_j, ed_j).mean(-1) / k["v_max"],
      rtol=1e-5,
      atol=1e-6,
    )


def test_osl_joint_group_pairs_errors_with_their_rates() -> None:
  """The joint group's ``ed`` is tau times the rate of its ``e`` (knee with knee,
  ankle with ankle, one scale), and ``edd`` the rate of ``ed``."""
  joints_only = {
    "human": stage1_channel_weights(True)["human"],
    "osl": {"human": 0.0, "osl_sites": 0.0, "osl_joints": 1.0},
  }
  term, env, cmd, params = _split("clf", initial_weights=joints_only)
  cmd.sample(error=0.0)  # The robot on the reference, but for the OSL joints.
  amp, phase = torch.tensor([0.12, 0.05]), torch.tensor([0.3, 1.9])
  omega = 2.0 * math.pi * 1.5
  ref_q, ref_qd = cmd.ref.joint_pos.clone(), cmd.ref.joint_vel.clone()
  k = _clf_constants()
  history: list[dict[str, torch.Tensor]] = []
  for step in range(8):
    t = step * DT
    dq = amp * torch.sin(omega * t + phase)
    dqd = amp * omega * torch.cos(omega * t + phase)
    joint_pos, joint_vel = ref_q.clone(), ref_qd.clone()
    joint_pos[:, OSL_IDS] += dq
    joint_vel[:, OSL_IDS] += dqd
    cmd.robot.data.joint_pos, cmd.robot.data.joint_vel = joint_pos, joint_vel
    env.episode_length_buf += 1
    term(env, **params)
    group = term._clf_joint
    torch.testing.assert_close(group["e"], (dq / 0.2).expand(NUM_ENVS, 2))
    torch.testing.assert_close(group["ed"], (dqd * (CLF.tau / 0.2)).expand(NUM_ENVS, 2))
    if step == 0:
      assert torch.equal(group["edd"], torch.zeros(NUM_ENVS, 2))
    else:
      expected = (group["ed"] - history[-1]["ed"]) * (CLF.tau / DT)
      torch.testing.assert_close(group["edd"], expected)
    history.append({key: value.clone() for key, value in group.items()})

    # The OSL channel's V is the joint group's alone; the human's ignores it.
    e, ed = group["e"].double(), group["ed"].double()
    qf = k["p11"] * e.square() + 2 * k["p12"] * e * ed + k["p22"] * ed.square()
    torch.testing.assert_close(
      term.components["split_V_norm_osl"].double(),
      qf.mean(-1) / k["v_max"],
      rtol=1e-5,
      atol=1e-7,
    )
    assert float(term.components["split_V_norm_human"].abs().max()) < 1e-6

  for prev, cur in zip(history, history[1:], strict=False):
    # The trapezoid rule: d e / dt = ed / tau, to O(dt^2).
    torch.testing.assert_close(
      (cur["e"] - prev["e"]) / DT * CLF.tau,
      (cur["ed"] + prev["ed"]) / 2.0,
      rtol=2e-3,
      atol=1e-5,
    )


def test_weighted_mimic_kernels_match_an_explicit_computation() -> None:
  weights = {
    "human": stage1_channel_weights(False)["human"],
    "osl": {"human": 0.5, "osl_sites": 2.0, "osl_joints": 10.0},
  }
  term, env, cmd, params = _split("mimic", initial_weights=weights)
  _advance(env, cmd, error=tuple(e / 4 for e in ERRORS))
  term(env, **params)

  ref, data = cmd.ref, cmd.robot.data
  root_q, root_v = cmd.root_qpos, cmd.root_qvel
  sim_rpos, sim_rangles, sim_rvel = relative_site_quantities(*cmd.site_state())
  site_err = {
    "rpos": (sim_rpos - ref.rpos).double().square().mean(-1),
    "rquat": (sim_rangles - ref.rangles).double().square().mean(-1),
    "rvel_rot": (sim_rvel - ref.rvel)[..., :3].double().square().mean(-1),
    "rvel_lin": (sim_rvel - ref.rvel)[..., 3:].double().square().mean(-1),
  }
  qpos_err = torch.cat(
    [root_q[:, :3] - ref.root_pos_w, data.joint_pos - ref.joint_pos], -1
  ).double()
  qvel_err = torch.cat([root_v - ref.root_qvel, data.joint_vel - ref.joint_vel], -1)
  angle = quat_angle(root_q[:, 3:7], ref.root_quat).double()
  osl_site = [body.startswith("OSL_KA_L1_") for body in MIMIC_BODIES[1:]]
  w, exps = MIMIC_WEIGHTS, MIMIC_EXPS
  for channel in CHANNELS:
    g = weights[channel]
    joints = [g["osl_joints"] if i in OSL_IDS else g["human"] for i in range(8)]
    w_qpos = torch.tensor([g["human"]] * 3 + joints, dtype=torch.float64)
    w_qvel = torch.tensor([g["human"]] * 6 + joints, dtype=torch.float64)
    w_site = torch.tensor(
      [g["osl_sites"] if osl else g["human"] for osl in osl_site], dtype=torch.float64
    )
    qpos = (w_qpos * qpos_err.square()).sum(-1) / w_qpos.sum()
    qpos = qpos + g["human"] / float(w_qpos.mean()) * angle
    qvel = (w_qvel * qvel_err.double().square()).sum(-1) / w_qvel.sum()
    dist = {k: (w_site * v).sum(-1) / w_site.sum() for k, v in site_err.items()}
    total = (
      w["qpos"] * torch.exp(-exps["qpos"] * qpos)
      + w["qvel"] * torch.exp(-exps["qvel"] * qvel)
      + w["root_pos"] * term.components["reward_root_pos"].double()
      + w["root_vel"] * term.components["reward_root_vel"].double()
      + w["rpos"] * torch.exp(-exps["rpos"] * dist["rpos"])
      + w["rquat"] * torch.exp(-exps["rquat"] * dist["rquat"])
      + w["rvel"] * torch.exp(-exps["rvel"] * dist["rvel_rot"])
      + w["rvel"] * torch.exp(-exps["rvel"] * dist["rvel_lin"])
    )
    got = (
      term.components[f"split_preclip_{channel}"]
      - term.components[f"split_penalty_{channel}"]
    )
    torch.testing.assert_close(got.double(), total, rtol=1e-5, atol=1e-6)
    if channel == "human":  # Uniform weights: the stage-1 imitation total.
      stage1 = term.components["reward_preclip"] - term.components["penalty_total"]
      torch.testing.assert_close(got, stage1, rtol=1e-5, atol=1e-6)
  # The re-weighting matters: the OSL channel differs from the human's.
  rewards = term.channel_rewards
  assert float((rewards[:, 0] - rewards[:, 1]).abs().max()) > 1e-3


def test_channel_rewards_survive_reset() -> None:
  term, env, cmd, params = _split("clf")
  _advance(env, cmd)
  term(env, **params)
  rewards = term.channel_rewards
  snapshot = rewards.clone()
  term.reset(torch.tensor([0, 3]))
  term.reset(None)
  assert term.channel_rewards is rewards
  assert torch.equal(term.channel_rewards, snapshot)
  # A new tensor every call, so a held reference stays this step's.
  _advance(env, cmd)
  term(env, **params)
  assert term.channel_rewards is not rewards
  assert torch.equal(rewards, snapshot)


@pytest.mark.parametrize("reward", ["mimic", "clf"])
def test_diagnostics(reward: str) -> None:
  term, env, cmd, params = _split(reward)
  _advance(env, cmd)
  term(env, **params)
  clf_keys = set(SPLIT_CLF_DIAGNOSTICS)
  expected = set(SPLIT_DIAGNOSTICS) - (set() if reward == "clf" else clf_keys)
  assert expected <= set(term.components)
  assert not (set(term.components) - expected) & clf_keys
  for key in expected:
    value = term.components[key]
    assert value.shape == (NUM_ENVS,) and bool(torch.isfinite(value).all())
  # Knee and ankle by name, whatever the robot's joint order.
  err = (cmd.robot.data.joint_pos - cmd.ref.joint_pos).abs()
  knee, ankle = OSL_TORQUE_JOINTS
  components = term.components
  torch.testing.assert_close(
    components["split_err_osl_knee"], err[:, JOINT_NAMES.index(knee)]
  )
  torch.testing.assert_close(
    components["split_err_osl_ankle"], err[:, JOINT_NAMES.index(ankle)]
  )
  for column, channel in enumerate(CHANNELS):
    reward_c = components[f"split_reward_{channel}"]
    assert torch.equal(reward_c, term.channel_rewards[:, column])
    preclip = components[f"split_preclip_{channel}"]
    assert torch.equal(components[f"split_clipped_{channel}"], (preclip < 0).float())
    assert torch.equal(reward_c, preclip.clamp(min=0.0))


def test_set_group_weights_validates() -> None:
  term, *_ = _split("clf")
  good = stage1_channel_weights(True)
  bad: list[dict[str, Any]] = [
    {"human": good["human"]},  # A channel missing.
    {**good, "extra": good["human"]},
    {**good, "osl": {"human": 1.0, "osl_sites": 1.0}},  # A group missing.
    {**good, "osl": {**good["osl"], "osl_feet": 1.0}},
    {**good, "osl": {**good["osl"], "human": -0.1}},
    {**good, "osl": {**good["osl"], "human": float("nan")}},
    {**good, "osl": {**good["osl"], "osl_joints": float("inf")}},
    {**good, "osl": dict.fromkeys(GROUPS, 0.0)},  # V undefined.
  ]
  term.set_group_weights({"human": good["human"], "osl": OSL_END})
  for weights in bad:
    with pytest.raises(ValueError):
      term.set_group_weights(weights)
  # A rejected update keeps the previous weights.
  assert term.group_weights == {"human": good["human"], "osl": OSL_END}
  # Under the CLF the joint group alone defines V; under MimicReward it does not
  # reach the site kernels, nor the sites the joint kernels.
  term.set_group_weights(
    {**good, "osl": {"human": 0.0, "osl_sites": 0.0, "osl_joints": 1.0}}
  )
  mimic, *_ = _split("mimic")
  for groups in (
    {"human": 0.0, "osl_sites": 0.0, "osl_joints": 1.0},
    {"human": 0.0, "osl_sites": 1.0, "osl_joints": 0.0},
  ):
    with pytest.raises(ValueError, match="kernel"):
      mimic.set_group_weights({**good, "osl": groups})
    with pytest.raises(ValueError, match="kernel"):
      mimic.check_group_weights({**good, "osl": groups})
  # check_group_weights refuses what set_group_weights refuses, and stores nothing.
  before = term.group_weights
  for weights in bad:
    with pytest.raises(ValueError):
      term.check_group_weights(weights)
  checked = term.check_group_weights({"human": OSL_END, "osl": good["osl"]})
  assert checked == {"human": OSL_END, "osl": good["osl"]}
  assert term.group_weights == before
  # The property is a copy.
  copy = term.group_weights
  copy["osl"]["human"] = 123.0
  assert term.group_weights["osl"]["human"] == 0.0


def test_construction() -> None:
  # Defaults, a dict of fields, and initial weights.
  term, *_ = _split("clf")
  assert term.group_weights == stage1_channel_weights(True)
  cmd = _FakeCommand()
  env = _fake_env(cmd)
  params = {
    **_params("mimic"),
    "channels": {"initial_weights": {"human": OSL_END, "osl": OSL_END}},
  }
  cfg = RewardTermCfg(func=split_mimic_reward, weight=1.0, params=params)
  assert split_mimic_reward(cfg, env).group_weights == {
    "human": OSL_END,
    "osl": OSL_END,
  }
  no_channels = {**_params("mimic")}
  cfg = RewardTermCfg(func=split_mimic_reward, weight=1.0, params=no_channels)
  assert split_mimic_reward(cfg, env).group_weights == stage1_channel_weights(False)

  def build(torque: bool = True, **channels: Any) -> split_mimic_reward:
    params = {**_params("clf"), "channels": SplitRewardCfg(**channels)}
    if not torque:
      params["joint_torque_weights"] = {}
    cfg = RewardTermCfg(func=split_mimic_reward, weight=1.0, params=params)
    return split_mimic_reward(cfg, env)

  with pytest.raises(ValueError, match="osl_joint_scale"):
    build(osl_joint_scale=0.0)
  with pytest.raises(ValueError):
    build(initial_weights={"human": OSL_END})
  # Exactly three OSL sites, and both OSL joints, are required.
  cmd.cfg = SimpleNamespace(site_bodies=MIMIC_BODIES[:-1] + ("toes_r",))
  with pytest.raises(ValueError, match="3 OSL mimic sites"):
    build()
  cmd.cfg = SimpleNamespace(site_bodies=MIMIC_BODIES)
  cmd.robot.joint_names = tuple(j for j in JOINT_NAMES if "knee" not in j)
  with pytest.raises(ValueError, match="OSL joints"):
    build(torque=False)  # The torque weights would fail first, in mimic_reward.


def test_effective_shares() -> None:
  term, *_ = _split("clf")
  for channel in CHANNELS:
    assert term.effective_shares()[channel] == pytest.approx(
      {"human": 84 / 102, "osl_sites": 18 / 102, "osl_joints": 0.0}
    )
  term.set_group_weights(
    {"human": stage1_channel_weights(True)["human"], "osl": OSL_END}
  )
  # The runner config's end weights: 84 x 0.5, 18 x 1, 2 x 10 of 80.
  assert term.effective_shares()["osl"] == pytest.approx(
    {"human": 0.525, "osl_sites": 0.225, "osl_joints": 0.25}
  )

  # MimicReward: each kernel's weight split by its own coordinates, the root
  # position and velocity kernels fully human. The fake has 8 joints.
  mimic, *_ = _split("mimic")
  assert mimic.group_coordinate_counts() == {
    "human": (3 + 6) + (6 + 6) + 4 * 13,
    "osl_sites": 4 * 3,
    "osl_joints": 2 + 2,
  }
  w = MIMIC_WEIGHTS
  total = sum(w.values()) + w["rvel"]  # The rotational and linear rvel kernels.
  site_weight = w["rpos"] + w["rquat"] + 2 * w["rvel"]
  osl_joints = (w["qpos"] * 2 / 11 + w["qvel"] * 2 / 14) / total
  osl_sites = site_weight * 3 / 16 / total
  shares = mimic.effective_shares()
  for channel in CHANNELS:
    assert shares[channel] == pytest.approx(
      {
        "human": 1.0 - osl_joints - osl_sites,
        "osl_sites": osl_sites,
        "osl_joints": osl_joints,
      }
    )


@pytest.mark.parametrize("reward", ["mimic", "clf"])
def test_runner_channel_defaults_match(reward: str) -> None:
  """The runner config starts both channels at the stage-1 weights, and its end
  weights are valid for the reward."""
  cfg = osl_split_runner_cfg("msk", "compliant", cast(Any, reward), "test")
  stage1 = stage1_channel_weights(reward == "clf")
  channels = cfg.channels
  assert asdict(channels.human_start) == stage1["human"]
  assert asdict(channels.osl_start) == stage1["osl"]
  assert asdict(channels.human_end) == stage1["human"]
  term, *_ = _split(reward)
  term.set_group_weights(
    {"human": asdict(channels.human_end), "osl": asdict(channels.osl_end)}
  )
  assert set(GROUPS) == set(asdict(channels.osl_end))
  assert SplitRewardCfg() == SplitRewardCfg(True, 0.2, None)
