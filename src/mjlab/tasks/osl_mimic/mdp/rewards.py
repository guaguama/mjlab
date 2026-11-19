"""musclemimic's ``MimicReward`` and ``CLFReward`` as one mjlab reward term, with
CLF-RL's decrease condition and regularizers.

The term returns musclemimic's per-step total exactly. It needs ``weight=1`` and
``scale_rewards_by_dt=False``, because the total is clipped at zero after the
weighted sum and so cannot be split into separately weighted mjlab terms.

Sources: ``musclemimic/core/reward/trajectory_based.py`` (``MimicReward.__call__``,
non-``use_mean_exp_reward`` branch) and ``musclemimic/core/reward/clf.py``.
Everything is compared with the command's reference one step ahead,
``reference(1)`` (see ``commands.py``); any :class:`~.reference.MimicReferenceSource`
works.

Two additions to musclemimic's reward, both inside its clip at 0 (a penalty outside
it could make a step's reward negative, which rewards ending episodes early):

* the CLF's decrease condition in CLF-RL's graded, rate-based form (``ClfCfg``'s
  ``decay_form="paper"``, the default; ``"musclemimic"`` is ``CLFReward``'s own);
* regularizers (:class:`RegularizerCfg`), subtracted with the action penalties
  inside the -1 floor: the action rate of the OSL motors and of the robot model's
  servos, the hard joint-limit excess of the legs and the OSL, and the robot
  servos' torque and torque saturation.

The total is ``max(0, imitation + max(-1, penalties - scale x regularizers) -
pen_scale x pen_w x violation)``. With ``decay_form="musclemimic"``, musclemimic's
``pen_w`` and every regularizer at 0 it is musclemimic's, bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, get_args

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuator
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

DecayForm = Literal["paper", "musclemimic"]
DECAY_FORMS: tuple[str, ...] = get_args(DecayForm)

OSL_ACTION_TERM = "osl_motors"
"""The action term of the OSL motors, in both models."""
SERVO_ACTION_TERM = "servos"
"""The action term of the robot model's position servos (absent on the
musculoskeletal model)."""

LIMIT_JOINTS: tuple[str, ...] = (
  "hip_flexion_r",
  "hip_adduction_r",
  "hip_rotation_r",
  "hip_flexion_l",
  "hip_adduction_l",
  "hip_rotation_l",
  "knee_angle_l",
  "ankle_angle_l",
  "OSL_KA_L1_osl_knee_angle_r",
  "OSL_KA_L1_osl_ankle_angle_r",
)
"""The joints the hard joint-limit penalty watches by default: both hips, the
intact knee and ankle, and the OSL knee and ankle. Left out on purpose: the wrists
(83% of the round-6 teacher's limit excess is the wrists hanging on their deviation
stops), the subtalar and mtp joints (the gait library itself passes their limits,
by up to 0.23 rad), the equality followers and the socket's joints."""

REGULARIZERS: tuple[str, ...] = (
  "osl_action_rate",
  "servo_action_rate",
  "joint_limits",
  "servo_torque",
  "servo_saturation",
)
""":class:`RegularizerCfg`'s coefficients, in the order of the ``reg_<name>``
diagnostics."""
SERVO_LOG_DIAGNOSTICS: tuple[str, ...] = ("servo_torque_cost", "servo_saturation_cost")
"""``RegularizerCfg.log_servo_terms``'s diagnostics: the robot servos' ``servo_torque``
and ``servo_saturation`` costs, unweighted (``reg_servo_*`` are the weighted costs
the reward subtracts)."""
REGULARIZER_DIAGNOSTICS: tuple[str, ...] = (
  *(f"reg_{name}" for name in REGULARIZERS),
  "reg_total",
)
"""``reg_<name>``: each regularizer's cost at its coefficient, before
``RegularizerCfg.scale`` (0 when off; the joint limits summed over the joints);
``reg_total``: the scaled sum subtracted inside the -1 floor. Present whenever the
term has a :class:`RegularizerCfg`."""
CLF_EXTRA_DIAGNOSTICS: tuple[str, ...] = (
  "clf_vdot_fd",
  "clf_violation_saturated",
  "clf_clipped_recovering",
)
"""``clf_vdot_fd``: the finite-difference ``(V_t - V_{t-w}) / (w step_dt)`` in 1/s over
``ClfCfg.vdot_window`` steps (0 until ``w`` fresh steps follow an episode start or a
reference jump), whatever the decay form;
musclemimic's semi-analytic ``clf_vdot`` is per ``tau`` instead (about ``tau x
clf_vdot_fd``). ``clf_violation_saturated``: 1 where the violation is at its cap;
``clf_clipped_recovering``: 1 where the reward is clipped at 0 while V falls."""
ACTIVATION_DIAGNOSTIC = "activation_energy"
"""``mean(act^2)`` over the muscles: a metric only, never part of the reward; absent
on a model without muscles (the robot)."""


@dataclass
class ClfCfg:
  """``CLFReward``'s constants, with ``conf_osl_fullbody_clf.yaml``'s values, and the
  form of its decrease condition.

  The level term ``w x exp(-min(V / V_max, 5))`` is the tracking reward. The
  decrease condition's violation, in [0, 1], is subtracted times ``pen_scale x
  pen_w``, after the penalties' -1 floor and inside the clip at 0. It is 0 on an
  episode's first step and where the reference jumped, and takes one of two forms
  (``decay_form``):

  * ``"paper"`` (the default): CLF-RL's (Olkin et al., arXiv 2603.25902) graded,
    rate-based condition, ``clip(max(Vdot + decay_rate V, 0) / (decay_norm_rate
    V_max), 0, 1)``, with the finite-difference ``Vdot = (V_t - V_{t-w}) / (w
    step_dt)`` over ``w = vdot_window`` steps. It is 0 until ``w`` steps of fresh
    history follow an episode start or a reference jump;
  * ``"musclemimic"``: ``CLFReward``'s ``clip((Vdot + alpha V - eps_dead V_max) /
    (c_viol V_max), 0, 1)``, with its semi-analytic Vdot. Its ``alpha V`` alone
    fills the cap once V_norm >= 0.171, so away from the reference it is a
    constant offset rather than a rate condition.

  musclemimic's reward is ``decay_form="musclemimic"`` with ``pen_w`` 0.25.
  """

  w: float = 1.0
  """Weight of the level term ``exp(-min(V / V_max, 5))``."""
  pen_w: float = 0.5
  """Weight of the decrease-condition violation: half the level term's, the
  paper's 10:-5 (musclemimic's config: 0.25)."""
  pen_scale: float = 1.0
  """A multiplier of ``pen_w``, read every step, for env stages
  (``rewards.mimic.params.clf.pen_scale``): the CLF tasks ramp it in, since early
  in training the clip at 0 binds."""
  decay_form: DecayForm = "paper"
  """The decrease condition's form (see the class docstring)."""
  decay_rate: float = 0.5
  """``"paper"``: the decay rate lambda the condition demands, in 1/s (read every
  step)."""
  decay_norm_rate: float = 4.5
  """``"paper"``: the violation's normalizer in units of V_max, in 1/s (read every
  step): the paper's worst-case violation ``2 ||P|| eta_max etadot_max + alpha
  lambda_max(P) eta_max^2`` in our V's units, with ``etadot_max / eta_max`` = 2/s and
  ``alpha`` = 0.5/s."""
  vdot_window: int = 1
  """``"paper"``: the env steps ``w`` the finite-difference V-dot spans (fixed at
  build: it sizes a history). The robot tasks take 10 (0.1 s, the CLF's ``tau``):
  their sampled position-servo targets make V jitter from step to step, so the
  one-step rate saturates the violation on about a quarter of the steps at training
  noise (the musculoskeletal model: about 1%)."""
  tau: float = 0.10
  q_pos: float = 1.0
  q_vel: float = 1.0
  r: float = 0.1
  alpha: float | None = None
  """``"musclemimic"``: None for ``alpha_safety`` times the exact achievable decay
  rate."""
  alpha_safety: float = 0.5
  c_viol: float = 0.10
  """``"musclemimic"``: the violation's normalizer, in units of V_max."""
  eps_dead: float = 0.01
  """``"musclemimic"``: the dead zone, in units of V_max."""
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


@dataclass
class RegularizerCfg:
  """The regularizers of :class:`mimic_reward`, each a cost subtracted inside the
  penalties' -1 floor (so inside the clip at 0) with its own coefficient, times
  ``scale``. Every field is read on every step, so env stages can change them
  (``rewards.mimic.params.regularizers.<field>``); a term at coefficient 0 is not
  computed. All coefficients default to 0: musclemimic's reward has none.

  * ``osl_action_rate``: ``mean over the OSL rows of (a_t - a_{t-1})^2``, on the
    policy's actions clipped to [-1, 1] (the commands the motors execute), 0 on an
    episode's first step (mjlab zeroes the previous action at reset). On the
    sampled action, never the policy mean, which would give PPO no gradient: its
    main effect is pressure on the action noise.
  * ``servo_action_rate``: the same over the robot's servo rows (absent on the
    musculoskeletal model, which gets no muscle action rate).
  * ``joint_limits``: ``sum over joint_limit_joints of relu(lo - q) + relu(q -
    hi)``, in rad, against the joints' hard limits.
  * ``servo_torque``: ``mean over the robot's servos of (tau / tau_max)^2``, tau
    each servo's actuator force (gear 1) and tau_max its force range.
  * ``servo_saturation``: ``mean over the servos of relu(|demand| - tau_max) /
    tau_max``, ``demand`` the PD force the step's last physics substep asked of the
    servo before its force limit: ``Kp (ctrl - q) - Kd qd`` from ``data.ctrl`` (the
    applied, possibly delayed, target), each world's (randomized) gains, and the
    servo's length and velocity (``q``, ``qd`` at gear 1).

  The servo terms are 0 on a model without servos. No term reads the OSL motors'
  actuator force, whose units differ between the models (the robot gears them).
  """

  scale: float = 1.0
  """A multiplier of every regularizer, for env stages
  (``rewards.mimic.params.regularizers.scale``)."""
  osl_action_rate: float = 0.0
  servo_action_rate: float = 0.0
  joint_limits: float = 0.0
  """Per radian of limit excess."""
  joint_limit_joints: tuple[str, ...] = LIMIT_JOINTS
  """The joints ``joint_limits`` watches (:data:`LIMIT_JOINTS` by default)."""
  servo_torque: float = 0.0
  servo_saturation: float = 0.0
  log_servo_terms: bool = False
  """Report the servo terms' costs, unweighted, as the diagnostics
  :data:`SERVO_LOG_DIAGNOSTICS` on every step, whatever their coefficients: a
  measurement only, which never enters the reward. A model without servos reports
  none. The robot CLF tasks turn it on."""


ROBOT_VDOT_WINDOW = 10
"""The robot model's paper-form V-dot window, in env steps (``ClfCfg.vdot_window``)."""


def clf_cfg(model: str) -> ClfCfg:
  """The CLF tasks' decrease condition: the paper form, with a 10-step V-dot window
  on the robot model (1 on the musculoskeletal one)."""
  return ClfCfg(vdot_window=ROBOT_VDOT_WINDOW if model == "robot" else 1)


def clf_regularizers(model: str) -> RegularizerCfg:
  """The CLF tasks' regularizers: the OSL action rate and the joint limits at 1,
  and on the robot model its servos' action rate at 1. The servo torque terms stay
  0 (opt-in), but the robot model logs their costs (``log_servo_terms``)."""
  return RegularizerCfg(
    osl_action_rate=1.0,
    servo_action_rate=1.0 if model == "robot" else 0.0,
    joint_limits=1.0,
    log_servo_terms=model == "robot",
  )


def _yaw(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _term_rows(action_manager: Any, name: str, device: str) -> torch.Tensor:
  """The action rows of action term ``name`` (none without the term)."""
  start = 0
  for term in action_manager.active_terms:
    dim = int(action_manager.get_term(term).action_dim)
    if term == name:
      return torch.arange(start, start + dim, device=device)
    start += dim
  return torch.zeros(0, dtype=torch.long, device=device)


def _position_servos(env: Any, robot: Any) -> tuple[torch.Tensor, str | None]:
  """The global actuator ids of the robot's position servos, and why the servo
  terms cannot use them (None when they can): each must be a plain PD servo,
  ``force = Kp ctrl - Kp length - Kd velocity`` at gear 1 with a force limit."""
  ids = [
    int(i)
    for act in getattr(robot, "actuators", None) or ()
    if isinstance(act, XmlActuator) and act.command_field == "position"
    for i in act.global_ctrl_ids.tolist()
  ]
  servos = torch.tensor(ids, dtype=torch.long, device=env.device)
  model = getattr(getattr(env, "sim", None), "mj_model", None)
  if model is None:
    return servos, None
  for i in ids:
    plain = (
      model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_NONE
      and model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
      and model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
      and bool(model.actuator_forcelimited[i])
      and model.actuator_forcerange[i, 1] > 0.0
      and model.actuator_gear[i, 0] == 1.0
    )
    if not plain:
      return servos, (
        f"Actuator {model.actuator(i).name!r} is not a plain PD servo (no dynamics, "
        "fixed gain, affine bias, gear 1, a positive force limit), which the servo "
        "regularizers assume."
      )
  return servos, None


def _muscle_act_ids(env: Any) -> torch.Tensor | None:
  """The activation-state indices of the model's muscles; None without any."""
  model = getattr(getattr(env, "sim", None), "mj_model", None)
  if model is None:
    return None
  ids = [
    int(model.actuator_actadr[i])
    for i in range(model.nu)
    if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
  ]
  return torch.tensor(ids, dtype=torch.long, device=env.device) if ids else None


class mimic_reward(ManagerTermBase):
  """musclemimic's imitation reward, clipped to be non-negative.

  Params:
    command_name: The reference command (a :class:`MimicReferenceSource`).
    weights: The seven ``*_w_sum`` weights, all required. ``rvel`` weighs the
      rotational and the linear relative-velocity kernels alike.
    action_out_of_bounds_coeff: musclemimic's default 0.01, active in every config.
    joint_torque_weights: ``{joint: w}`` adding ``-w * qfrc_actuator**2``.
    clf: A :class:`ClfCfg` to add the CLF terms, or None for plain MimicReward.
    regularizers: A :class:`RegularizerCfg`, or None for none.

  Diagnostics for the last step are kept in :attr:`components`, (B,) each.
  """

  LIVE_PARAMS: ClassVar[frozenset[str]] = frozenset(
    {
      *(f"clf.{name}" for name in ("w", "pen_w", "pen_scale", "tau")),
      "clf.decay_rate",
      "clf.decay_norm_rate",
      *(f"regularizers.{name}" for name in ("scale", *REGULARIZERS)),
    }
  )
  """The params the term reads on every step, as ``<param>.<field>``: the only ones
  an env stage may set (``rewards.<term>.params.<param>.<field>``, see
  ``osl_mimic.rl.runner.stage_accessors``). Every other param is read once, when
  the term is built."""

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
    # Each active regularizer's cost at its coefficient, before the scale: (B,),
    # or (B, J) per joint for the joint limits.
    self._reg_cache: dict[str, torch.Tensor] = {}

    self._clf: ClfCfg | None = params.get("clf")
    if self._clf is not None:
      self._setup_clf(self._clf, cmd, env)
    regularizers = params.get("regularizers")
    if regularizers is not None and not isinstance(regularizers, RegularizerCfg):
      raise TypeError(
        f"regularizers must be a RegularizerCfg or None, got {type(regularizers)}"
      )
    self._regularizers: RegularizerCfg | None = regularizers
    if regularizers is not None:
      self._setup_regularizers(regularizers, cmd, env)
    self._muscle_acts = _muscle_act_ids(env)

  def _setup_clf(
    self, clf: ClfCfg, cmd: MimicReferenceSource, env: ManagerBasedRlEnv
  ) -> None:
    unknown = set(clf.scales) - set(CLF_GROUPS)
    if unknown:
      raise ValueError(f"Unknown CLF groups {sorted(unknown)}; valid: {CLF_GROUPS}")
    if clf.tau <= 0.0 or clf.c_viol <= 0.0:
      raise ValueError("CLF tau and c_viol must be positive.")
    if clf.decay_form not in DECAY_FORMS:
      raise ValueError(
        f"Unknown CLF decay_form {clf.decay_form!r}; valid: {DECAY_FORMS}"
      )
    for name in ("pen_w", "pen_scale", "decay_rate"):
      value = float(getattr(clf, name))
      if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"CLF {name} must be finite and >= 0, got {value}")
    if not (math.isfinite(clf.decay_norm_rate) and clf.decay_norm_rate > 0.0):
      raise ValueError(f"CLF decay_norm_rate must be > 0, got {clf.decay_norm_rate}")
    window = clf.vdot_window
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
      raise ValueError(f"CLF vdot_window must be an int >= 1, got {window!r}")
    self._vdot_window = window
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
    # The last vdot_window values of V, oldest first, and per env the fresh steps
    # since its last episode start or reference jump (capped at the window).
    self._v_hist = torch.zeros(env.num_envs, window, device=env.device)
    self._v_since = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._clf_valid = torch.zeros(env.num_envs, device=env.device)

  def _setup_regularizers(
    self, cfg: RegularizerCfg, cmd: MimicReferenceSource, env: ManagerBasedRlEnv
  ) -> None:
    for name in ("scale", *REGULARIZERS):
      value = getattr(cfg, name)
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"regularizers.{name} must be a number, got {value!r}")
      if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"regularizers.{name} must be finite and >= 0, got {value}")
    device = env.device
    self._osl_rows = _term_rows(env.action_manager, OSL_ACTION_TERM, device)
    self._servo_rows = _term_rows(env.action_manager, SERVO_ACTION_TERM, device)
    names = list(cmd.robot.joint_names)
    joints = tuple(cfg.joint_limit_joints)
    unknown = [j for j in joints if j not in names]
    if unknown or len(set(joints)) != len(joints):
      raise ValueError(
        f"regularizers.joint_limit_joints must name distinct joints of the robot; "
        f"unknown {unknown} in {list(joints)}."
      )
    self._limit_joints = joints
    self._limit_ids = torch.tensor(
      [names.index(j) for j in joints], dtype=torch.long, device=device
    )
    self._servo_ids, self._servo_problem = _position_servos(env, cmd.robot)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    weights: dict[str, float],
    action_out_of_bounds_coeff: float = 0.01,
    joint_torque_weights: dict[str, float] | None = None,
    clf: ClfCfg | None = None,
    regularizers: RegularizerCfg | None = None,
  ) -> torch.Tensor:
    del weights, action_out_of_bounds_coeff, joint_torque_weights, clf, regularizers
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
    # excess over [-1, 1], summed and divided by the action dimension), with the
    # regularizers.
    action = env.action_manager.action
    excess = (-1.0 - action).clamp(min=0.0) + (action - 1.0).clamp(min=0.0)
    penalty = -self._oob_coeff * excess.square().sum(-1) / action.shape[-1]
    qfrc = None
    if self._torque_dofs.numel():
      qfrc = env.sim.data.qfrc_actuator[:, self._torque_dofs]
      penalty = penalty - (self._torque_weights * qfrc.square()).sum(-1)
    reg_total = None
    servo_log: dict[str, torch.Tensor] = {}
    if self._regularizers is not None:
      self._reg_cache, servo_log = self._regularizer_costs(env, cmd, action)
      scale = float(self._regularizers.scale)
      if self._reg_cache and scale != 0.0:
        reg_total = scale * _summed(self._reg_cache.values())
        penalty = penalty - reg_total
    penalty = penalty.clamp(min=-1.0)
    floored_penalty = penalty

    clf_terms: dict[str, torch.Tensor] = {}
    if self._clf is not None:
      clf_terms = self._clf_terms(env, cmd, ref, root_q, root_v, rpos, rangles)
      total = total + self._clf.w * clf_terms["level"]
      # Outside the -1 floor, so a saturated penalty cannot hide it.
      pen_w = self._clf.pen_scale * self._clf.pen_w
      penalty = penalty - pen_w * clf_terms["violation"]

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
    if clf_terms:
      recovering = (preclip < 0.0) & (clf_terms["vdot_fd"] < 0.0)
      self.components["clf_clipped_recovering"] = recovering.float()
    if self._regularizers is not None:
      zeros = torch.zeros_like(reward)
      for name in REGULARIZERS:
        cost = self._reg_cache.get(name)
        if cost is not None and cost.dim() > 1:
          cost = cost.sum(-1)
        self.components[f"reg_{name}"] = zeros if cost is None else cost
      self.components["reg_total"] = zeros if reg_total is None else reg_total
    self.components.update(servo_log)  # Log only: never in the reward.
    if self._muscle_acts is not None:
      act = env.sim.data.act[:, self._muscle_acts]
      self.components[ACTIVATION_DIAGNOSTIC] = act.square().mean(-1)
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

  ##
  # Regularizers.
  ##

  def _regularizer_costs(
    self, env: ManagerBasedRlEnv, cmd: MimicReferenceSource, action: torch.Tensor
  ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Each regularizer with a non-zero coefficient (and something to act on), at
    its coefficient and before ``scale``: (B,), the joint limits (B, J) per joint;
    and, with ``log_servo_terms``, the servo terms' unweighted costs (log only)."""
    cfg = self._regularizers
    assert cfg is not None
    costs: dict[str, torch.Tensor] = {}
    logged: dict[str, torch.Tensor] = {}
    for name, rows in (
      ("osl_action_rate", self._osl_rows),
      ("servo_action_rate", self._servo_rows),
    ):
      c = float(getattr(cfg, name))
      if c != 0.0 and rows.numel():
        costs[name] = c * self._action_rate(env, action, rows)
    c = float(cfg.joint_limits)
    if c != 0.0 and self._limit_ids.numel():
      costs["joint_limits"] = c * self._limit_excess(cmd)
    want = {n: float(getattr(cfg, n)) for n in ("servo_torque", "servo_saturation")}
    log = bool(cfg.log_servo_terms) and self._servo_problem is None
    if (any(want.values()) or log) and self._servo_ids.numel():
      if any(want.values()) and self._servo_problem is not None:
        raise RuntimeError(self._servo_problem)
      raw = self._servo_costs(env, dict.fromkeys(want, 1.0) if log else want)
      for name, value in raw.items():
        if want[name] != 0.0:
          costs[name] = want[name] * value
      if log:
        logged = {f"{name}_cost": value for name, value in raw.items()}
    return costs, logged

  @staticmethod
  def _action_rate(
    env: ManagerBasedRlEnv, action: torch.Tensor, rows: torch.Tensor
  ) -> torch.Tensor:
    """Mean over ``rows`` of the squared change of the clipped action; 0 on an
    episode's first step, whose previous action mjlab has zeroed."""
    now = action.index_select(-1, rows).clamp(-1.0, 1.0)
    before = env.action_manager.prev_action.index_select(-1, rows).clamp(-1.0, 1.0)
    rate = (now - before).square().mean(-1)
    return torch.where(env.episode_length_buf <= 1, torch.zeros_like(rate), rate)

  def _limit_excess(self, cmd: MimicReferenceSource) -> torch.Tensor:
    """(B, J) each watched joint's excess over its hard limits, in rad."""
    data = cmd.robot.data
    q = data.joint_pos.index_select(-1, self._limit_ids)
    limits = data.joint_pos_limits.index_select(-2, self._limit_ids)
    return (limits[..., 0] - q).clamp(min=0.0) + (q - limits[..., 1]).clamp(min=0.0)

  def _servo_costs(
    self, env: ManagerBasedRlEnv, want: dict[str, float]
  ) -> dict[str, torch.Tensor]:
    """The servo terms ``want`` asks for (non-zero), unweighted."""
    ids = self._servo_ids
    model, data = env.sim.model, env.sim.data
    tau_max = model.actuator_forcerange[..., ids, 1]
    out: dict[str, torch.Tensor] = {}
    if want["servo_torque"] != 0.0:
      ratio = data.actuator_force[:, ids] / tau_max
      out["servo_torque"] = ratio.square().mean(-1)
    if want["servo_saturation"] != 0.0:
      bias = model.actuator_biasprm[..., ids, :3]
      demand = (
        model.actuator_gainprm[..., ids, 0] * data.ctrl[:, ids]
        + bias[..., 0]
        + bias[..., 1] * data.actuator_length[:, ids]
        + bias[..., 2] * data.actuator_velocity[:, ids]
      )
      excess = (demand.abs() - tau_max).clamp(min=0.0) / tau_max
      out["servo_saturation"] = excess.mean(-1)
    return out

  ##
  # The CLF.
  ##

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
    # Bound whenever a group below reads them.
    v_rel = omega_rel = None
    if "rpos" in self._clf_groups or "rangles" in self._clf_groups:
      _, site_mat, site_ang, site_lin = cmd.site_state()
      v_rel, omega_rel = relative_site_velocities(site_mat, site_ang, site_lin)
    for group in self._clf_groups:
      if group == "rpos":
        es.append((rpos - ref.rpos).reshape(batch, -1))
        eds.append((cast(torch.Tensor, v_rel) - ref.v_rel).reshape(batch, -1))
      elif group == "rangles":
        es.append((rangles - ref.rangles).reshape(batch, -1))
        eds.append((cast(torch.Tensor, omega_rel) - ref.omega_rel).reshape(batch, -1))
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
    # velocity (or V) to difference against.
    first = (env.episode_length_buf <= 1) | cmd.reference_jumped
    edd = torch.where(
      first[:, None], torch.zeros_like(ed), (ed - self._ed_prev) * (tau / self._clf_dt)
    )
    vdot = (2.0 * inv_n) * (
      (ed * (p11 * e + p12 * ed)).sum(-1) + (edd * (p12 * e + p22 * ed)).sum(-1)
    )
    # The finite difference spans vdot_window steps, never across an episode start,
    # a reference jump or the term's first call: until the window holds that many
    # fresh steps since one, it is 0, as is the paper form's violation.
    window = self._vdot_window
    since = self._fresh_steps(first | (self._clf_valid == 0.0), self._v_since)
    fresh = since >= window
    vdot_fd = torch.where(
      fresh, (v - self._v_hist[:, 0]) / (window * self._clf_dt), torch.zeros_like(v)
    )
    violation = self._violation(v, vdot_fd, vdot)
    violation = violation * (~first).float() * self._clf_valid
    if self._clf.decay_form == "paper" and window > 1:
      violation = violation * fresh.float()

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
    history = self._v_hist[:, 1:].to(v.dtype)
    self._v_hist = torch.cat([history, v.detach()[:, None]], dim=-1)
    self._v_since = since
    self._clf_valid.fill_(1.0)
    return {
      "V": v,
      "V_norm": v_norm,
      "vdot": vdot,
      "cross": cross,
      "violation": violation,
      "level": level,
      "vdot_fd": vdot_fd,
      "violation_saturated": (violation >= 1.0).float(),
    }

  def _fresh_steps(self, event: torch.Tensor, since: torch.Tensor) -> torch.Tensor:
    """Per env, the fresh steps of history after this one, capped at the V-dot
    window: 0 where ``event`` (an episode start, a reference jump, the term's first
    call), one more than ``since`` elsewhere."""
    grown = (since + 1).clamp(max=self._vdot_window)
    return torch.where(event, torch.zeros_like(since), grown)

  def _violation(
    self, v: torch.Tensor, vdot_fd: torch.Tensor, vdot: torch.Tensor | None
  ) -> torch.Tensor:
    """The decrease condition's violation in [0, 1], before the first-step masks:
    the paper's from the finite-difference ``vdot_fd``, musclemimic's from the
    semi-analytic ``vdot`` (see :class:`ClfCfg`)."""
    clf = self._clf
    assert clf is not None
    if clf.decay_form == "paper":
      norm = clf.decay_norm_rate * self._v_max
      return ((vdot_fd + clf.decay_rate * v).clamp(min=0.0) / norm).clamp(0, 1)
    assert vdot is not None, "The musclemimic decay form needs the analytic V-dot."
    return ((vdot + self._alpha * v - self._eps_dead) / self._viol_norm).clamp(0, 1)


def _summed(values: Any) -> torch.Tensor:
  """The sum of per-env costs, (B,) or (B, J) summed over J."""
  total: torch.Tensor | None = None
  for value in values:
    value = value.sum(-1) if value.dim() > 1 else value
    total = value if total is None else total + value
  assert total is not None
  return total


def mimic_component(
  env: ManagerBasedRlEnv, name: str, reward_name: str = "mimic"
) -> torch.Tensor:
  """One per-env diagnostic of :class:`mimic_reward` from the current step."""
  term = env.reward_manager.get_term_cfg(reward_name).func
  components = cast(mimic_reward, term).components
  if name in components:
    return components[name]
  return torch.zeros(env.num_envs, device=env.device)
