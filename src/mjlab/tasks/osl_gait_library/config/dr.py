"""Domain randomization for the OSL gait-library tasks (stage 1, the unified policy).

The split follows the user's decision of 2026-11-06. Physics randomization changes
what the human experiences, so it is on from stage 1 and stays on when the human and
the OSL become separate actors in stage 2:

* physical properties, drawn once per env at startup: torso mass and COM, foot
  friction, joint armature, and each actuator kind's strength (servo stiffness and
  damping, OSL motor torque, muscle force);
* random pushes (training only);
* actuator delay, one draw per env per episode, separately for the OSL motors (the
  device's command latency) and the human actuators (neural latency).

Perception randomization (encoder bias, observation noise) models sensors, so it
belongs to the stage-2 OSL actor's own observations. Here it is a switch, off by
default: the unified policy also reads clean, redundant state (simulated site
kinematics, the reference) and would route around it.

The values start from sesteban's G1 unicycle base (``tracking/config/g1_custom``),
adjusted to a human body where noted.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache

import mujoco

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  OSL_TORQUE_JOINTS,
  Model,
  Socket,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.osl_gait_library import mdp
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode
from mjlab.utils.noise import UniformNoiseCfg

TORSO_BODY = "torso"
"""The thorax, 18.6 of the model's 82.3 kg: the analogue of G1's ``torso_link``."""
FOOT_GEOMS = (
  r"^(foot_col[134]_l|bofoot_col[12]_l|OSL_KA_L1_osl_foot_col[123]"
  r"|OSL_KA_L1_osl_foot_assembly_geom_1)$"
)
"""Every geom that carries a foot's floor contact: the intact foot's capsules, the OSL
foot's three capsules, and the OSL foot shell mesh, which also touches the floor (in
54-65% of the library's frames)."""
FOOT_PRIORITY = 1
"""Contact priority of the foot geoms, above the floor's 0. At equal priority MuJoCo
takes the larger of the two frictions, and the floor's is 1.0, so a foot draw below
1.0 would never reach the contact. With priority the foot's own parameters apply;
its solref, solimp and condim equal the floor's, so nominal contacts are unchanged."""
SLIDE_SCALE = 0.01
"""Observation noise on the coupled knees' slide joints (metres, moving by
millimetres) relative to the hinge values (radians)."""


@dataclass
class OslDrCfg:
  """Stage-1 randomization ranges. ``(1, 1)`` scales and zero ranges disable a term."""

  torso_com: tuple[float, float] = (-0.05, 0.05)
  """Torso COM offset along each body axis (m): G1's."""
  torso_mass: tuple[float, float] = (0.9, 1.1)
  """Torso mass scale, inertia scaled with it (a density change): G1's."""
  foot_friction: tuple[float, float] = (0.6, 1.2)
  """Absolute foot friction, one draw per env shared by both feet. G1's upper bound.
  The lower bound keeps every library gait feasible: the tightest arcs need 0.5 of
  the vertical force horizontally, and mjlab's pyramidal cone grants 0.71-1.0 times
  the coefficient depending on direction (level walking needs only 0.2-0.3). The
  model's own value, which the library was solved at, is 1.0."""
  joint_armature: tuple[float, float] = (0.95, 1.05)
  """Joint armature scale: G1's."""
  servo_kp: tuple[float, float] = (0.8, 1.2)
  servo_kd: tuple[float, float] = (0.8, 1.2)
  """Robot model's human servos: G1's stiffness and damping scales."""
  muscle_strength: tuple[float, float] = (0.8, 1.2)
  """MSK model's muscle peak-force scale: the human servos' +-20%, as strength."""
  osl_motor_strength: tuple[float, float] = (0.9, 1.1)
  """OSL motor torque scale: motor constant and drivetrain efficiency."""
  push_interval_s: tuple[float, float] = (2.0, 5.0)
  push_velocity: dict[str, tuple[float, float]] = field(
    default_factory=lambda: {
      "x": (-0.3, 0.3),
      "y": (-0.3, 0.3),
      "z": (-0.1, 0.1),
      "roll": (-0.3, 0.3),
      "pitch": (-0.3, 0.3),
      "yaw": (-0.45, 0.45),
    }
  )
  """Root velocity kicks: about 60% of mjlab tracking's (G1's) every 1-3 s. A 0.3 m/s
  change of an 82 kg body's velocity is a firm shove, which a person recovers from
  with a step."""
  osl_delay_s: tuple[float, float] = (0.0, 0.02)
  """OSL motor command latency. A placeholder until measured on the device; G1's
  0-40 ms models Unitree bus latency."""
  human_delay_s: tuple[float, float] = (0.02, 0.04)
  """Human actuator command delay. Monosynaptic stretch-reflex latencies run from
  about 20 ms (arm, thigh) to 40 ms (calf), the fastest feedback a person has; one
  lag per env, drawn over that span, delays every human actuator alike (per-region
  lags would need the actuator groups split by region). On the MSK model the
  muscles' activation dynamics (MuJoCo's, time constants 15 ms on and 50 ms off,
  scaled by activation) add their own lag. On the robot model it delays the servo
  targets only: each servo's PD loop still acts on the current state at once."""
  encoder_bias: bool = False
  """Perception switch, off in stage 1. Biases the OSL knee and ankle encoders as the
  actor reads them. Only the OSL joints: they are torque-driven, so the bias stays a
  sensing error, whereas on a position servo mjlab's ``JointPositionAction`` would
  also shift the setpoint."""
  encoder_bias_rad: float = 0.015
  """OSL encoder offset bound (rad)."""
  obs_noise: bool = False
  """Perception switch, off in stage 1: uniform noise on the actor's
  proprioception (root height and orientation, root and joint velocities, joint
  positions). The goal, touch and muscle terms stay clean."""


def _lags(seconds: tuple[float, float], timestep: float) -> tuple[int, int]:
  return round(seconds[0] / timestep), round(seconds[1] / timestep)


def _with_delays(
  entity: EntityCfg, osl: tuple[int, int], human: tuple[int, int]
) -> EntityCfg:
  """The entity with each actuator group delayed: one lag per env, drawn at every
  episode reset and held (``delay_hold_prob=1``; the buffer redraws after reset).

  mjlab writes the actuators once more after every reset, outside the physics
  substeps, so training's effective lag runs up to one substep (2 ms) short of the
  drawn one. The command fills the reset history with the start pose (see
  ``GaitLibraryCommand._start_episode``)."""
  assert entity.articulation is not None
  actuators = []
  for act in entity.articulation.actuators:
    lo, hi = osl if tuple(act.target_names_expr) == OSL_TORQUE_JOINTS else human
    actuators.append(
      replace(act, delay_min_lag=lo, delay_max_lag=hi, delay_hold_prob=1.0)
    )
  return replace(
    entity, articulation=replace(entity.articulation, actuators=tuple(actuators))
  )


def _foot_priority(
  spec_fn: Callable[[], mujoco.MjSpec],
) -> Callable[[], mujoco.MjSpec]:
  """``spec_fn`` with the foot geoms raised to ``FOOT_PRIORITY``."""

  def fn() -> mujoco.MjSpec:
    spec = spec_fn()
    for geom in spec.geoms:
      if re.fullmatch(FOOT_GEOMS, geom.name):
        geom.priority = FOOT_PRIORITY
    return spec

  return fn


@lru_cache(maxsize=None)
def _joint_split(
  model: Model, socket: Socket
) -> tuple[tuple[str, ...], frozenset[str]]:
  """The entity's non-root joints in order, and the slide joints among them."""
  spec = get_spec(model, socket)  # Held: joints read from a freed spec dangle.
  joints = [j for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
  names = tuple(str(j.name) for j in joints)
  slides = frozenset(
    str(j.name) for j in joints if j.type == mujoco.mjtJoint.mjJNT_SLIDE
  )
  return names, slides


def _per_joint(
  names: tuple[str, ...], slides: frozenset[str], value: float
) -> UniformNoiseCfg:
  magnitudes = tuple(value * SLIDE_SCALE if n in slides else value for n in names)
  return UniformNoiseCfg(n_min=tuple(-m for m in magnitudes), n_max=magnitudes)


def apply_osl_dr(
  cfg: ManagerBasedRlEnvCfg,
  model: Model,
  socket: Socket,
  mode: Mode,
  dr_cfg: OslDrCfg | None = None,
) -> None:
  """Add the randomization to a gait-library env config, in place.

  Training gets everything; play keeps the physical draws and delays but no pushes
  or noise (G1's convention, except that the delays stay: the human's range does not
  include zero, so a policy never saw zero delay). Validation gets only the delays,
  fixed at the middle of each range, on the nominal model. The foot contact priority
  is raised in every mode, so all three simulate the same contact model.
  """
  dr_cfg = dr_cfg or OslDrCfg()
  timestep = cfg.sim.mujoco.timestep
  osl_lags = _lags(dr_cfg.osl_delay_s, timestep)
  human_lags = _lags(dr_cfg.human_delay_s, timestep)
  if mode == "validation":
    osl_mid, human_mid = sum(osl_lags) // 2, sum(human_lags) // 2
    osl_lags, human_lags = (osl_mid, osl_mid), (human_mid, human_mid)
  entity = cfg.scene.entities["robot"]
  assert isinstance(entity, EntityCfg) and entity.spec_fn is not None
  entity = replace(entity, spec_fn=_foot_priority(entity.spec_fn))
  cfg.scene.entities["robot"] = _with_delays(entity, osl_lags, human_lags)
  if mode == "validation":
    return

  robot = "robot"
  events = cfg.events
  lo, hi = dr_cfg.torso_mass
  com = dr_cfg.torso_com
  # One event for mass and COM: pseudo_inertia rebuilds the body's inertial fields
  # from their defaults, so a separate COM event before it would be undone. Its t
  # parameters shift the COM along the body axes and keep the inertia about the COM.
  events["torso_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg(robot, body_names=(TORSO_BODY,)),
      # pseudo_inertia scales mass by exp(2 alpha).
      "alpha_range": (0.5 * math.log(lo), 0.5 * math.log(hi)),
      "t1_range": com,
      "t2_range": com,
      "t3_range": com,
    },
  )
  events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg(robot, geom_names=(FOOT_GEOMS,)),
      "operation": "abs",
      "ranges": dr_cfg.foot_friction,
      "shared_random": True,
    },
  )
  events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg(robot, joint_names=(".*",)),
      "operation": "scale",
      "ranges": dr_cfg.joint_armature,
    },
  )
  if model == "robot":
    events["servo_gains"] = EventTermCfg(
      mode="startup",
      func=mdp.servo_pd_gains,
      params={"kp_range": dr_cfg.servo_kp, "kd_range": dr_cfg.servo_kd},
    )
  else:
    events["muscle_strength"] = EventTermCfg(
      mode="startup",
      func=mdp.muscle_strength,
      params={"scale_range": dr_cfg.muscle_strength},
    )
  events["osl_motor_strength"] = EventTermCfg(
    mode="startup",
    func=mdp.motor_strength,
    params={"scale_range": dr_cfg.osl_motor_strength},
  )
  if mode == "train":
    events["push_robot"] = EventTermCfg(
      mode="interval",
      func=mdp.push_by_setting_velocity_marked,
      interval_range_s=dr_cfg.push_interval_s,
      params={"velocity_range": dict(dr_cfg.push_velocity), "command_name": "motion"},
    )

  actor = cfg.observations["actor"]
  if dr_cfg.encoder_bias:
    events["osl_encoder_bias"] = EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg(robot, joint_names=OSL_TORQUE_JOINTS),
        "bias_range": (-dr_cfg.encoder_bias_rad, dr_cfg.encoder_bias_rad),
      },
    )
    # The actor reads its joints through the biased encoders; the critic, which
    # stays privileged, keeps the true positions.
    actor.terms["joint_pos"].func = mdp.joint_pos_biased
  if dr_cfg.obs_noise and mode == "train":
    names, slides = _joint_split(model, socket)
    root = (0.01,) + (0.02,) * 4  # height (m), then the quaternion
    actor.terms["root_pos"].noise = UniformNoiseCfg(
      n_min=tuple(-v for v in root), n_max=root
    )
    vel = (0.1,) * 3 + (0.2,) * 3  # world linear (m/s), body angular (rad/s)
    actor.terms["root_vel"].noise = UniformNoiseCfg(
      n_min=tuple(-v for v in vel), n_max=vel
    )
    actor.terms["joint_pos"].noise = _per_joint(names, slides, 0.01)
    actor.terms["joint_vel"].noise = _per_joint(names, slides, 0.5)
    actor.enable_corruption = True
