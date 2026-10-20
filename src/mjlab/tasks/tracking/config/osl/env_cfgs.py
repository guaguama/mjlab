"""Full-body OSL walking tracking for the four v3 model variants.

Stock mjlab tracking behaviour and rewards, following the G1 task. The deviations
are the ones the model forces; each is commented where it is made.
"""

from copy import deepcopy
from dataclasses import fields

import mujoco

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  FULL_COLLISION,
  MUSCLE_TENDONS_EXPR,
  OSL_TORQUE_JOINTS,
  SERVO_JOINTS,
  Model,
  Socket,
  get_osl_msk_cfg,
  get_osl_robot_cfg,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import (
  JointEffortActionCfg,
  JointPositionActionCfg,
  TendonEffortActionCfg,
)
from mjlab.managers import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from .mdp import OslMotionCommandCfg

TRACKED_BODIES = (
  "Full Body",
  "pelvis",
  "lumbar1",
  "head",
  "humerus_l",
  "ulna_l",
  "lunate_l",
  "humerus_r",
  "ulna_r",
  "lunate_r",
  "femur_l",
  "tibia_l",
  "talus_l",
  "toes_l",
  "femur_r",
  "OSL_KA_L1_osl_knee_assembly",
  "OSL_KA_L1_osl_ankle_assembly",
  "OSL_KA_L1_osl_foot_assembly",
)


def _add_tracking_imu(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Add the pelvis IMU the stock observations read (G1 ships it in its XML)."""
  pelvis = spec.body("pelvis")
  assert pelvis is not None
  pelvis.add_site(name="tracking_imu")
  for name, kind in (
    ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
  ):
    spec.add_sensor(
      name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tracking_imu"
    )
  return spec


# Named per variant rather than lambdas, so a dumped EntityCfg names a real function.
def tracking_robot_spec() -> mujoco.MjSpec:
  return _add_tracking_imu(get_spec("robot", "rigid"))


def tracking_robot_compliant_spec() -> mujoco.MjSpec:
  return _add_tracking_imu(get_spec("robot", "compliant"))


def tracking_msk_spec() -> mujoco.MjSpec:
  return _add_tracking_imu(get_spec("msk", "rigid"))


def tracking_msk_compliant_spec() -> mujoco.MjSpec:
  return _add_tracking_imu(get_spec("msk", "compliant"))


_TRACKING_SPEC_FNS = {
  ("robot", "rigid"): tracking_robot_spec,
  ("robot", "compliant"): tracking_robot_compliant_spec,
  ("msk", "rigid"): tracking_msk_spec,
  ("msk", "compliant"): tracking_msk_compliant_spec,
}


def _entity_cfg(model: Model, socket: Socket) -> EntityCfg:
  cfg = get_osl_robot_cfg(socket) if model == "robot" else get_osl_msk_cfg(socket)
  cfg.spec_fn = _TRACKING_SPEC_FNS[(model, socket)]
  return cfg


def _servo_action_scale(spec: mujoco.MjSpec) -> dict[str, float]:
  """G1's convention, ``0.25 * effort_limit / stiffness``, per servo.

  Read from the reparameterized spec, where ``gainprm[0]`` is Kp. Spans 0.087 to
  0.31 rad across the 27 servos.
  """
  return {
    a.target: float(0.25 * a.forcerange[1] / a.gainprm[0])
    for a in spec.actuators
    if a.target in SERVO_JOINTS
  }


def osl_fullbody_tracking_env_cfg(
  model: Model = "robot",
  socket: Socket = "rigid",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an OSL full-body flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": _entity_cfg(model, socket)}
  spec = get_spec(model, socket)

  osl_effort = JointEffortActionCfg(
    entity_name="robot",
    actuator_names=OSL_TORQUE_JOINTS,
    preserve_order=True,
    # Raw torque command; the motors' +-2.88 ctrlrange is never renormalized.
    scale=1.0,
    offset=0.0,
  )
  if model == "robot":
    cfg.actions = {
      # Keeps the name the exporter looks up, although export is deferred for OSL.
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=SERVO_JOINTS,
        preserve_order=True,
        scale=_servo_action_scale(spec),
        use_default_offset=True,
      ),
      "osl_effort": osl_effort,
    }
  else:
    cfg.actions = {
      "osl_effort": osl_effort,
      # Muscle ctrlrange is [0, 1] and clamped, so a negative action is zero
      # excitation: the same effective mapping as musclemimic's [-1, 1] range.
      "muscles": TendonEffortActionCfg(
        entity_name="robot",
        actuator_names=(MUSCLE_TENDONS_EXPR,),
        scale=1.0,
        offset=0.0,
      ),
    }

  base = cfg.commands["motion"]
  assert isinstance(base, MotionCommandCfg)
  motion = OslMotionCommandCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
  motion.anchor_body_name = "pelvis"
  motion.body_names = TRACKED_BODIES
  cfg.commands["motion"] = motion

  # The model has no self-collision sensor, and self collision is off by
  # construction.
  cfg.rewards.pop("self_collisions")

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    FULL_COLLISION.geom_names_expr[0],
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)

  # The stock joint noise and encoder bias are radian magnitudes; the coupled knee
  # translations are slide joints in metres and move by millimetres, so they get
  # magnitudes 100x smaller.
  joints = [j for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
  slides = tuple(j.name for j in joints if j.type == mujoco.mjtJoint.mjJNT_SLIDE)
  hinges = tuple(j.name for j in joints if j.type != mujoco.mjtJoint.mjJNT_SLIDE)
  cfg.events["encoder_bias"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=hinges
  )
  cfg.events["translation_encoder_bias"] = deepcopy(cfg.events["encoder_bias"])
  cfg.events["translation_encoder_bias"].params.update(
    asset_cfg=SceneEntityCfg("robot", joint_names=slides), bias_range=(-0.0001, 0.0001)
  )
  for term in ("joint_pos", "joint_vel"):
    noise = cfg.observations["actor"].terms[term].noise
    assert isinstance(noise, UniformNoiseCfg)
    angular = float(noise.n_max)  # type: ignore[arg-type]
    magnitudes = tuple(angular * 0.01 if j.name in slides else angular for j in joints)
    cfg.observations["actor"].terms[term].noise = UniformNoiseCfg(
      n_min=tuple(-v for v in magnitudes), n_max=magnitudes
    )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "toes_l",
    "OSL_KA_L1_osl_foot_assembly",
    "lunate_l",
    "lunate_r",
  )

  cfg.viewer.body_name = "pelvis"

  # 2 ms is musclemimic's training timestep; the stock 5 ms is unsafe for the stiff
  # servos and for the muscles, whose damping mujoco_warp integrates explicitly.
  # Decimation 10 keeps step_dt at the 0.02 s the 50 Hz motion NPZ assumes.
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10
  # The 44 equality rows are unconditional, so the stock 250/35 budget leaves too
  # little room under RSI; overflow silently drops constraints.
  cfg.sim.njmax = 256
  cfg.sim.nconmax = 48

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion.pose_range = {}
    motion.velocity_range = {}

    motion.sampling_mode = "start"

  return cfg
