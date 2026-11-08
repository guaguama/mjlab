"""Full-body OSL walking tracking for the robot model (both socket variants).

Stock mjlab tracking behaviour and rewards, following the G1 task. The deviations
are the ones the model forces; each is commented where it is made. The
musculoskeletal model and musclemimic's training recipe live in ``tasks/osl_mimic``.
"""

from copy import deepcopy
from dataclasses import fields

import mujoco

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  CONTACT_CONE,
  CONTACT_IMPRATIO,
  FULL_COLLISION,
  OSL_TORQUE_JOINTS,
  SERVO_JOINTS,
  Socket,
  get_osl_robot_cfg,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from .mdp import OslMotionCommandCfg, bad_osl_anchor_ori

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

# 2 ms is musclemimic's training timestep; the stock 5 ms is unsafe for the stiff
# servos. Decimation 10 keeps step_dt at 0.02 s, so motion clips must be 50 Hz.
# The stock implicitfast integrator treats joint damping and servo Kd implicitly, so
# the compliant socket needs none of the Euler-damping toggle musclemimic uses.
TIMESTEP = 0.002
DECIMATION = 10


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


_TRACKING_SPEC_FNS = {
  "rigid": tracking_robot_spec,
  "compliant": tracking_robot_compliant_spec,
}


def _entity_cfg(socket: Socket) -> EntityCfg:
  cfg = get_osl_robot_cfg(socket)
  cfg.spec_fn = _TRACKING_SPEC_FNS[socket]
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
  socket: Socket = "rigid",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an OSL full-body (robot) flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": _entity_cfg(socket)}
  spec = get_spec("robot", socket)

  # The 46 explicit contact pairs (arms against thorax and pelvis, among others)
  # are active even though FULL_COLLISION gives every geom contype 0: explicit pairs
  # ignore contype/conaffinity. "Full Body" is the root, so its subtree is the
  # whole model. One history slot per physics substep, as G1 has.
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="Full Body", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="Full Body", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=DECIMATION,
    ),
  )

  cfg.actions = {
    # Keeps the name the exporter looks up, although export is deferred for OSL.
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=SERVO_JOINTS,
      preserve_order=True,
      scale=_servo_action_scale(spec),
      use_default_offset=True,
    ),
    "osl_effort": JointEffortActionCfg(
      entity_name="robot",
      actuator_names=OSL_TORQUE_JOINTS,
      preserve_order=True,
      # Raw torque command; the motors' +-2.88 ctrlrange is never renormalized.
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

  # The stock check compares the z component of projected gravity in the anchor
  # frame. The OSL pelvis frame is x forward, y up, z lateral, so that check never
  # fires on pitch and fires on roll only past about 53 degrees. This one compares
  # the whole gravity direction: 1 - cos(tilt error) > 0.5 is 60 degrees.
  cfg.terminations["anchor_ori"] = TerminationTermCfg(
    func=bad_osl_anchor_ori,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_name": "motion",
      "threshold": 0.5,
    },
  )
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "toes_l",
    "OSL_KA_L1_osl_foot_assembly",
    "lunate_l",
    "lunate_r",
  )

  cfg.viewer.body_name = "pelvis"

  cfg.sim.mujoco.timestep = TIMESTEP
  cfg.sim.mujoco.cone = CONTACT_CONE
  cfg.sim.mujoco.impratio = CONTACT_IMPRATIO
  cfg.decimation = DECIMATION
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
