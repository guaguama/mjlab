"""OSL full-body model constants (v3).

A right transfemoral amputee: the left leg is intact, and the right leg below the
femur is an Open-Source Leg (OSL) prosthesis. This is musclemimic's ``osl_fullbody``
v3, the myo_sim 0.2.3 MyoAssist export with the OSL tibial pylon lengthened by
22.5 mm so the prosthetic leg matches the intact one.

Four variants are vendored, selected by ``model`` and ``socket``:

* ``model="robot"``: the joint-actuated twin, fingers removed. 27 position servos
  plus the 2 OSL torque motors. nq 80 (rigid) / 84 (compliant).
* ``model="msk"``: the musculoskeletal model, fingers kept. 401 Hill-type muscles on
  spatial tendons plus the same 2 OSL torque motors. nq 120 / 124.
* ``socket="compliant"`` adds the four socket DoFs (``socket_piston``,
  ``socket_rotation_1..3``) between the residual femur and the prosthesis.

In every variant 44 of the non-free joints are driven by ``mjEQ_JOINT`` equality
constraints. The vendored ``xmls/{body,meshes,scene}`` tree is a **byte-identical**
copy of musclemimic's ``models/osl_fullbody/``, laid out the same way, so both XML
forms resolve their assets unchanged and a re-import is ``cp -r``. Every adaptation
mjlab needs is applied programmatically in :func:`get_spec`. Never hand-edit the
XMLs; regenerate them in musclemimic.

.. warning::
   ``XmlActuatorCfg`` inherits ``armature``, ``frictionloss`` and ``viscous_damping``
   from ``ActuatorCfg``, but ``xml_actuator.py`` never reads them -- only the builtin
   and PD actuators do. Setting them here is a silent no-op, and there is no
   ``effort_limit`` field either. The model's own ``<default>`` classes supply all of
   these. Any change to servo or muscle physics means regenerating the MJCF in
   musclemimic.
"""

from pathlib import Path
from typing import Literal

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

Model = Literal["robot", "msk"]
Socket = Literal["rigid", "compliant"]

##
# MJCF and assets.
##

OSL_BODY_DIR: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "osl_fullbody" / "xmls" / "body"
)

OSL_XMLS: dict[tuple[Model, Socket], Path] = {
  ("robot", "rigid"): OSL_BODY_DIR / "osl_fullbody_robot.xml",
  ("robot", "compliant"): OSL_BODY_DIR / "osl_fullbody_robot_compliant_socket.xml",
  ("msk", "rigid"): OSL_BODY_DIR / "osl_fullbody.xml",
  ("msk", "compliant"): OSL_BODY_DIR / "osl_fullbody_compliant_socket.xml",
}
for _path in OSL_XMLS.values():
  assert _path.exists(), _path

# Joints driven by the 27 position servos, in the robot XML's actuator order.
SERVO_JOINTS: tuple[str, ...] = (
  "flex_extension",
  "lat_bending",
  "axial_rotation",
  "elv_angle_r",
  "shoulder_elv_r",
  "shoulder_rot_r",
  "elbow_flexion_r",
  "pro_sup_r",
  "deviation_r",
  "flexion_r",
  "elv_angle_l",
  "shoulder_elv_l",
  "shoulder_rot_l",
  "elbow_flexion_l",
  "pro_sup_l",
  "deviation_l",
  "flexion_l",
  "hip_flexion_r",
  "hip_adduction_r",
  "hip_rotation_r",
  "hip_flexion_l",
  "hip_adduction_l",
  "hip_rotation_l",
  "knee_angle_l",
  "ankle_angle_l",
  "subtalar_angle_l",
  "mtp_angle_l",
)

# Joints driven by the 2 OSL torque motors, in both models. Their ctrlrange of
# +-2.88 and their force scale of 49.4 / 58.4 are load-bearing and must never be
# renormalized. The robot writes the scale as ``gear``, the musculoskeletal model
# as ``gainprm[0]`` with gear 1; force is their product either way.
OSL_TORQUE_JOINTS: tuple[str, ...] = (
  "OSL_KA_L1_osl_knee_angle_r",
  "OSL_KA_L1_osl_ankle_angle_r",
)

SOCKET_JOINTS: tuple[str, ...] = (
  "socket_piston",
  "socket_rotation_1",
  "socket_rotation_2",
  "socket_rotation_3",
)


def _strip_scene(spec: mujoco.MjSpec) -> None:
  """Remove everything that is scene furniture rather than robot.

  The source MJCFs are standalone simulation models: they ship (or, for the
  musculoskeletal model, ``<include>``) a ground plane, a directional light, a
  skybox and viewer settings. mjlab owns all of those, so an entity carrying its own
  would duplicate them -- most damagingly the floor, which would sit coincident with
  the terrain plane and double every foot contact.
  """
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.worldbody.lights):
    spec.delete(light)
  # ``texplane``/``MatPlane`` exist only to texture the floor; ``sky`` is the skybox
  # and the model's only non-mesh asset file.
  for texture in list(spec.textures):
    if texture.name in ("sky", "texplane"):
      spec.delete(texture)
  for material in list(spec.materials):
    if material.name == "MatPlane":
      spec.delete(material)
  # Viewer framing and render settings are mutated back to MjSpec defaults rather
  # than deleted, since <visual> is a settings block and not a list of elements.
  defaults = mujoco.MjSpec()
  spec.visual.global_.azimuth = defaults.visual.global_.azimuth
  spec.visual.global_.elevation = defaults.visual.global_.elevation
  spec.visual.headlight.ambient = defaults.visual.headlight.ambient
  spec.visual.headlight.diffuse = defaults.visual.headlight.diffuse
  spec.visual.headlight.specular = defaults.visual.headlight.specular
  spec.visual.rgba.haze = defaults.visual.rgba.haze


def _reparameterize_servos(spec: mujoco.MjSpec) -> None:
  """Rewrite the robot's 27 affine ``<general>`` servos as true position actuators.

  musclemimic encodes each servo with a normalized command: ``ctrlrange=(-1, 1)``
  spanning the joint's travel, with ``gainprm[0] = Kp * half_range`` and
  ``biasprm = (Kp * midpoint, -Kp, -Kd)``. mjlab assumes throughout that a position
  actuator's ``ctrl`` is a joint angle in radians -- ``JointPositionAction`` computes
  ``ctrl = action * scale + default_joint_pos`` -- and ``is_position_actuator``
  requires ``gainprm[0] == -biasprm[1]``, which the normalized form fails.

  This is a pure change of command units, not of physics::

      before:  F = Kp * (midpoint + half_range * ctrl - q) - Kd * qdot
      after:   F = Kp * (ctrl - q) - Kd * qdot

  Kp, Kd, ``forcerange`` and the reachable target set are all unchanged; only the
  units the policy commands in change, from a fraction of travel to radians. The
  recovered ``ctrlrange`` reproduces the joint's own range.

  It also makes ``rl.exporter_utils.get_base_metadata`` correct for these actuators,
  since it reads ``gainprm[0]`` as stiffness -- which in the normalized form would
  report ``Kp * half_range``.

  Only affine-bias actuators are touched: the OSL torque motors (``biastype=NONE``)
  and the muscles (``biastype=MUSCLE``) pass through, so this is a no-op on the
  musculoskeletal model.
  """
  for actuator in spec.actuators:
    if actuator.biastype != mujoco.mjtBias.mjBIAS_AFFINE:
      continue
    stiffness = -actuator.biasprm[1]
    damping = -actuator.biasprm[2]
    midpoint = actuator.biasprm[0] / stiffness
    half_range = actuator.gainprm[0] / stiffness
    actuator.gainprm[0] = stiffness
    actuator.biasprm[0] = 0.0
    actuator.biasprm[1] = -stiffness
    actuator.biasprm[2] = -damping
    actuator.ctrlrange = (midpoint - half_range, midpoint + half_range)


def _strip_keyframes(spec: mujoco.MjSpec) -> None:
  """Drop the musculoskeletal model's five placeholder keyframes.

  ``stand``, ``walk_left``, ``walk_right``, ``squat`` and ``lunge`` float 7-39 cm
  above the floor, and ``squat``/``lunge`` sit ~1 rad off the equality manifold, so
  none is a usable initial state. Entity adds its own ``init_state`` key; leaving
  these would put that key at index 5 while index 0 stays a floating placeholder.
  ``<size nkey="5"/>`` is cleared too, or compile pads the model back to five empty
  keys. Deleting keyframes is safe; deleting joints from a spec with keyframes is not.
  """
  for key in list(spec.keys):
    spec.delete(key)
  spec.nkey = 0


def get_spec(model: Model = "robot", socket: Socket = "rigid") -> mujoco.MjSpec:
  # Load by absolute path: MuJoCo resolves the musculoskeletal model's nested
  # include and skybox against the main file's directory.
  spec = mujoco.MjSpec.from_file(str(OSL_XMLS[(model, socket)]))
  _strip_scene(spec)
  _reparameterize_servos(spec)
  _strip_keyframes(spec)
  return spec


# Named per variant rather than lambdas, so a dumped EntityCfg names a real function.
def get_robot_spec() -> mujoco.MjSpec:
  return get_spec("robot", "rigid")


def get_robot_compliant_spec() -> mujoco.MjSpec:
  return get_spec("robot", "compliant")


def get_msk_spec() -> mujoco.MjSpec:
  return get_spec("msk", "rigid")


def get_msk_compliant_spec() -> mujoco.MjSpec:
  return get_spec("msk", "compliant")


_SPEC_FNS = {
  ("robot", "rigid"): get_robot_spec,
  ("robot", "compliant"): get_robot_compliant_spec,
  ("msk", "rigid"): get_msk_spec,
  ("msk", "compliant"): get_msk_compliant_spec,
}


##
# Keyframe config.
##

# Frame 0 of the v3 reference walking clip (clipper's
# ``WalkingStraightForwards03_poses_crop0-521_z-0.0016_ankle-7_bioankle-5``, whose 44
# coupled joints are recomputed exactly from the v3 couplers). It is mid-stride rather
# than a symmetric stand; tracking does RSI from the motion anyway, so this mainly
# serves the zero-action rollout and the viewer. The clip names only the robot's 73
# joints, so the compliant socket joints and the musculoskeletal model's 40 finger
# joints take 0 -- the neutral socket and inside every finger joint's range.
WALK_START_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.948891),
  rot=(0.697515, -0.038748, 0.045989, -0.714042),
  joint_pos={
    "flex_extension": -0.048864,
    "lat_bending": -0.002068,
    "axial_rotation": -0.028541,
    "Abs_t1": 0.0,
    "Abs_t2": 0.0,
    "Abs_r3": -0.031414,
    "L4_L5_FE": -0.00904,
    "L4_L5_LB": -0.000375,
    "L4_L5_AR": -0.001078,
    "L3_L4_FE": -0.009968,
    "L3_L4_LB": -0.000507,
    "L3_L4_AR": -0.001078,
    "L2_L3_FE": -0.011287,
    "L2_L3_LB": -0.000517,
    "L2_L3_AR": -0.000888,
    "L1_L2_FE": -0.01246,
    "L1_L2_LB": -0.000389,
    "L1_L2_AR": -0.000825,
    "sternoclavicular_r2_r": -0.039061,
    "sternoclavicular_r3_r": 0.016545,
    "unrotscap_r3_r": -0.016545,
    "unrotscap_r2_r": 0.039061,
    "acromioclavicular_r2_r": -0.007909,
    "acromioclavicular_r3_r": 0.063918,
    "acromioclavicular_r1_r": 0.028731,
    "unrothum_r1_r": -0.028731,
    "unrothum_r3_r": -0.063918,
    "unrothum_r2_r": 0.007909,
    "elv_angle_r": 0.29269,
    "shoulder_elv_r": 0.16141,
    "shoulder1_r2_r": -0.29269,
    "shoulder_rot_r": 0.212504,
    "elbow_flexion_r": 0.399311,
    "pro_sup_r": 0.115771,
    "deviation_r": 0.040524,
    "flexion_r": 0.094928,
    "sternoclavicular_r2_l": -0.035854,
    "sternoclavicular_r3_l": 0.015186,
    "unrotscap_r3_l": -0.015186,
    "unrotscap_r2_l": 0.035854,
    "acromioclavicular_r2_l": -0.00726,
    "acromioclavicular_r3_l": 0.05867,
    "acromioclavicular_r1_l": 0.026372,
    "unrothum_r1_l": -0.026372,
    "unrothum_r3_l": -0.05867,
    "unrothum_r2_l": 0.00726,
    "elv_angle_l": 0.057378,
    "shoulder_elv_l": 0.148157,
    "shoulder1_r2_l": -0.057378,
    "shoulder_rot_l": 0.177591,
    "elbow_flexion_l": 0.389324,
    "pro_sup_l": 0.075838,
    "deviation_l": 0.060494,
    "flexion_l": 0.089487,
    "hip_flexion_r": -0.133701,
    "hip_adduction_r": -0.000223,
    "hip_rotation_r": 0.001957,
    "OSL_KA_L1_osl_knee_angle_r": 0.09376,
    "OSL_KA_L1_osl_ankle_angle_r": 0.103797,
    "hip_flexion_l": -0.210351,
    "hip_adduction_l": 0.018941,
    "hip_rotation_l": 0.002122,
    "knee_angle_translation2_l": 0.0,
    "knee_angle_translation1_l": 0.0,
    "knee_angle_l": -0.0,
    "knee_angle_rotation2_l": -0.0,
    "knee_angle_rotation3_l": 0.0,
    "ankle_angle_l": 0.101506,
    "subtalar_angle_l": -0.000561,
    "mtp_angle_l": 0.039119,
    "knee_angle_beta_translation2_l": -0.010828,
    "knee_angle_beta_translation1_l": 0.052419,
    "knee_angle_beta_rotation1_l": 0.010506,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# Contact geoms are named ``*_col*`` in this model, not G1's ``*_collision``. The
# same 18 geoms exist in all four variants.
#
# Two things are deliberate here. First, ``margin`` is zeroed: every geom in the
# source carries ``margin=0.001``, and mujoco_warp refuses non-zero margin under
# MULTICCD outright (``NotImplementedError`` from ``put_model``). Contacts therefore
# no longer form 1 mm early, which is a small but real difference from musclemimic.
# Second, only primitives are kept -- the OSL mesh and cylinder geoms are excluded
# because CAPSULE/MESH and CAPSULE/CYLINDER pairs have no multicontact support, so
# including them would silently reduce the prosthetic foot to a single contact point.
# The three ``OSL_KA_L1_osl_foot_col*`` capsules are its intended contact surface.
#
# contype=0/conaffinity=1 follows mjlab's idiom: the geoms collide with the terrain
# plane but not with each other, matching the source model, which disables self
# collision by construction.
_FOOT_GEOMS = r"^(foot_col[134]_l|bofoot_col[12]_l|OSL_KA_L1_osl_foot_col[123])$"
_BODY_GEOMS = (
  r"^(tibia[12]_col_l|pelvis_col_[rl]|thorax_coll[123]|femur1_col_r|femur[12]_col_l)$"
)

# Feet plus the shank/trunk/thigh geoms that let a fall register.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_GEOMS, _BODY_GEOMS),
  contype=0,
  conaffinity=1,
  condim={_FOOT_GEOMS: 3, ".*": 1},
  priority={_FOOT_GEOMS: 1, ".*": 0},
  friction={_FOOT_GEOMS: (0.6,)},
  margin=0.0,
)

# Feet only. Everything else is disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_GEOMS,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  margin=0.0,
)


##
# Final config.
##

# One group per command field: XmlActuatorCfg raises on mixed types. Each is passed
# explicitly rather than auto-detected, so a regenerated model that changes an
# actuator form fails loudly at entity build.
_SERVOS = XmlActuatorCfg(target_names_expr=SERVO_JOINTS, command_field="position")
_OSL_MOTORS = XmlActuatorCfg(
  target_names_expr=OSL_TORQUE_JOINTS, command_field="effort"
)
# The 401 muscle tendons are ``<muscle>_tendon`` (``_tendon_l`` on 63 left-arm
# muscles). The pattern leaves out the 8 passive shoulder ligaments, which carry no
# actuator, and matches no joint or site, which Entity would warn about.
MUSCLE_TENDONS_EXPR = r"^.*_tendon(_l)?$"
_MUSCLES = XmlActuatorCfg(
  target_names_expr=(MUSCLE_TENDONS_EXPR,),
  transmission_type=TransmissionType.TENDON,
  command_field="effort",
)

# Not G1's 0.9: at that factor the v3 reference walking clip itself violates the
# soft limits on 31 of 73 joints, worst 0.108 rad on the OSL knee, which would put a
# large constant joint-limit penalty on perfect tracking. At 1.0 only 8 joints
# violate, by at most 3.1 mrad.
_SOFT_JOINT_POS_LIMIT_FACTOR = 1.0


def get_osl_robot_cfg(socket: Socket = "rigid") -> EntityCfg:
  """Get a fresh OSL full-body robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=WALK_START_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=_SPEC_FNS[("robot", socket)],
    articulation=EntityArticulationInfoCfg(
      actuators=(_SERVOS, _OSL_MOTORS),
      soft_joint_pos_limit_factor=_SOFT_JOINT_POS_LIMIT_FACTOR,
    ),
  )


def get_osl_msk_cfg(socket: Socket = "rigid") -> EntityCfg:
  """Get a fresh OSL full-body musculoskeletal configuration instance."""
  return EntityCfg(
    init_state=WALK_START_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=_SPEC_FNS[("msk", socket)],
    articulation=EntityArticulationInfoCfg(
      actuators=(_OSL_MOTORS, _MUSCLES),
      soft_joint_pos_limit_factor=_SOFT_JOINT_POS_LIMIT_FACTOR,
    ),
  )


if __name__ == "__main__":
  import sys

  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  model = sys.argv[1] if len(sys.argv) > 1 else "robot"
  socket = sys.argv[2] if len(sys.argv) > 2 else "rigid"
  get_cfg = get_osl_msk_cfg if model == "msk" else get_osl_robot_cfg
  robot = Entity(get_cfg(socket))  # type: ignore[arg-type]

  viewer.launch(robot.spec.compile())
