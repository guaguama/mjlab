"""Tests for the four vendored OSL full-body variants (v3)."""

import filecmp
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.osl_fullbody import osl_constants
from mjlab.entity import Entity
from mjlab.utils.mujoco import detect_command_field, is_position_actuator

MUSCLEMIMIC_MODEL = (
  Path.home() / "src/00_AMBER/01_projpegleg/musclemimic/models/osl_fullbody"
)
Variant = tuple[osl_constants.Model, osl_constants.Socket]
VARIANTS = list(osl_constants.OSL_XMLS)
# (nq, number of non-free joints, nu) per variant.
DIMS = {
  ("robot", "rigid"): (80, 73, 29),
  ("robot", "compliant"): (84, 77, 29),
  ("msk", "rigid"): (80, 73, 341),
  ("msk", "compliant"): (84, 77, 341),
}


def _cfg(model: osl_constants.Model, socket: osl_constants.Socket):
  if model == "robot":
    return osl_constants.get_osl_robot_cfg(socket)
  return osl_constants.get_osl_msk_cfg(socket)


@pytest.fixture(scope="module", params=VARIANTS, ids=lambda v: "-".join(v))
def variant(request) -> Variant:
  return request.param


@pytest.fixture(scope="module")
def entity(variant: Variant) -> Entity:
  return Entity(_cfg(*variant))


@pytest.fixture(scope="module")
def model(entity: Entity) -> mujoco.MjModel:
  return entity.spec.compile()


@pytest.fixture(scope="module")
def source_model(variant: Variant) -> mujoco.MjModel:
  """The vendored XML as musclemimic wrote it, with the actuators untouched."""
  return mujoco.MjModel.from_xml_path(str(osl_constants.OSL_XMLS[variant]))


@pytest.fixture(scope="module")
def keyframe_data(model: mujoco.MjModel) -> mujoco.MjData:
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return data


def _actuator_for_joint(model: mujoco.MjModel, name: str) -> int:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  joint_actuated = model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
  (i,) = np.flatnonzero(joint_actuated & (model.actuator_trnid[:, 0] == joint_id))
  return int(i)


def test_vendored_tree_matches_musclemimic() -> None:
  """A re-import is a plain copy, so any drift from the source is a bug."""
  if not MUSCLEMIMIC_MODEL.exists():
    pytest.skip("musclemimic checkout not available")
  vendored = osl_constants.OSL_BODY_DIR.parent
  for sub in ("body", "meshes", "scene"):
    comparison = filecmp.dircmp(MUSCLEMIMIC_MODEL / sub, vendored / sub)
    assert not comparison.left_only, sub
    assert not comparison.right_only, sub
    _, mismatch, errors = filecmp.cmpfiles(
      MUSCLEMIMIC_MODEL / sub, vendored / sub, comparison.common_files, shallow=False
    )
    assert mismatch == [] and errors == [], sub


def test_entity_shape(variant: Variant, entity: Entity, model: mujoco.MjModel) -> None:
  nq, num_joints, nu = DIMS[variant]
  assert model.nq == nq
  assert entity.num_joints == num_joints
  assert model.nu == nu
  assert not entity.is_fixed_base
  assert model.nbody == 106
  # The coupled joints are driven by mjEQ_JOINT equality constraints.
  assert model.neq == 44
  assert set(model.eq_type) == {mujoco.mjtEq.mjEQ_JOINT}
  # Only Entity's own key: the source's placeholder keyframes are stripped.
  assert model.nkey == 1
  socket_joints = set(osl_constants.SOCKET_JOINTS) & set(entity.joint_names)
  assert len(socket_joints) == (4 if variant[1] == "compliant" else 0)


def test_command_fields_detected(variant: Variant, entity: Entity) -> None:
  """XmlActuatorCfg rejects <general> actuators it cannot classify."""
  fields = [detect_command_field(a) for a in entity.spec.actuators]
  if variant[0] == "robot":
    assert fields.count("position") == 27
    assert fields.count("effort") == 2
    for actuator in entity.spec.actuators:
      if actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE:
        assert is_position_actuator(actuator)
  else:
    assert fields.count("effort") == 341
    muscles = [
      a for a in entity.spec.actuators if a.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    ]
    assert len(muscles) == 339
    assert all(a.trntype == mujoco.mjtTrn.mjTRN_TENDON for a in muscles)
  wrapped = sum(len(a.target_names) for a in entity.actuators)
  assert wrapped == entity.spec.compile().nu


def test_servo_force_law_unchanged(
  variant: Variant, model: mujoco.MjModel, source_model: mujoco.MjModel
) -> None:
  """The reparameterization changes command units, not physics.

  For each servo, the force the source model produces at a normalized command
  must equal the force the reparameterized model produces at the corresponding
  angle. This is the test that protects the physics rather than the bookkeeping.
  """
  if variant[0] != "robot":
    pytest.skip("servos exist only in the robot model")
  rng = np.random.default_rng(0)
  for name in osl_constants.SERVO_JOINTS:
    i = _actuator_for_joint(model, name)
    joint_id = model.actuator_trnid[i, 0]
    src_gain = source_model.actuator_gainprm[i]
    src_bias = source_model.actuator_biasprm[i]
    dst_gain = model.actuator_gainprm[i]
    dst_bias = model.actuator_biasprm[i]
    assert dst_gain[0] == -dst_bias[1]
    assert dst_bias[0] == 0.0
    # The range is kept but no longer enforced (G1's convention).
    assert not model.actuator_ctrllimited[i]
    # The recovered ctrlrange reproduces the joint's own range. Not exact: the
    # source XML writes these as float32 text, so the round trip lands within
    # ~6e-6. Note np.allclose's default rtol=1e-5 would hide this.
    np.testing.assert_allclose(
      model.actuator_ctrlrange[i], model.jnt_range[joint_id], rtol=0, atol=1e-5
    )
    stiffness = -src_bias[1]
    midpoint = src_bias[0] / stiffness
    half_range = src_gain[0] / stiffness
    for ctrl in (-1.0, -0.3, 0.0, 0.7, 1.0):
      q, qd = rng.uniform(-1.0, 1.0, size=2)
      src_force = src_gain[0] * ctrl + src_bias[0] + src_bias[1] * q + src_bias[2] * qd
      target = midpoint + half_range * ctrl
      dst_force = (
        dst_gain[0] * target + dst_bias[0] + dst_bias[1] * q + dst_bias[2] * qd
      )
      np.testing.assert_allclose(dst_force, src_force, rtol=0, atol=1e-9)


def test_osl_torque_motors_untouched(
  model: mujoco.MjModel, source_model: mujoco.MjModel
) -> None:
  """The OSL motors' +-2.88 ctrlrange must never be renormalized.

  The robot writes the force scale as gear, the musculoskeletal model as gainprm
  with gear 1, so the product is what is checked.
  """
  for name, scale in zip(osl_constants.OSL_TORQUE_JOINTS, (49.4, 58.4), strict=True):
    i = _actuator_for_joint(model, name)
    j = _actuator_for_joint(source_model, name)
    np.testing.assert_array_equal(model.actuator_ctrlrange[i], (-2.88, 2.88))
    product = model.actuator_gear[i, 0] * model.actuator_gainprm[i, 0]
    assert product == pytest.approx(scale)
    assert model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
    np.testing.assert_array_equal(
      model.actuator_gainprm[i], source_model.actuator_gainprm[j]
    )
    np.testing.assert_array_equal(model.actuator_gear[i], source_model.actuator_gear[j])


def test_muscles_untouched(
  variant: Variant, model: mujoco.MjModel, source_model: mujoco.MjModel
) -> None:
  if variant[0] != "msk":
    pytest.skip("muscles exist only in the musculoskeletal model")
  muscles = np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
  source = np.flatnonzero(source_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
  assert len(muscles) == 339
  assert len(source) == 401
  # The 339 kept muscles are the source's non-finger muscles, parameters untouched.
  names = [source_model.actuator(int(i)).name for i in source]
  kept = [model.actuator(int(i)).name.split("/")[-1] for i in muscles]
  assert kept == [n for n in names if n not in osl_constants.FINGER_MUSCLES]
  matched = [int(source[names.index(n)]) for n in kept]
  for attr in ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm"):
    np.testing.assert_array_equal(
      getattr(model, attr)[muscles], getattr(source_model, attr)[matched]
    )


def test_scene_furniture_removed(entity: Entity, model: mujoco.MjModel) -> None:
  """mjlab owns the terrain, lighting and camera framing, not the entity."""
  assert entity.spec.worldbody.geoms == []
  assert entity.spec.worldbody.lights == []
  assert model.nlight == 0
  texture_names = {t.name for t in entity.spec.textures}
  assert "sky" not in texture_names
  assert "texplane" not in texture_names
  assert all(not t.file for t in entity.spec.textures)
  assert "MatPlane" not in {m.name for m in entity.spec.materials}
  # The model's own skin textures and its body-attached cameras survive.
  assert model.ntex == 3
  assert model.ncam == 10


def test_collision_set(model: mujoco.MjModel) -> None:
  collidable = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
  # 14 capsules + 4 ellipsoids. No mesh or cylinder: CAPSULE/MESH and
  # CAPSULE/CYLINDER pairs have no multicontact support under MULTICCD, which
  # would silently reduce the prosthetic foot to a single contact point.
  assert len(collidable) == 18
  assert set(model.geom_type[collidable]) == {
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID,
  }
  # mujoco_warp refuses non-zero margin under MULTICCD.
  assert float(np.max(model.geom_margin[collidable])) == 0.0
  # Self collision is off by construction, as in the source model.
  assert set(model.geom_contype[collidable]) == {0}
  for name in ("foot_col1_l", "OSL_KA_L1_osl_foot_col1"):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert model.geom_condim[i] == 3
    assert model.geom_friction[i][0] == pytest.approx(0.6)


def test_initial_pose_is_on_equality_manifold(keyframe_data: mujoco.MjData) -> None:
  """The 44 coupled joints must be consistent with the independent ones.

  The v3 clip's followers are recomputed exactly from the v3 couplers, so frame 0
  sits on the manifold to float precision (6.7e-07, against 4.3e-04 for v2).
  """
  data = keyframe_data
  residual = max(
    abs(data.efc_pos[i])
    for i in range(data.nefc)
    if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
  )
  assert residual < 1e-5


def test_initial_pose_rests_on_the_ground(
  model: mujoco.MjModel, keyframe_data: mujoco.MjData
) -> None:
  collidable = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
  lowest = keyframe_data.geom_xpos[collidable, 2].min()
  assert 0.0 <= lowest < 0.02


def test_initial_pose_joint_names_cannot_alias(entity: Entity) -> None:
  """No pose key may prefix another joint name.

  ``InitialStateCfg.joint_pos`` keys are resolved with an unanchored ``re.match``
  (``utils.string.resolve_expr``), so a key that prefixes another joint's name
  would silently assign to that joint too. Some finger joints do prefix each other
  (``cmc_flexion``/``cmc_flexion_l``), which is harmless only while the pose names
  no finger joint.
  """
  names = list(entity.joint_names)
  pose = set(osl_constants.WALK_START_KEYFRAME.joint_pos or {})
  assert pose <= set(names)
  assert len(pose) == 73
  collisions = [(a, b) for a in pose for b in names if a != b and b.startswith(a)]
  assert collisions == []


@pytest.mark.parametrize("socket", ["rigid", "compliant"])
def test_msk_fingerless_skeleton_matches_robot(socket: osl_constants.Socket) -> None:
  """musclemimic's finger deletion leaves exactly the robot's joints, in order."""
  msk, robot = (
    osl_constants.get_spec("msk", socket),
    osl_constants.get_spec("robot", socket),
  )
  msk_joints = [j.name for j in msk.joints]
  robot_joints = [j.name for j in robot.joints]
  assert msk_joints == robot_joints
  compiled = msk.compile()
  assert (compiled.nq, compiled.nu, compiled.na) == (DIMS[("msk", socket)][0], 341, 339)


@pytest.mark.parametrize("socket", ["rigid", "compliant"])
def test_msk_fingers_opt_in(socket: osl_constants.Socket) -> None:
  spec = osl_constants.get_spec("msk", socket, fingers=True)
  model = spec.compile()
  assert model.nu == 403
  assert model.nq == (120 if socket == "rigid" else 124)
  names = {j.name for j in spec.joints}
  assert set(osl_constants.FINGER_JOINTS) <= names


def test_robot_has_no_fingers_to_keep() -> None:
  with pytest.raises(ValueError, match="fingers"):
    osl_constants.get_spec("robot", "rigid", fingers=True)
