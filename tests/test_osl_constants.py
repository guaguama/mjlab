"""Tests for osl_constants.py."""

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.osl_fullbody_robot import osl_constants
from mjlab.entity import Entity
from mjlab.utils.mujoco import detect_command_field, is_position_actuator


@pytest.fixture(scope="module")
def osl_entity() -> Entity:
  return Entity(osl_constants.get_osl_robot_cfg())


@pytest.fixture(scope="module")
def osl_model(osl_entity: Entity) -> mujoco.MjModel:
  return osl_entity.spec.compile()


@pytest.fixture(scope="module")
def source_model() -> mujoco.MjModel:
  """The vendored XML as musclemimic wrote it, with the actuators untouched."""
  spec = mujoco.MjSpec.from_file(str(osl_constants.OSL_XML))
  osl_constants._rebase_assets(spec)
  # The skybox is the model's only non-mesh asset file and is deliberately not
  # vendored, so it must go before this will compile. Nothing else is changed:
  # in particular the actuators are left in their original affine form.
  for texture in list(spec.textures):
    if texture.file:
      spec.delete(texture)
  return spec.compile()


def test_entity_shape(osl_entity: Entity, osl_model: mujoco.MjModel) -> None:
  assert osl_entity.num_joints == 73
  assert not osl_entity.is_fixed_base
  assert osl_entity.is_actuated
  assert osl_model.nu == 29
  assert osl_model.nbody == 106
  # All 44 coupled joints are driven by mjEQ_JOINT equality constraints.
  assert osl_model.neq == 44
  assert set(osl_model.eq_type) == {mujoco.mjtEq.mjEQ_JOINT}


def test_servo_reparameterization(osl_model: mujoco.MjModel) -> None:
  for name in osl_constants.SERVO_JOINTS:
    joint_id = mujoco.mj_name2id(osl_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    (actuator_id,) = np.flatnonzero(osl_model.actuator_trnid[:, 0] == joint_id)
    gainprm = osl_model.actuator_gainprm[actuator_id]
    biasprm = osl_model.actuator_biasprm[actuator_id]
    assert gainprm[0] == -biasprm[1]
    assert biasprm[0] == 0.0
    # The recovered ctrlrange reproduces the joint's own range. Not exact: the
    # source XML writes these as float32 text, so the round trip lands within
    # ~6e-6. Note np.allclose's default rtol=1e-5 would hide this.
    np.testing.assert_allclose(
      osl_model.actuator_ctrlrange[actuator_id],
      osl_model.jnt_range[joint_id],
      rtol=0,
      atol=1e-5,
    )


def test_command_fields_detected(osl_entity: Entity) -> None:
  """XmlActuatorCfg rejects <general> actuators it cannot classify."""
  fields = {a.name: detect_command_field(a) for a in osl_entity.spec.actuators}
  position = [n for n, f in fields.items() if f == "position"]
  effort = [n for n, f in fields.items() if f == "effort"]
  assert len(position) == 27
  assert len(effort) == 2
  for actuator in osl_entity.spec.actuators:
    if actuator.name in position:
      assert is_position_actuator(actuator)


def test_servo_force_law_unchanged(
  osl_model: mujoco.MjModel, source_model: mujoco.MjModel
) -> None:
  """The reparameterization changes command units, not physics.

  For each servo, the force the source model produces at a normalized command
  must equal the force the reparameterized model produces at the corresponding
  angle. This is the test that protects the physics rather than the bookkeeping.
  """
  rng = np.random.default_rng(0)
  for name in osl_constants.SERVO_JOINTS:
    joint_id = mujoco.mj_name2id(osl_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    (i,) = np.flatnonzero(osl_model.actuator_trnid[:, 0] == joint_id)
    src_gain = source_model.actuator_gainprm[i]
    src_bias = source_model.actuator_biasprm[i]
    dst_gain = osl_model.actuator_gainprm[i]
    dst_bias = osl_model.actuator_biasprm[i]
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
  osl_model: mujoco.MjModel, source_model: mujoco.MjModel
) -> None:
  """The OSL motors' +-2.88 ctrlrange must never be renormalized."""
  for name, gear in zip(osl_constants.OSL_TORQUE_JOINTS, (49.4, 58.4), strict=True):
    joint_id = mujoco.mj_name2id(osl_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    (i,) = np.flatnonzero(osl_model.actuator_trnid[:, 0] == joint_id)
    np.testing.assert_array_equal(osl_model.actuator_ctrlrange[i], (-2.88, 2.88))
    assert osl_model.actuator_gear[i, 0] == pytest.approx(gear)
    assert osl_model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
    np.testing.assert_array_equal(
      osl_model.actuator_gainprm[i], source_model.actuator_gainprm[i]
    )
    np.testing.assert_array_equal(
      osl_model.actuator_biasprm[i], source_model.actuator_biasprm[i]
    )


def test_scene_furniture_removed(osl_entity: Entity, osl_model: mujoco.MjModel) -> None:
  """mjlab owns the terrain, lighting and camera framing, not the entity."""
  assert osl_entity.spec.worldbody.geoms == []
  assert osl_entity.spec.worldbody.lights == []
  assert osl_model.nlight == 0
  texture_names = {t.name for t in osl_entity.spec.textures}
  assert "sky" not in texture_names
  assert "texplane" not in texture_names
  assert all(not t.file for t in osl_entity.spec.textures)
  assert "MatPlane" not in {m.name for m in osl_entity.spec.materials}
  # The robot's own skin textures and its body-attached cameras survive.
  assert osl_model.ntex == 3
  assert osl_model.ncam == 10


def test_collision_set(osl_model: mujoco.MjModel) -> None:
  collidable = np.flatnonzero(
    (osl_model.geom_contype != 0) | (osl_model.geom_conaffinity != 0)
  )
  # 14 capsules + 4 ellipsoids. No mesh or cylinder: CAPSULE/MESH and
  # CAPSULE/CYLINDER pairs have no multicontact support under MULTICCD, which
  # would silently reduce the prosthetic foot to a single contact point.
  assert len(collidable) == 18
  assert set(osl_model.geom_type[collidable]) == {
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID,
  }
  # mujoco_warp refuses non-zero margin under MULTICCD.
  assert float(np.max(osl_model.geom_margin[collidable])) == 0.0
  # Self collision is off by construction, as in the source model.
  assert set(osl_model.geom_contype[collidable]) == {0}
  for name in ("foot_col1_l", "OSL_KA_L1_osl_foot_col1"):
    i = mujoco.mj_name2id(osl_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert osl_model.geom_condim[i] == 3
    assert osl_model.geom_friction[i][0] == pytest.approx(0.6)


def test_initial_pose_is_on_equality_manifold(osl_model: mujoco.MjModel) -> None:
  """The 44 coupled joints must be consistent with the actuated ones.

  qpos0 (all joints zero) is off the manifold by 5.2e-02, which would put a
  constant bias into joint_pos_rel observations.
  """
  data = mujoco.MjData(osl_model)
  mujoco.mj_resetDataKeyframe(osl_model, data, 0)
  mujoco.mj_forward(osl_model, data)
  residual = max(
    (
      abs(data.efc_pos[i])
      for i in range(data.nefc)
      if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    ),
    default=0.0,
  )
  assert residual < 1e-3


def test_initial_pose_rests_on_the_ground(osl_model: mujoco.MjModel) -> None:
  data = mujoco.MjData(osl_model)
  mujoco.mj_resetDataKeyframe(osl_model, data, 0)
  mujoco.mj_forward(osl_model, data)
  collidable = np.flatnonzero(
    (osl_model.geom_contype != 0) | (osl_model.geom_conaffinity != 0)
  )
  lowest = data.geom_xpos[collidable, 2].min()
  assert 0.0 <= lowest < 0.02


def test_initial_pose_joint_names_cannot_alias(osl_model: mujoco.MjModel) -> None:
  """No joint name may prefix another.

  ``InitialStateCfg.joint_pos`` keys are resolved with an unanchored ``re.match``
  (``utils.string.resolve_expr``), so if one joint name were a prefix of another,
  this pose's 73 literal keys would silently assign to the wrong joints.
  """
  names = [
    mujoco.mj_id2name(osl_model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(osl_model.njnt)
    if osl_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  ]
  assert set(osl_constants.WALK_START_KEYFRAME.joint_pos or {}) == set(names)
  collisions = [(a, b) for a in names for b in names if a != b and b.startswith(a)]
  assert collisions == []
