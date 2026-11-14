"""Tests for the four vendored OSL full-body variants (v3)."""

import filecmp
import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
import pytest

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.asset_zoo.robots.osl_fullbody import osl_constants
from mjlab.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg
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


def test_xml_contacts_drop_the_shell_and_socket_pelvis(variant: Variant) -> None:
  """With the model's own contacts (``collision="xml"``), the OSL foot touches the
  floor only through its capsules, and the socket never touches the pelvis."""
  model = osl_constants.get_spec(*variant).compile()
  geom = model.geom(osl_constants.OSL_FOOT_SHELL_GEOM)
  assert (int(geom.contype[0]), int(geom.conaffinity[0])) == (0, 0)
  for i in range(3):
    capsule = model.geom(f"OSL_KA_L1_osl_foot_col{i + 1}")
    assert int(capsule.contype[0]) == 1
  pelvis = model.body(osl_constants.PELVIS_BODY).id
  socket = model.body(osl_constants.OSL_SOCKET_BODY).id
  signatures = set(model.exclude_signature.tolist())
  assert (pelvis << 16) + socket in signatures or (socket << 16) + pelvis in signatures
  # The socket body carries nothing but the socket mesh, so the exclude drops no
  # other contact.
  assert int(model.body_geomnum[socket]) == 1


def test_every_osl_task_uses_the_contact_solver_settings() -> None:
  """mj-nlp's elliptic cone and impratio, in training and play, for all 46 tasks."""
  tasks = [task for task in list_tasks() if "OSL" in task]
  assert len(tasks) == 46
  for task in tasks:
    for play in (False, True):
      sim = load_env_cfg(task, play=play).sim.mujoco
      assert (sim.cone, sim.impratio) == ("elliptic", 3.0), task


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


##
# The OSL foot's touch sites.
##

# Each version's boxes as the briefs tabulate them: centre and box half-sizes in
# the site's body frame. "original" is the XML's; the user approved
# "heel_extended" in a viewer and asked for "feet_extended" ("just extend
# slightly"). r_toes is the XML's in every version.
R_FOOT, L_FOOT, L_TOES = "OSL_KA_L1_r_foot_touch", "foot_touch_l", "toes_touch_l"
XML_R_FOOT = ((0.025, -0.04, 0.0), (0.05, 0.03, 0.05))
HEEL_R_FOOT = ((0.01, -0.04, 0.0), (0.065, 0.03, 0.05))
XML_L_FOOT = ((0.09, -0.01, 0.0), (0.1, 0.01, 0.055))
LOW_L_FOOT = ((0.09, -0.014, 0.0), (0.1, 0.014, 0.055))
XML_L_TOES = ((0.0275, -0.01, 0.0), (0.04, 0.01, 0.0675))
LOW_L_TOES = ((0.0275, -0.014, 0.0), (0.04, 0.014, 0.0675))
TOUCH_BOXES = {
  "original": {R_FOOT: XML_R_FOOT, L_FOOT: XML_L_FOOT, L_TOES: XML_L_TOES},
  "heel_extended": {R_FOOT: HEEL_R_FOOT, L_FOOT: XML_L_FOOT, L_TOES: XML_L_TOES},
  "feet_extended": {R_FOOT: HEEL_R_FOOT, L_FOOT: LOW_L_FOOT, L_TOES: LOW_L_TOES},
}
# Each site's body and quat (wxyz), which no version changes.
TOUCH_SITE_FRAMES = {
  R_FOOT: ("OSL_KA_L1_osl_foot_assembly", (1.0, 0.0, 0.0, 0.0)),
  L_FOOT: ("calcn_l", (1.0, 0.0, 0.0, 0.0)),
  L_TOES: ("toes_l", (0.9393726845901743, 0.0, 0.3428978848663966, 0.0)),
}
# The intact foot's colliders on each left zone's body.
LEFT_COLLIDERS = {
  L_FOOT: ("foot_col1_l", "foot_col3_l", "foot_col4_l"),
  L_TOES: ("bofoot_col1_l", "bofoot_col2_l"),
}
OSL_FOOT_BODY = "OSL_KA_L1_osl_foot_assembly"
OSL_FOOT_CAPSULES = tuple(f"OSL_KA_L1_osl_foot_col{i}" for i in (1, 2, 3))
# Everything get_spec builds: the four variants, and the musculoskeletal ones with
# their fingers kept.
SpecVariant = tuple[osl_constants.Model, osl_constants.Socket, bool]
SPEC_VARIANTS: list[SpecVariant] = [
  ("robot", "rigid", False),
  ("robot", "compliant", False),
  ("msk", "rigid", False),
  ("msk", "compliant", False),
  ("msk", "rigid", True),
  ("msk", "compliant", True),
]


def _spec_id(variant: SpecVariant) -> str:
  return "-".join(variant[:2]) + ("-fingers" if variant[2] else "")


def _model_fields(obj: object, prefix: str = "") -> dict[str, Any]:
  """Every data attribute of a compiled model: its arrays, sizes, names and
  signature, with the option, statistic and visual structs flattened."""
  fields: dict[str, Any] = {}
  for name in dir(obj):
    if name.startswith("_"):
      continue
    value = getattr(obj, name)
    if callable(value):
      continue
    if type(value).__module__.startswith("mujoco"):
      fields.update(_model_fields(value, f"{prefix}{name}."))
    else:
      fields[prefix + name] = value
  return fields


def _same(a: Any, b: Any) -> bool:
  if isinstance(a, np.ndarray):
    return np.array_equal(a, b, equal_nan=True)
  return a == b


def _differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
  assert a.keys() == b.keys()
  return sorted(name for name in a if not _same(a[name], b[name]))


def _box_bounds(model: mujoco.MjModel, name: str) -> tuple[np.ndarray, np.ndarray]:
  site = model.site(name)
  return site.pos - site.size, site.pos + site.size


def _cap_centres(model: mujoco.MjModel, name: str) -> tuple[np.ndarray, np.ndarray]:
  """A capsule's two end-cap centres in its body's frame, the back (heel) one
  first."""
  geom = model.geom(name)
  rotation = np.zeros(9)
  mujoco.mju_quat2Mat(rotation, geom.quat)
  half = rotation.reshape(3, 3)[:, 2] * geom.size[1]
  back, front = sorted((geom.pos - half, geom.pos + half), key=lambda p: p[0])
  return back, front


@pytest.fixture(scope="module", params=SPEC_VARIANTS, ids=_spec_id)
def spec_variant(request) -> SpecVariant:
  return request.param


@pytest.fixture(scope="module")
def touch_specs(spec_variant: SpecVariant) -> dict[str, mujoco.MjSpec]:
  """The variant from get_spec with each touch-site version."""
  model, socket, fingers = spec_variant
  return {
    version: osl_constants.get_spec(model, socket, fingers, touch_sites=version)
    for version in osl_constants.TOUCH_SITES
  }


@pytest.fixture(scope="module")
def touch_models(touch_specs: dict[str, mujoco.MjSpec]) -> dict[str, mujoco.MjModel]:
  return {version: spec.compile() for version, spec in touch_specs.items()}


@pytest.fixture(scope="module")
def xml_model(spec_variant: SpecVariant) -> mujoco.MjModel:
  model, socket, _ = spec_variant
  return mujoco.MjModel.from_xml_path(str(osl_constants.OSL_XMLS[(model, socket)]))


def test_touch_site_versions_and_defaults() -> None:
  """Three versions. Every newly configured model or entity builds the extended
  feet; anything that records no version trained with the XML's boxes."""
  assert osl_constants.TOUCH_SITES == ("original", "heel_extended", "feet_extended")
  assert osl_constants.DEFAULT_TOUCH_SITES == "feet_extended"
  assert osl_constants.LEGACY_TOUCH_SITES == "original"
  assert {v[:2] for v in SPEC_VARIANTS} == set(VARIANTS)
  boxes = {
    version: {site: (box.pos, box.size) for site, box in version_boxes.items()}
    for version, version_boxes in osl_constants.TOUCH_SITE_BOXES.items()
  }
  assert boxes == TOUCH_BOXES
  assert osl_constants.TOUCH_SITE_EDITS == {
    "original": (),
    "heel_extended": (R_FOOT,),
    "feet_extended": (R_FOOT, L_FOOT, L_TOES),
  }
  # r_foot: only the back edge moves, from x = -0.025 to -0.055.
  for (pos, size), edges in (
    (XML_R_FOOT, (-0.025, 0.075)),
    (HEEL_R_FOOT, (-0.055, 0.075)),
  ):
    assert (pos[0] - size[0], pos[0] + size[0]) == pytest.approx(edges)
  # The left zones: only the bottom face moves, 8 mm down from y = -0.020.
  for xml, low in ((XML_L_FOOT, LOW_L_FOOT), (XML_L_TOES, LOW_L_TOES)):
    (xml_pos, xml_size), (low_pos, low_size) = xml, low
    assert xml_pos[1] - xml_size[1] == pytest.approx(-0.020)
    assert low_pos[1] - low_size[1] == pytest.approx(-0.028)
    assert xml_pos[1] + xml_size[1] == pytest.approx(low_pos[1] + low_size[1])
    for axis in (0, 2):
      assert (xml_pos[axis], xml_size[axis]) == (low_pos[axis], low_size[axis])
  for fn in (
    osl_constants.get_spec,
    osl_constants.get_osl_robot_cfg,
    osl_constants.get_osl_msk_cfg,
  ):
    assert inspect.signature(fn).parameters["touch_sites"].default == "feet_extended"
  robot = osl_constants.get_osl_robot_cfg()
  assert isinstance(robot, osl_constants.OslEntityCfg)
  assert robot.touch_sites == "feet_extended"
  msk = osl_constants.get_osl_msk_cfg("compliant", fingers=True, touch_sites="original")
  assert isinstance(msk, osl_constants.OslEntityCfg)
  assert msk.touch_sites == "original"


def test_touch_sites_build_their_boxes(
  touch_models: dict[str, mujoco.MjModel],
) -> None:
  for version, model in touch_models.items():
    for name, (pos, size) in TOUCH_BOXES[version].items():
      site = model.site(name)
      body, quat = TOUCH_SITE_FRAMES[name]
      np.testing.assert_array_equal(site.pos, pos, err_msg=name)
      np.testing.assert_array_equal(site.size, size, err_msg=name)
      np.testing.assert_allclose(site.quat, quat, rtol=0, atol=1e-15, err_msg=name)
      assert site.type[0] == mujoco.mjtGeom.mjGEOM_BOX
      assert model.body(int(site.bodyid[0])).name == body
    assert osl_constants.touch_sites_of_model(model) == version


@pytest.mark.parametrize(
  ("a", "b", "changes"),
  [
    ("original", "heel_extended", {R_FOOT: (0,)}),
    ("heel_extended", "feet_extended", {L_FOOT: (1,), L_TOES: (1,)}),
    ("original", "feet_extended", {R_FOOT: (0,), L_FOOT: (1,), L_TOES: (1,)}),
  ],
)
def test_touch_sites_change_only_their_boxes(
  touch_models: dict[str, mujoco.MjModel],
  a: str,
  b: str,
  changes: dict[str, tuple[int, ...]],
) -> None:
  """Every array, size, name and option of two versions' compiled models is
  identical, every geom and all physics included, except the site_pos and
  site_size rows of the sites they edit differently, and only along the moved axis:
  x for r_foot's back edge, y for the left zones' bottom faces."""
  first, second = _model_fields(touch_models[a]), _model_fields(touch_models[b])
  sample = {"site_pos", "geom_size", "body_mass", "opt.timestep", "stat.extent"}
  assert sample <= first.keys()
  differ = _differences(first, second)
  assert differ == ["site_pos", "site_size"]
  model = touch_models[a]
  expected = {
    (model.site(name).id, axis) for name, axes in changes.items() for axis in axes
  }
  for field in differ:
    cells = np.argwhere(first[field] != second[field])
    assert {(int(row), int(axis)) for row, axis in cells} == expected, field


def test_original_touch_sites_are_the_xml(
  touch_models: dict[str, mujoco.MjModel], xml_model: mujoco.MjModel
) -> None:
  """``original`` keeps the four touch sites exactly as the vendored XML writes
  them, each carrying its touch sensor. The other versions change only their
  listed boxes (``test_touch_sites_change_only_their_boxes``)."""
  model = touch_models["original"]
  for site_name, sensor_name in osl_constants.TOUCH_SITE_SENSORS.items():
    ours, xml = model.site(site_name), xml_model.site(site_name)
    for attr in ("pos", "quat", "size", "type"):
      np.testing.assert_array_equal(
        getattr(ours, attr), getattr(xml, attr), err_msg=f"{site_name} {attr}"
      )
    body = xml_model.body(int(xml.bodyid[0])).name
    assert model.body(int(ours.bodyid[0])).name == body
    sensor = model.sensor(sensor_name)
    assert sensor.type[0] == mujoco.mjtSensor.mjSENS_TOUCH
    assert sensor.objtype[0] == mujoco.mjtObj.mjOBJ_SITE
    assert sensor.objid[0] == ours.id


def test_r_foot_box_covers_the_heel_caps(
  touch_models: dict[str, mujoco.MjModel],
) -> None:
  """The OSL foot meets the floor only at its three capsules' end caps. The XML's
  r_foot box ends in front of the heel caps' centres, so it never fires; the
  extended one holds each whole heel cap, at least 1.3 cm clear of its back edge,
  and still stops short of the toe caps, which r_toes holds."""
  for version, model in touch_models.items():
    foot_lo, foot_hi = _box_bounds(model, osl_constants.R_FOOT_TOUCH_SITE)
    toes_lo, toes_hi = _box_bounds(model, osl_constants.R_TOES_TOUCH_SITE)
    for name in OSL_FOOT_CAPSULES:
      # The capsules and the touch sites share the foot body's frame.
      assert model.body(int(model.geom(name).bodyid[0])).name == OSL_FOOT_BODY
      radius = float(model.geom(name).size[0])
      heel, toe = _cap_centres(model, name)
      assert np.all(toe - radius >= toes_lo) and np.all(toe + radius <= toes_hi)
      assert toe[0] - radius > foot_hi[0]
      if version == "original":
        assert heel[0] < foot_lo[0], name
      else:
        assert np.all(heel - radius >= foot_lo), name
        assert np.all(heel + radius <= foot_hi), name
        assert heel[0] - radius - foot_lo[0] >= 0.013 - 1e-9, name
      # Both OSL zones reach 8 mm below the capsules in every version.
      for lo in (foot_lo, toes_lo):
        assert min(heel[1], toe[1]) - radius - lo[1] == pytest.approx(0.008), name


def _lowest_y(model: mujoco.MjModel, name: str) -> float:
  """A convex geom's lowest point along its body's y axis (the foot's up axis)."""
  geom = model.geom(name)
  rotation = np.zeros(9)
  mujoco.mju_quat2Mat(rotation, geom.quat)
  # The body's y axis in the geom frame: the second row of the geom's rotation.
  up = rotation.reshape(3, 3)[1]
  kind, size = int(geom.type[0]), geom.size
  if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
    extent = abs(up[2]) * size[1] + size[0]
  elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    extent = size[0]
  elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    extent = float(np.linalg.norm(up * size))
  elif kind == mujoco.mjtGeom.mjGEOM_BOX:
    extent = float(np.abs(up) @ size)
  else:
    raise AssertionError(f"{name}: no support function for geom type {kind}")
  return float(geom.pos[1] - extent)


def test_left_zones_reach_below_the_intact_foot(
  touch_models: dict[str, mujoco.MjModel],
) -> None:
  """The XML's left zones stop exactly at their lowest colliders' lowest points, so
  a foot resting in the 1 mm contact margin (contact point below the capsule) is
  missed. feet_extended's reach 8 mm below every left collider, as the OSL zones
  do, while their tops stay at the body origin."""
  for version, model in touch_models.items():
    for zone, colliders in LEFT_COLLIDERS.items():
      site = model.site(zone)
      # Rotated about y only, so site y is body y.
      assert abs(site.quat[1]) == abs(site.quat[3]) == 0.0
      bottom, top = site.pos[1] - site.size[1], site.pos[1] + site.size[1]
      assert top == pytest.approx(0.0, abs=1e-15)
      lowest = {}
      for name in colliders:
        geom = model.geom(name)
        assert (
          model.body(int(geom.bodyid[0])).name == model.body(int(site.bodyid[0])).name
        ), name
        assert geom.contype[0] or geom.conaffinity[0], name
        assert float(geom.margin[0]) == pytest.approx(0.001), name
        lowest[name] = _lowest_y(model, name)
      spare = min(lowest.values()) - bottom
      if version == "feet_extended":
        assert spare == pytest.approx(0.008, abs=1e-9), (zone, lowest)
      else:
        assert spare == pytest.approx(0.0, abs=1e-9), (zone, lowest)


def test_touch_sites_leave_the_dynamics_bit_identical(
  touch_specs: dict[str, mujoco.MjSpec],
) -> None:
  """200 CPU steps from qpos0 at zero ctrl, over a floor so that the feet land (the
  entity has none: mjlab owns the terrain). qpos and every sensor reading but the
  edited zones' are bit-identical across the three versions. r_foot is the first
  fix: with the XML's box it never fires although the OSL foot is loaded; the
  extended box catches the heel. The lowered left zones contain the XML's, so
  they read at least as much at every step."""
  models: dict[str, mujoco.MjModel] = {}
  for version, spec in touch_specs.items():
    floored = spec.copy()
    floored.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05), contype=1, conaffinity=1
    )
    models[version] = floored.compile()
  r_foot, r_toes, l_foot, l_toes = (
    int(models["original"].sensor_adr[models["original"].sensor(name).id])
    for name in ("r_foot", "r_toes", "l_foot", "l_toes")
  )
  qpos: dict[str, np.ndarray] = {}
  readings: dict[str, np.ndarray] = {}
  for version, model in models.items():
    data = mujoco.MjData(model)
    assert not data.ctrl.any()
    steps = []
    for _ in range(200):
      mujoco.mj_step(model, data)
      steps.append(data.sensordata.copy())
    qpos[version], readings[version] = data.qpos.copy(), np.stack(steps)
  assert np.isfinite(qpos["original"]).all()
  assert np.array_equal(qpos["original"], qpos["heel_extended"])
  assert np.array_equal(qpos["original"], qpos["feet_extended"])
  columns = np.arange(readings["original"].shape[1])
  untouched = ~np.isin(columns, (r_foot, l_foot, l_toes))
  for version in ("heel_extended", "feet_extended"):
    np.testing.assert_array_equal(
      readings["original"][:, untouched], readings[version][:, untouched]
    )
  # heel_extended edits only r_foot; feet_extended has the same r_foot.
  for column in (l_foot, l_toes):
    np.testing.assert_array_equal(
      readings["original"][:, column], readings["heel_extended"][:, column]
    )
    assert np.all(
      readings["feet_extended"][:, column] >= readings["original"][:, column]
    )
  np.testing.assert_array_equal(
    readings["heel_extended"][:, r_foot], readings["feet_extended"][:, r_foot]
  )
  assert readings["original"][:, r_toes].max() > 0.0
  assert readings["original"][:, r_foot].max() == 0.0
  assert readings["heel_extended"][:, r_foot].max() > 0.0


def test_apply_touch_sites_either_way_and_idempotently() -> None:
  """Either version applies over either and the last call wins; a second call
  changes nothing. Each result compiles to get_spec's own model of that version."""
  specs = {
    v: osl_constants.get_spec("robot", "rigid", touch_sites=v)
    for v in osl_constants.TOUCH_SITES
  }
  expected = {v: _model_fields(spec.compile()) for v, spec in specs.items()}
  for start in osl_constants.TOUCH_SITES:
    for target in osl_constants.TOUCH_SITES:
      spec = specs[start].copy()
      for _ in range(2):
        assert osl_constants.apply_touch_sites(spec, target) is spec
      built = _model_fields(spec.compile())
      assert _differences(built, expected[target]) == [], (start, target)


def _spec_site(spec: mujoco.MjSpec, name: str) -> Any:
  (site,) = [s for s in spec.sites if s.name == name]
  return site


@pytest.mark.parametrize(
  ("name", "pos", "size"),
  [
    # Half of each: the XML's centre with the extended size.
    (R_FOOT, XML_R_FOOT[0], HEEL_R_FOOT[1]),
    (L_FOOT, LOW_L_FOOT[0], XML_L_FOOT[1]),
    (L_TOES, XML_L_TOES[0], LOW_L_TOES[1]),
  ],
)
def test_apply_touch_sites_refuses_an_unknown_box(
  name: str, pos: tuple[float, ...], size: tuple[float, ...]
) -> None:
  """A box that is none of a site's known ones (a regenerated model) is refused,
  and the spec is left untouched, the sites that were fine included."""
  spec = osl_constants.get_spec("robot", "rigid", touch_sites="original")
  site = _spec_site(spec, name)
  site.pos, site.size = pos, size
  before = {
    n: (tuple(_spec_site(spec, n).pos), tuple(_spec_site(spec, n).size))
    for n in TOUCH_SITE_FRAMES
  }
  for version in osl_constants.TOUCH_SITES:
    with pytest.raises(ValueError, match="none of the OSL touch-site versions' boxes"):
      osl_constants.apply_touch_sites(spec, version)
    after = {
      n: (tuple(_spec_site(spec, n).pos), tuple(_spec_site(spec, n).size))
      for n in TOUCH_SITE_FRAMES
    }
    assert after == before, version
  with pytest.raises(ValueError, match="none of the OSL touch-site versions' boxes"):
    osl_constants.touch_sites_of_model(spec.compile())


def test_touch_sites_of_model_refuses_a_mix() -> None:
  """Known boxes from different versions (heel_extended's r_foot with the
  lowered left zones, the XML's r_foot with them, or one lowered left zone) are no
  version; applying a version over the mix gives that version."""
  for r_foot, l_foot, l_toes in (
    (HEEL_R_FOOT, LOW_L_FOOT, XML_L_TOES),
    (XML_R_FOOT, LOW_L_FOOT, LOW_L_TOES),
    (HEEL_R_FOOT, XML_L_FOOT, LOW_L_TOES),
  ):
    spec = osl_constants.get_spec("robot", "rigid")
    for name, (pos, size) in ((R_FOOT, r_foot), (L_FOOT, l_foot), (L_TOES, l_toes)):
      site = _spec_site(spec, name)
      site.pos, site.size = pos, size
    with pytest.raises(ValueError, match="belong to different versions"):
      osl_constants.touch_sites_of_model(spec.compile())
    for version in osl_constants.TOUCH_SITES:
      built = osl_constants.apply_touch_sites(spec.copy(), version).compile()
      assert osl_constants.touch_sites_of_model(built) == version
  # Some touch sites but not all.
  spec = osl_constants.get_spec("robot", "rigid")
  (sensor,) = [s for s in spec.sensors if s.name == "l_toes"]
  spec.delete(sensor)
  spec.delete(_spec_site(spec, L_TOES))
  with pytest.raises(ValueError, match="some touch sites but not"):
    osl_constants.touch_sites_of_model(spec.compile())


def test_apply_touch_sites_refuses_unknown_versions_and_models() -> None:
  """Unknown versions are refused everywhere, and so is a model without the
  touch sites."""
  spec = osl_constants.get_spec("robot", "rigid")

  unknown = cast(Any, "heel")
  with pytest.raises(ValueError, match="expected one of"):
    osl_constants.apply_touch_sites(spec, unknown)
  with pytest.raises(ValueError, match="expected one of"):
    osl_constants.get_spec("robot", "rigid", touch_sites=unknown)
  with pytest.raises(ValueError, match="expected one of"):
    osl_constants.get_osl_robot_cfg(touch_sites=unknown)
  with pytest.raises(ValueError, match="expected one of"):
    replace(osl_constants.get_osl_robot_cfg(), touch_sites=unknown).build()
  with pytest.raises(ValueError, match="Expected one site"):
    osl_constants.apply_touch_sites(mujoco.MjSpec(), "original")
  assert osl_constants.touch_sites_of_model(mujoco.MjSpec().compile()) is None


@pytest.mark.parametrize("version", osl_constants.TOUCH_SITES)
def test_entity_cfg_applies_its_touch_sites_last(
  version: osl_constants.TouchSites,
) -> None:
  """A task may wrap spec_fn around a get_spec that builds some version (the
  stage-2 OSL sensors and the tracking IMU do); the entity's own setting still wins
  when it builds, as a scene builds it, and the config is left as it was."""
  other: osl_constants.TouchSites = next(
    v for v in osl_constants.TOUCH_SITES if v != version
  )
  cfg = osl_constants.get_osl_robot_cfg("rigid", touch_sites=version)
  assert isinstance(cfg, osl_constants.OslEntityCfg)
  inner = cfg.spec_fn
  assert inner is not None

  def chain() -> mujoco.MjSpec:
    return osl_constants.apply_touch_sites(inner(), other)

  cfg = replace(cfg, spec_fn=chain)
  built = cfg.build().spec.compile()
  assert osl_constants.touch_sites_of_model(built) == version
  assert cfg.spec_fn is chain and cfg.touch_sites == version
  # Entity(cfg) skips build(), so it keeps the chain's version.
  assert osl_constants.touch_sites_of_model(Entity(cfg).spec.compile()) == other
