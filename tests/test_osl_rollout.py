"""Acceptance test for the OSL full-body import, all four variants.

The cheapest decisive experiment for these models: hold a fixed command and roll
the same compiled model forward through both mjlab/mujoco_warp and plain CPU
MuJoCo, then compare qpos. One test covers the MuJoCo/mujoco_warp version gap,
the handling of the 44 mjEQ_JOINT equality constraints, the contact budget and,
for the musculoskeletal model, the muscle and spatial-tendon stack.

Three things make or break the comparison:

* **Zero action is not zero ctrl.** ``JointPositionAction`` maps a zero action to
  ``ctrl = default_joint_pos``, while a naive CPU rollout with ``d.ctrl = 0``
  commands every servo to 0 rad. Both sides are driven from the keyframe's ctrl.
* **Both sides must share the zeroed geom margin**, which they do here by using
  one compiled model, otherwise the 1 mm difference in contact timing reads as a
  version gap.
* **The musculoskeletal model cannot stand on a constant command.** Under a fixed
  low activation it sags and, after ~0.3 s, collapses, and a collapse is chaotic
  enough that float32 and float64 part ways by ~1e-2. Its window is therefore
  0.2 s, long enough to engage contact and the activation dynamics.

The musculoskeletal comparison runs under explicit Euler, because the two engines
disagree under mjlab's default ``implicitfast``: MuJoCo folds each muscle's
force-velocity derivative into the implicit damping term, and mujoco_warp 3.11
does not (``derivative.py`` takes gain 0 for ``GainType.MUSCLE`` and ignores
``BiasType.MUSCLE``). One step from the same state with active muscles moves
ankle velocity to -4.25 on CPU against -12.99 on warp. Under Euler both are
explicit and agree. ``test_msk_implicitfast_gap`` pins the gap so that a
mujoco_warp release closing it is noticed.
"""

import mujoco
import numpy as np
import pytest
from conftest import get_test_device

from mjlab.asset_zoo.robots.osl_fullbody import osl_constants
from mjlab.entity import Entity
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

# The 44 equality rows are unconditional, so the constraint budget starts at 44
# before a single contact. mjlab's default heuristic underflows here: mujoco_warp
# reports "nefc overflow - please increase njmax" and silently drops constraints,
# which shows up as ~1e-1 of spurious drift against CPU MuJoCo. Measured here: max
# nefc 88 (robot) / 99 (musculoskeletal), at most 11 contacts.
SIM_CFGS = {
  "robot": SimulationCfg(njmax=256, nconmax=48),
  "msk": SimulationCfg(njmax=256, nconmax=48, mujoco=MujocoCfg(integrator="euler")),
}

# Constant activation command for every muscle: enough to exercise activation
# dynamics and the force-length-velocity path, not enough to hold a pose.
MUSCLE_CTRL = 0.1
NUM_STEPS = {"robot": 300, "msk": 100}
# (early, whole-window) bounds on |qpos warp - qpos cpu|. Measured: robot 4.6e-07
# early and 2.6e-06 over 300 steps; musculoskeletal (fingers kept) 9.1e-05 early
# and 7.4e-04 over 100, all of it in the wrists and fingers -- light distal joints
# under strong muscles, integrated explicitly. The model is now fingerless by
# default; a scratch prototype measured about 1.6e-06 over 0.2 s, so these bounds
# hold with room to spare but have not been re-tightened.
TOLERANCES = {"robot": (1e-5, 1e-4), "msk": (1e-3, 1e-2)}
# The window where a structural error - a mismapped equality, a dropped contact,
# a misread actuator - shows up immediately.
EARLY_STEPS = 20

Variant = tuple[osl_constants.Model, osl_constants.Socket]
VARIANTS = list(osl_constants.OSL_XMLS)


@pytest.fixture(scope="module", params=VARIANTS, ids=lambda v: "-".join(v))
def variant(request) -> Variant:
  return request.param


@pytest.fixture(scope="module")
def osl_model(variant: Variant) -> mujoco.MjModel:
  """The model standing on a ground plane.

  The entity deliberately ships no floor, so one has to be added here or the
  rollout is pure free fall and never exercises contact at all. This plane is
  identical to the one ``TerrainEntity._import_ground_plane`` creates, including
  its default contype/conaffinity of 1/1 -- which is what makes the model's
  contype=0/conaffinity=1 geoms collide with the ground but not each other.
  """
  model, socket = variant
  if model == "robot":
    cfg = osl_constants.get_osl_robot_cfg(socket)
  else:
    cfg = osl_constants.get_osl_msk_cfg(socket)
  entity = Entity(cfg)
  # After construction, so the entity's CollisionCfg -- which runs with
  # disable_other_geoms=True -- does not zero the plane's contype/conaffinity
  # along with every other geom it does not name.
  # contype/conaffinity are set explicitly because this model's main <default>
  # declares contype="0" conaffinity="0", which a bare add_geom would inherit.
  entity.spec.worldbody.add_body(name="terrain").add_geom(
    name="terrain",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),
    contype=1,
    conaffinity=1,
  )
  compiled = entity.spec.compile()
  SIM_CFGS[model].mujoco.apply(compiled)
  return compiled


def _initial_state(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  ctrl = data.ctrl.copy()
  # The OSL motors are torque channels; the keyframe writes an angle into them.
  for name in osl_constants.OSL_TORQUE_JOINTS:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_actuated = model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
    ctrl[joint_actuated & (model.actuator_trnid[:, 0] == joint_id)] = 0.0
  ctrl[model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE] = MUSCLE_CTRL
  return data.qpos.copy(), ctrl


def _cpu_rollout(model: mujoco.MjModel, steps: int) -> np.ndarray:
  qpos, ctrl = _initial_state(model)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.ctrl[:] = ctrl
  trajectory = np.empty((steps, model.nq))
  for step in range(steps):
    mujoco.mj_step(model, data)
    trajectory[step] = data.qpos
  return trajectory


def _warp_rollout(model: mujoco.MjModel, steps: int, cfg: SimulationCfg) -> np.ndarray:
  qpos, ctrl = _initial_state(model)
  # Simulation re-applies cfg.mujoco to the model, so cfg must match the CPU side.
  sim = Simulation(num_envs=1, cfg=cfg, model=model, device=get_test_device())
  sim.data.qpos[:] = sim.data.qpos.new_tensor(qpos)
  sim.data.qvel[:] = 0.0
  sim.data.ctrl[:] = sim.data.ctrl.new_tensor(ctrl)
  sim.forward()
  trajectory = np.empty((steps, model.nq))
  for step in range(steps):
    sim.step()
    trajectory[step] = sim.data.qpos[0].cpu().numpy()
  return trajectory


@pytest.mark.slow
def test_warp_matches_cpu_mujoco(variant: Variant, osl_model: mujoco.MjModel) -> None:
  steps = NUM_STEPS[variant[0]]
  cpu = _cpu_rollout(osl_model, steps)
  warp = _warp_rollout(osl_model, steps, SIM_CFGS[variant[0]])

  assert np.isfinite(cpu).all(), "CPU MuJoCo rollout diverged"
  assert np.isfinite(warp).all(), "mujoco_warp rollout diverged"

  # The model must actually be standing on the ground. Without the terrain plane
  # this rollout is a free fall that agrees beautifully and proves nothing.
  assert cpu[-1, 2] > 0.8

  error = np.abs(warp - cpu)
  # mujoco_warp is float32 against MuJoCo's float64 over a model with 44 stiff
  # equality constraints and up to 11 contacts, so the two drift apart. Bounds sit
  # over an order of magnitude above measured drift, which leaves room for float32
  # nondeterminism while still catching a structural regression -- an undersized
  # constraint budget, for instance, lands at ~1e-01.
  early_bound, bound = TOLERANCES[variant[0]]
  assert error[:EARLY_STEPS].max() < early_bound
  assert error.max() < bound
  # Root translation, where drift is easiest to interpret physically. Measured at
  # 3e-07 or less for every variant.
  assert np.abs(warp[:, :3] - cpu[:, :3]).max() < 1e-5


@pytest.mark.slow
def test_equality_constraints_hold(variant: Variant, osl_model: mujoco.MjModel) -> None:
  """The 44 coupled joints stay coupled through a contact-rich rollout.

  ``test_warp_matches_cpu_mujoco`` already shows that mujoco_warp reproduces
  MuJoCo's evaluation of these 5-term polycoef constraints. This bounds their
  absolute residual, whose floor is set by the model's own solimp softness rather
  than by solver iterations -- so it is a property of the model, and it puts a
  floor on how well any tracked body on a coupled chain can be positioned.
  """
  qpos, ctrl = _initial_state(osl_model)
  data = mujoco.MjData(osl_model)
  data.qpos[:] = qpos
  data.ctrl[:] = ctrl
  for _ in range(NUM_STEPS[variant[0]]):
    mujoco.mj_step(osl_model, data)
  residual = max(
    abs(data.efc_pos[i])
    for i in range(data.nefc)
    if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
  )
  assert residual < 3e-2


@pytest.mark.slow
@pytest.mark.xfail(
  strict=True,
  reason="mujoco_warp implicitfast omits the muscle force-velocity derivative",
)
def test_msk_implicitfast_gap() -> None:
  """One implicitfast step with active muscles, CPU against warp.

  Strict xfail: when this starts passing, mujoco_warp has closed the gap, and the
  musculoskeletal rollout above can move to mjlab's default integrator.
  """
  model = Entity(osl_constants.get_osl_msk_cfg()).spec.compile()
  cfg = SimulationCfg(njmax=256, nconmax=48)
  cfg.mujoco.apply(model)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.ctrl[:] = MUSCLE_CTRL
  data.act[:] = 0.3
  sim = Simulation(num_envs=1, cfg=cfg, model=model, device=get_test_device())
  sim.data.qpos[:] = sim.data.qpos.new_tensor(data.qpos)
  sim.data.ctrl[:] = sim.data.ctrl.new_tensor(data.ctrl)
  sim.data.act[:] = sim.data.act.new_tensor(data.act)
  mujoco.mj_step(model, data)
  sim.step()
  np.testing.assert_allclose(
    sim.data.qvel[0].cpu().numpy(), data.qvel, rtol=0, atol=1e-2
  )
