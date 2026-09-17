"""Acceptance test for the OSL full-body robot import.

The cheapest decisive experiment for this model: hold a fixed command and roll
the same compiled model forward through both mjlab/mujoco_warp and plain CPU
MuJoCo, then compare qpos. One test covers the MuJoCo/mujoco_warp version gap,
the handling of the 44 mjEQ_JOINT equality constraints, and the contact budget.

Two things make or break the comparison:

* **Zero action is not zero ctrl.** ``JointPositionAction`` maps a zero action to
  ``ctrl = default_joint_pos``, while a naive CPU rollout with ``d.ctrl = 0``
  commands every servo to 0 rad. Both sides are driven from the keyframe's ctrl.
* **Both sides must share the zeroed geom margin**, which they do here by using
  one compiled model, otherwise the 1 mm difference in contact timing reads as a
  version gap.
"""

import mujoco
import numpy as np
import pytest
from conftest import get_test_device

from mjlab.asset_zoo.robots.osl_fullbody_robot import osl_constants
from mjlab.entity import Entity
from mjlab.sim.sim import Simulation, SimulationCfg

# The 44 equality rows are unconditional, so the constraint budget starts at 44
# before a single contact. mjlab's default heuristic underflows here: mujoco_warp
# reports "nefc overflow - please increase njmax to 69" and silently drops
# constraints, which shows up as ~1e-1 of spurious drift against CPU MuJoCo.
SIM_CFG = SimulationCfg(njmax=256, nconmax=48)

NUM_STEPS = 300
# The window where a structural error - a mismapped equality, a dropped contact,
# a misread actuator - shows up immediately.
EARLY_STEPS = 20


@pytest.fixture(scope="module")
def osl_model() -> mujoco.MjModel:
  """The robot standing on a ground plane.

  The entity deliberately ships no floor, so one has to be added here or the
  rollout is pure free fall and never exercises contact at all. This plane is
  identical to the one ``TerrainEntity._import_ground_plane`` creates, including
  its default contype/conaffinity of 1/1 -- which is what makes the robot's
  contype=0/conaffinity=1 geoms collide with the ground but not each other.
  """
  entity = Entity(osl_constants.get_osl_robot_cfg())
  # After construction, so the entity's CollisionCfg -- which runs with
  # disable_other_geoms=True -- does not zero the plane's contype/conaffinity
  # along with every other geom it does not name.
  # contype/conaffinity are set explicitly because this model's main <default>
  # declares contype="0" conaffinity="0", which a bare add_geom would inherit.
  # mjlab's real terrain is a separate entity with its own spec, so it keeps the
  # MjSpec defaults of 1/1 and this only matters here.
  entity.spec.worldbody.add_body(name="terrain").add_geom(
    name="terrain",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),
    contype=1,
    conaffinity=1,
  )
  model = entity.spec.compile()
  SIM_CFG.mujoco.apply(model)
  return model


def _initial_state(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  ctrl = data.ctrl.copy()
  # The OSL motors are torque channels; the keyframe writes an angle into them.
  for name in osl_constants.OSL_TORQUE_JOINTS:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    ctrl[np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)] = 0.0
  return data.qpos.copy(), np.zeros(model.nv), ctrl


def _cpu_rollout(model: mujoco.MjModel) -> np.ndarray:
  qpos, qvel, ctrl = _initial_state(model)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  data.ctrl[:] = ctrl
  trajectory = np.empty((NUM_STEPS, model.nq))
  for step in range(NUM_STEPS):
    mujoco.mj_step(model, data)
    trajectory[step] = data.qpos
  return trajectory


def _warp_rollout(model: mujoco.MjModel) -> np.ndarray:
  qpos, qvel, ctrl = _initial_state(model)
  sim = Simulation(num_envs=1, cfg=SIM_CFG, model=model, device=get_test_device())
  sim.data.qpos[:] = sim.data.qpos.new_tensor(qpos)
  sim.data.qvel[:] = sim.data.qvel.new_tensor(qvel)
  sim.data.ctrl[:] = sim.data.ctrl.new_tensor(ctrl)
  sim.forward()
  trajectory = np.empty((NUM_STEPS, model.nq))
  for step in range(NUM_STEPS):
    sim.step()
    trajectory[step] = sim.data.qpos[0].cpu().numpy()
  return trajectory


@pytest.mark.slow
def test_warp_matches_cpu_mujoco(osl_model: mujoco.MjModel) -> None:
  cpu = _cpu_rollout(osl_model)
  warp = _warp_rollout(osl_model)

  assert np.isfinite(cpu).all(), "CPU MuJoCo rollout diverged"
  assert np.isfinite(warp).all(), "mujoco_warp rollout diverged"

  # The robot must actually be standing on the ground. Without the terrain plane
  # this rollout is a 0.3 s free fall that agrees beautifully and proves nothing.
  assert cpu[-1, 2] > 0.8

  error = np.abs(warp - cpu)
  # mujoco_warp is float32 against MuJoCo's float64 over a model with 44 stiff
  # equality constraints and up to 15 contacts, so the two drift apart. Bounds are
  # set about an order of magnitude above measured drift (8.8e-07 early, 7.0e-05
  # over the full rollout), which leaves room for float32 nondeterminism while
  # still catching a structural regression -- an undersized constraint budget, for
  # instance, lands at 2.8e-01.
  assert error[:EARLY_STEPS].max() < 1e-5
  assert error.max() < 1e-3
  # Root translation, where drift is easiest to interpret physically.
  assert np.abs(warp[:, :3] - cpu[:, :3]).max() < 1e-4


@pytest.mark.slow
def test_equality_constraints_hold_on_both_paths(osl_model: mujoco.MjModel) -> None:
  """The 44 coupled joints stay coupled through a contact-rich rollout.

  ``test_warp_matches_cpu_mujoco`` already shows that mujoco_warp reproduces
  MuJoCo's evaluation of these 5-term polycoef constraints. This bounds their
  absolute residual, whose floor is set by the model's own solimp softness rather
  than by solver iterations -- so it is a property of the model, and it puts a
  floor on how well any tracked body on a coupled chain can be positioned.
  """
  qpos, qvel, ctrl = _initial_state(osl_model)
  data = mujoco.MjData(osl_model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  data.ctrl[:] = ctrl
  for _ in range(NUM_STEPS):
    mujoco.mj_step(osl_model, data)
  residual = max(
    (
      abs(data.efc_pos[i])
      for i in range(data.nefc)
      if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    ),
    default=0.0,
  )
  assert residual < 3e-2
