"""The OSL's onboard sensors, added to the model through MjSpec.

The stage-2 OSL actor sees only what the prosthesis itself measures. The v3
``osl_fullbody`` XMLs carry none of it (their four foot touch sensors are the
human's, and the OSL foot's misses the heel), and the XMLs are never hand-edited, so
:func:`add_osl_sensors` adds every channel programmatically:

* the knee and ankle encoders (``jointpos``, ``jointvel``);
* a shank IMU, a gyro and an accelerometer on a new site on the knee assembly, just
  below the knee joint where the knee ActPack sits. Neither has a MuJoCo ``cutoff``:
  the observation saturates them at the ActPack's ranges (:data:`GYRO_RANGE`,
  :data:`ACC_RANGE`) after its per-episode mount rotation and bias, as the chip
  saturates what it measures, and a cutoff would clip the site-frame vector before
  those;
* two six-axis load cells at the model's existing load sites: above the ankle
  (on the ankle assembly) and below the socket (on the femoral pylon);
* each joint's applied actuator torque (``jointactuatorfrc``). ``actuatorfrc``
  would read ctrl units on the robot model (its gear is the 49.4 / 58.4 N m scale)
  but N m on the musculoskeletal one (gain, gear 1); the joint-side sensor reads N m
  on both.

Every sensor is always added; the observation config picks which reach the actor.
The knee assembly, tibial pylon and ankle assembly are one rigid shank, so the IMU
and the ankle cell share their axes. All frames are the OSL bodies' OpenSim axes: x
anterior, y along the limb (up), z to the right. A force or torque sensor reports
what the parent body applies to the site body's subtree, in the site's frame, so
the hanging leg reads its distal weight along +y: 19.9 N (2.03 kg) at the ankle and
44.5 N (4.54 kg) at the socket. :func:`static_load_cell_tare` computes those
readings, which the observation subtracts as a device's tare removes them.

mjlab attaches the entity under the prefix ``"robot/"``, so in the scene these are
``robot/osl_imu``, ``robot/osl_imu_gyro`` and so on (:func:`scene_sensor_name`), and
every sensor is registered as a :class:`~mjlab.sensor.BuiltinSensor`.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OSL_TORQUE_JOINTS

OSL_IMU_SITE = "osl_imu"
OSL_IMU_BODY = "OSL_KA_L1_osl_knee_assembly"
OSL_IMU_POS: tuple[float, float, float] = (0.0, -0.06, 0.0)
"""In the knee assembly's frame, whose origin is the knee joint: 6 cm down the shank,
near the knee actuator."""
ANKLE_LOAD_SITE = "OSL_KA_L1_r_osl_load_force"
"""On the ankle assembly, 10.5 cm above its origin: its cell carries the ankle
assembly and the foot."""
SOCKET_LOAD_SITE = "OSL_KA_L1_r_socket_load_force"
"""On the femoral pylon, the first body below the socket (and below the compliant
socket's four joints): its cell carries the whole prosthesis."""
GYRO_RANGE = math.radians(1000.0)
"""The ActPack IMU's gyro range, +-1000 deg/s (rad/s), per axis. The observation
clamps to it; the MuJoCo sensor does not."""
ACC_RANGE = 4.0 * 9.80665
"""The ActPack IMU's accelerometer range, +-4 g (m/s^2), per axis. The observation
clamps to it; the MuJoCo sensor does not."""

# Entity-level sensor names; the scene registers each as "robot/<name>".
IMU_GYRO = "osl_imu_gyro"
IMU_ACC = "osl_imu_acc"
ANKLE_FORCE = "osl_ankle_load_force"
ANKLE_LOAD_TORQUE = "osl_ankle_load_torque"
"""The ankle load cell's torque. ``ANKLE_TORQUE`` is the ankle joint's applied
actuator torque, the counterpart of ``KNEE_TORQUE``."""
SOCKET_FORCE = "osl_socket_load_force"
SOCKET_TORQUE = "osl_socket_load_torque"
KNEE_POS = "osl_knee_pos"
KNEE_VEL = "osl_knee_vel"
ANKLE_POS = "osl_ankle_pos"
ANKLE_VEL = "osl_ankle_vel"
KNEE_TORQUE = "osl_knee_torque"
ANKLE_TORQUE = "osl_ankle_torque"

LOAD_CELL_SENSORS: tuple[str, ...] = (
  ANKLE_FORCE,
  ANKLE_LOAD_TORQUE,
  SOCKET_FORCE,
  SOCKET_TORQUE,
)
"""The keys of :func:`static_load_cell_tare`."""


@dataclass(frozen=True)
class OslSensorSpec:
  """One added sensor: what MuJoCo measures, on which object, in which unit."""

  name: str
  type: mujoco.mjtSensor
  objtype: mujoco.mjtObj
  objname: str
  """The site or joint, by its entity-level name."""
  dim: int
  unit: str


_S = mujoco.mjtSensor
_SITE = mujoco.mjtObj.mjOBJ_SITE
_JOINT = mujoco.mjtObj.mjOBJ_JOINT
_KNEE_JOINT, _ANKLE_JOINT = OSL_TORQUE_JOINTS

OSL_SENSOR_SPECS: tuple[OslSensorSpec, ...] = (
  OslSensorSpec(IMU_GYRO, _S.mjSENS_GYRO, _SITE, OSL_IMU_SITE, 3, "rad/s"),
  OslSensorSpec(IMU_ACC, _S.mjSENS_ACCELEROMETER, _SITE, OSL_IMU_SITE, 3, "m/s^2"),
  OslSensorSpec(ANKLE_FORCE, _S.mjSENS_FORCE, _SITE, ANKLE_LOAD_SITE, 3, "N"),
  OslSensorSpec(ANKLE_LOAD_TORQUE, _S.mjSENS_TORQUE, _SITE, ANKLE_LOAD_SITE, 3, "N m"),
  OslSensorSpec(SOCKET_FORCE, _S.mjSENS_FORCE, _SITE, SOCKET_LOAD_SITE, 3, "N"),
  OslSensorSpec(SOCKET_TORQUE, _S.mjSENS_TORQUE, _SITE, SOCKET_LOAD_SITE, 3, "N m"),
  OslSensorSpec(KNEE_POS, _S.mjSENS_JOINTPOS, _JOINT, _KNEE_JOINT, 1, "rad"),
  OslSensorSpec(KNEE_VEL, _S.mjSENS_JOINTVEL, _JOINT, _KNEE_JOINT, 1, "rad/s"),
  OslSensorSpec(ANKLE_POS, _S.mjSENS_JOINTPOS, _JOINT, _ANKLE_JOINT, 1, "rad"),
  OslSensorSpec(ANKLE_VEL, _S.mjSENS_JOINTVEL, _JOINT, _ANKLE_JOINT, 1, "rad/s"),
  OslSensorSpec(KNEE_TORQUE, _S.mjSENS_JOINTACTFRC, _JOINT, _KNEE_JOINT, 1, "N m"),
  OslSensorSpec(ANKLE_TORQUE, _S.mjSENS_JOINTACTFRC, _JOINT, _ANKLE_JOINT, 1, "N m"),
)
"""Every added sensor, in a fixed order. None has a MuJoCo ``cutoff``: the hardware
ranges are the observation's (``mdp.observations.osl_actpack``)."""
OSL_SENSORS: tuple[str, ...] = tuple(s.name for s in OSL_SENSOR_SPECS)

TARE_LIFT = 1.0
"""How far :func:`static_load_cell_tare` raises the free joint (m)."""


def add_osl_sensors(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Add the IMU site and every OSL sensor to an ``osl_fullbody`` spec, in place.

  Returns the spec, for chaining.

  Raises:
    ValueError: If the spec lacks an OSL body, site or joint the sensors need, or
      already has the IMU site or one of the sensor names (a second call would
      otherwise fail only at compile, with a less useful message).
  """
  body = spec.body(OSL_IMU_BODY)
  if body is None:
    raise ValueError(f"No body {OSL_IMU_BODY!r} for the OSL IMU: not an OSL model?")
  # Names from the element lists, not spec.sensor(name): once anything has been
  # deleted from a spec (get_spec deletes the scene furniture), MuJoCo 3.11's
  # sensor lookup misses sensors added afterwards, until the next compile.
  sites = {site.name for site in spec.sites}
  missing = [s for s in (ANKLE_LOAD_SITE, SOCKET_LOAD_SITE) if s not in sites]
  missing += [j for j in OSL_TORQUE_JOINTS if spec.joint(j) is None]
  if missing:
    raise ValueError(f"The OSL sensors need {missing}, which the spec lacks.")
  if OSL_IMU_SITE in sites:
    raise ValueError(f"The spec already has a site {OSL_IMU_SITE!r}.")
  taken = sorted({sensor.name for sensor in spec.sensors} & set(OSL_SENSORS))
  if taken:
    raise ValueError(f"The spec already has the sensors {taken}.")

  body.add_site(name=OSL_IMU_SITE, pos=OSL_IMU_POS)
  for sensor in OSL_SENSOR_SPECS:
    spec.add_sensor(
      name=sensor.name,
      type=sensor.type,
      objtype=sensor.objtype,
      objname=sensor.objname,
    )
  return spec


def with_osl_sensors(
  spec_fn: Callable[[], mujoco.MjSpec],
) -> Callable[[], mujoco.MjSpec]:
  """``spec_fn`` with :func:`add_osl_sensors` applied to every spec it builds."""

  def spec_with_osl_sensors() -> mujoco.MjSpec:
    return add_osl_sensors(spec_fn())

  return spec_with_osl_sensors


def scene_sensor_name(name: str, entity: str = "robot") -> str:
  """The scene-level name of an entity's sensor or site."""
  return f"{entity}/{name}"


def static_load_cell_tare(
  model: mujoco.MjModel, prefix: str = "robot/"
) -> dict[str, np.ndarray]:
  """The load cells' readings with the leg hanging still: their tare.

  Static inverse dynamics on CPU at the model's reference pose (``qpos0``, the
  model standing upright): the entity's free joint raised by :data:`TARE_LIFT`,
  ``qvel = qacc = 0``, and ``mj_inverse``, which fills the acceleration-stage
  sensors (force and torque need MuJoCo's post-constraint RNE). Constraints are
  disabled on a copy of the model, so no contact or equality force reaches the leg
  whatever the scene: each cell reads exactly the weight of what hangs below it.
  Subtracting these reproduces a device tared with its leg unloaded.

  The stage-1 randomization never changes the OSL's masses, so one nominal tare
  serves every env.

  Args:
    model: A compiled model with the OSL sensors (:func:`add_osl_sensors`).
    prefix: The entity's name prefix: ``"robot/"`` in an mjlab scene, ``""`` for a
      bare entity model.

  Returns:
    Each of :data:`LOAD_CELL_SENSORS` (entity-level names) to its (3,) reading.
  """
  model = copy.copy(model)  # The flag must not reach the caller's model.
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  for joint in range(model.njnt):
    name = model.joint(joint).name
    if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE and name.startswith(prefix):
      data.qpos[model.jnt_qposadr[joint] + 2] += TARE_LIFT
  data.qvel[:] = 0.0
  data.qacc[:] = 0.0
  mujoco.mj_inverse(model, data)
  tare: dict[str, np.ndarray] = {}
  for name in LOAD_CELL_SENSORS:
    sensor = model.sensor(prefix + name)
    adr, dim = int(sensor.adr[0]), int(sensor.dim[0])
    tare[name] = data.sensordata[adr : adr + dim].copy()
  return tare
