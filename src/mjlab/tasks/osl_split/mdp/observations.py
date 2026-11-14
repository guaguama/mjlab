"""The OSL actor's observation: the prosthesis's own sensors, as the device reads them.

The stage-2 OSL actor is meant to run on the prosthesis, so it observes only what the
prosthesis measures (``config/sensors.py``): never the simulation state, the
reference, the commanded twist or the stride clock. Up to four terms, one per
hardware path, in this order (:data:`OSL_OBS_TERMS`):

* ``actpack``: the ActPack bus, the knee and ankle encoders and the shank IMU's gyro
  and accelerometer, plus two off-by-default extras: projected gravity (an
  orientation estimator's output rather than a raw channel) and the joints' applied
  torque (the drive's current-based estimate);
* ``loadcell``: the load-cell bus, the six-axis cell above the ankle (on by default)
  and the one below the socket (off by default), both tared;
* ``contact``: the foot-contact bus, the OSL foot's heel and toe contact forces (the
  model's ``r_foot`` and ``r_toes`` touch sensors), raw, both on by default;
* ``last_action``: the controller's own previous command, which crosses no bus.

The default set is the real leg's: encoders, IMU, the ankle cell and the two foot
contacts, 500 inputs. Each sensor has one switch, a param of its term (``ankle``,
``socket``, ``heel``, ``toe``), and a sensor term whose switches are all off is left
out of the group (the drop rule; ``actpack`` always stays). Old runs recorded other
sets (both cells and no contacts, :data:`LEGACY_OSL_SENSORS`); the record helpers at
the end of this module rebuild any of them from a run's record.

Perception randomization models the hardware and applies in training only, from the
first stage-2 iteration:

* per-episode errors, drawn at every reset by the class terms: encoder offsets, IMU,
  load-cell and contact biases, and a small IMU mount rotation;
* white noise on every sensor channel, one Gaussian std per component, added by the
  class terms as the last step of their output, so before the manager's delay and
  history;
* latency, one lag per bus (:data:`OSL_BUS_TERMS`), drawn per env at every reset and
  held for the episode (the manager's ``DelayBuffer`` with ``delay_hold_prob=1``,
  one buffer per term): the channels of a bus share their lag, and the buses lag
  independently.

A class term reads its switches, error bounds and noise stds from its own params,
and its ``randomize`` param turns both the errors and the noise on. Its output and
its noise therefore follow the same params however they are set: through
:class:`OslObsCfg`, or one at a time on the train CLI (e.g.
``--env.observations.osl.terms.actpack.params.projected-gravity True``).

Validation reads the sensors exactly, each bus at a fixed lag, the middle of its
training range; play reads them exactly with training's lags. Both are the training
group with its randomization off (:func:`osl_validation_group` for validation), so
they observe what training does, at its dims. The previous command is always clean
and immediate. Every term keeps the same history, flattened term-major and oldest to
newest.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

import numpy as np
import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OSL_TORQUE_JOINTS
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.osl_split.config.sensors import (
  ACC_RANGE,
  ANKLE_FORCE,
  ANKLE_LOAD_TORQUE,
  ANKLE_POS,
  ANKLE_TORQUE,
  ANKLE_VEL,
  GYRO_RANGE,
  IMU_ACC,
  IMU_GYRO,
  KNEE_POS,
  KNEE_TORQUE,
  KNEE_VEL,
  OSL_CONTACT_SPECS,
  OSL_IMU_SITE,
  OSL_SENSOR_SPECS,
  SOCKET_FORCE,
  SOCKET_TORQUE,
  OslSensorSpec,
  scene_sensor_name,
  static_load_cell_tare,
)
from mjlab.tasks.osl_split.rl.layout import OSL_ACTION_TERM
from mjlab.utils.lab_api.math import matrix_from_euler

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.osl_mimic.config.env_cfgs import Mode

OSL_OBS_GROUP = "osl"
OSL_OBS_TERMS: tuple[str, ...] = ("actpack", "loadcell", "contact", "last_action")
"""The ``osl`` group's terms, in order. A sensor set may leave out ``loadcell`` and
``contact`` (:data:`SENSOR_SWITCHES`)."""
OSL_BUS_TERMS: tuple[str, ...] = ("actpack", "loadcell", "contact")
"""The terms that cross a hardware bus, so lag: every sensor term. Validation fixes
each one's lag (:func:`osl_validation_group`); ``last_action`` never lags."""
OSL_LAST_ACTION_DIM = len(OSL_TORQUE_JOINTS)
"""The knee and ankle commands."""
SENSOR_SWITCHES: Mapping[str, tuple[str, ...]] = MappingProxyType(
  {"loadcell": ("ankle", "socket"), "contact": ("heel", "toe")}
)
"""The terms a sensor set may leave out, with the params that switch their sensors.
A term whose switches are all off is left out of the group (the drop rule)."""
SWITCH_FIELDS: Mapping[tuple[str, str], str] = MappingProxyType(
  {
    ("loadcell", "ankle"): "ankle_load_cell",
    ("loadcell", "socket"): "socket_load_cell",
    ("contact", "heel"): "heel_contact",
    ("contact", "toe"): "toe_contact",
  }
)
"""Each sensor switch, ``(term, param)``, and its :class:`OslObsCfg` field."""
_SWITCH_DEFAULTS = {
  "projected_gravity": False,
  "applied_torque": False,
  "ankle": True,
  "socket": True,
  "heel": True,
  "toe": True,
}
"""A class term's switches when its params leave them out: what a MISSING key
means, not :class:`OslObsCfg`'s defaults. Never change the load cells' entries:
the records of runs before the ``ankle`` param (M0, R1, R2, s2_A, s2_B) have no
``ankle`` key and read both cells."""


class OslSensorSwitches(TypedDict):
  """A sensor set as :class:`OslObsCfg`'s switch fields."""

  ankle_load_cell: bool
  socket_load_cell: bool
  heel_contact: bool
  toe_contact: bool


LEGACY_OSL_SENSORS: OslSensorSwitches = cast(
  OslSensorSwitches,
  MappingProxyType(
    {
      "ankle_load_cell": True,
      "socket_load_cell": True,
      "heel_contact": False,
      "toe_contact": False,
    }
  ),
)
"""The sensor set of every run before the foot contacts (both load cells, no
contacts), as :class:`OslObsCfg` fields: ``OslObsCfg(**LEGACY_OSL_SENSORS)``
(read-only). Stage 1.5's clean group keeps it, and a checkpoint recorded nowhere is
read as it."""


@dataclass
class OslObsCfg:
  """The OSL actor's observation and its perception randomization.

  The randomization applies in training only. Stds are Gaussian white noise per
  sample; biases are uniform in +-value per component, drawn at every reset and held
  for the episode. The switches, bounds and stds reach the class terms as their
  params, under the same names (``ankle_load_cell`` as ``ankle``,
  ``socket_load_cell`` as ``socket``).
  """

  history_length: int = 25
  """Frames per term: 0.25 s at 100 Hz."""
  delay_steps: tuple[int, int] = (0, 2)
  """Sensor latency per bus (min, max) in control steps, 0-20 ms at 100 Hz."""
  validation_delay_steps: int = 1
  """The fixed latency of validation: the middle of the training range."""
  ankle_load_cell: bool = True
  """The six-axis cell above the ankle (``loadcell``'s first six channels)."""
  socket_load_cell: bool = False
  """The six-axis cell below the socket, which the real leg does not have. With both
  cells off the ``loadcell`` term is left out."""
  heel_contact: bool = True
  """The OSL foot's heel contact force (``contact``'s first channel). It needs the
  extended heel zone: ``r_foot`` never fires under ``osl_touch_sites`` original."""
  toe_contact: bool = True
  """The OSL foot's toe contact force. With both off the ``contact`` term is left
  out."""
  projected_gravity: bool = False
  applied_torque: bool = False
  encoder_bias_rad: float = 0.015
  encoder_pos_noise: float = 0.002
  """rad."""
  encoder_vel_noise: float = 0.05
  """rad/s."""
  gyro_bias: float = 0.02
  """rad/s."""
  gyro_noise: float = 0.02
  acc_bias: float = 0.2
  """m/s^2."""
  acc_noise: float = 0.1
  gravity_noise: float = 0.01
  """Unit-vector components, when ``projected_gravity``."""
  imu_mount_deg: float = 3.0
  """The IMU's mount rotation about each of its axes, drawn per episode."""
  force_bias: float = 2.0
  """N."""
  force_noise: float = 1.0
  torque_bias: float = 0.2
  """N m."""
  torque_noise: float = 0.1
  applied_torque_noise: float = 1.0
  """N m, when ``applied_torque``."""
  contact_bias: float = 2.0
  """N, per contact channel: the load cell's force bias."""
  contact_noise: float = 1.0
  """N, per contact channel: the load cell's force noise."""

  def __post_init__(self) -> None:
    lo, hi = self.delay_steps
    if not 0 <= lo <= hi:
      raise ValueError(f"delay_steps must satisfy 0 <= min <= max: {self.delay_steps}")
    if self.history_length < 0 or self.validation_delay_steps < 0:
      raise ValueError("history_length and validation_delay_steps must be >= 0.")
    magnitudes = {
      name: getattr(self, name)
      for name in self.__dataclass_fields__
      if name.endswith(("_bias", "_bias_rad", "_noise", "_deg"))
    }
    negative = sorted(name for name, value in magnitudes.items() if value < 0)
    if negative:
      raise ValueError(f"Bias, noise and mount bounds must be >= 0: {negative}")


def osl_actpack_dim(cfg: OslObsCfg) -> int:
  """Encoders 4, gyro 3, accelerometer 3, then gravity 3 and torque 2 if on."""
  return 10 + 3 * cfg.projected_gravity + 2 * cfg.applied_torque


def osl_loadcell_dim(cfg: OslObsCfg) -> int:
  """The ankle cell's force and torque if on, then the socket cell's if on."""
  return 6 * cfg.ankle_load_cell + 6 * cfg.socket_load_cell


def osl_contact_dim(cfg: OslObsCfg) -> int:
  """The heel contact force if on, then the toe's if on."""
  return int(cfg.heel_contact) + int(cfg.toe_contact)


def osl_obs_frame_dim(cfg: OslObsCfg) -> int:
  """One frame of every term: 20 by default (24 for the legacy set)."""
  return (
    osl_actpack_dim(cfg)
    + osl_loadcell_dim(cfg)
    + osl_contact_dim(cfg)
    + OSL_LAST_ACTION_DIM
  )


def osl_obs_dim(cfg: OslObsCfg | None = None) -> int:
  """The whole ``osl`` group, history included: 500 by default (600 for the legacy
  set)."""
  cfg = cfg or OslObsCfg()
  return osl_obs_frame_dim(cfg) * max(cfg.history_length, 1)


def osl_noise_std(term: str, params: Mapping[str, Any]) -> tuple[float, ...] | None:
  """The white noise's std per output component of the ``osl`` term ``term`` under
  its params: what the term adds to every sample when ``randomize``.

  One std per channel the term outputs, in its order, so an extra's std joins
  exactly when its switch does. Stds the params leave out take :class:`OslObsCfg`'s
  defaults. None for ``last_action``, which is never noisy, and any other term.
  """
  defaults = OslObsCfg()

  def std(name: str, n: int) -> tuple[float, ...]:
    return (float(params.get(name, getattr(defaults, name))),) * n

  if term == "actpack":
    stds = std("encoder_pos_noise", 2) + std("encoder_vel_noise", 2)
    stds += std("gyro_noise", 3) + std("acc_noise", 3)
    if osl_switch(params, "projected_gravity"):
      stds += std("gravity_noise", 3)
    if osl_switch(params, "applied_torque"):
      stds += std("applied_torque_noise", 2)
    return stds
  if term == "loadcell":
    cell = std("force_noise", 3) + std("torque_noise", 3)
    return cell * (int(osl_switch(params, "ankle")) + int(osl_switch(params, "socket")))
  if term == "contact":
    return std("contact_noise", 1) * len(contact_sensor_names(params, allow_none=True))
  return None


def osl_switch(params: Mapping[str, Any], name: str) -> bool:
  """A class term's switch under its params; a missing key reads as
  :data:`_SWITCH_DEFAULTS` says."""
  return bool(params.get(name, _SWITCH_DEFAULTS[name]))


def load_cell_sensor_names(params: Mapping[str, Any]) -> list[str]:
  """The sensors the ``loadcell`` term reads under its params, in channel order: the
  ankle cell's force and torque when ``ankle``, then the socket cell's when
  ``socket``.

  Raises:
    ValueError: If both cells are switched off.
  """
  names: list[str] = []
  if osl_switch(params, "ankle"):
    names += [ANKLE_FORCE, ANKLE_LOAD_TORQUE]
  if osl_switch(params, "socket"):
    names += [SOCKET_FORCE, SOCKET_TORQUE]
  if not names:
    raise ValueError(
      "The loadcell term needs at least one load cell: ankle and socket are both off."
    )
  return names


def contact_sensor_names(
  params: Mapping[str, Any], allow_none: bool = False
) -> list[str]:
  """The sensors the ``contact`` term reads under its params, in channel order: the
  heel contact when ``heel``, then the toe contact when ``toe``.

  Raises:
    ValueError: If both are switched off, unless ``allow_none``.
  """
  names = [
    spec.name
    for spec, switch in zip(OSL_CONTACT_SPECS, ("heel", "toe"), strict=True)
    if osl_switch(params, switch)
  ]
  if not names and not allow_none:
    raise ValueError(
      "The contact term needs at least one foot contact sensor: heel and toe are "
      "both off."
    )
  return names


##
# Channels.
##

Channel = tuple[str, str, str | None]
"""``(name, unit, sensor)``: one observation channel and the sensor it reads (None
for derived channels)."""

_XYZ = ("x", "y", "z")
_SPECS: Mapping[str, OslSensorSpec] = MappingProxyType(
  {spec.name: spec for spec in (*OSL_SENSOR_SPECS, *OSL_CONTACT_SPECS)}
)
_JOINT_LABELS = ("knee", "ankle")
"""``OSL_TORQUE_JOINTS`` in order; the action term preserves it."""


def term_channels(name: str, params: Mapping[str, Any]) -> list[Channel]:
  """One frame of an ``osl`` observation term, channel by channel, following the
  term's own params.

  Raises:
    ValueError: For an unknown term.
  """
  if name == "actpack":
    channels = [_scalar(s) for s in (KNEE_POS, ANKLE_POS, KNEE_VEL, ANKLE_VEL)]
    channels += _vector(IMU_GYRO) + _vector(IMU_ACC)
    if osl_switch(params, "projected_gravity"):
      # The IMU's orientation estimate, not a raw channel.
      channels += [(f"projected_gravity_{axis}", "1", None) for axis in _XYZ]
    if osl_switch(params, "applied_torque"):
      channels += [_scalar(KNEE_TORQUE), _scalar(ANKLE_TORQUE)]
    return channels
  if name == "loadcell":
    return [c for sensor in load_cell_sensor_names(params) for c in _vector(sensor)]
  if name == "contact":
    return [_scalar(sensor) for sensor in contact_sensor_names(params)]
  if name == "last_action":
    # The previous command, in normalized units: what the controller sent.
    return [(f"osl_{joint}_command", "1", None) for joint in _JOINT_LABELS]
  raise ValueError(f"Unknown OSL observation term {name!r}.")


def _scalar(sensor: str) -> Channel:
  return sensor, _SPECS[sensor].unit, sensor


def _vector(sensor: str) -> list[Channel]:
  return [(f"{sensor}_{axis}", _SPECS[sensor].unit, sensor) for axis in _XYZ]


def _noise_std(term: str, params: Mapping[str, Any], device: str) -> torch.Tensor:
  stds = osl_noise_std(term, params)
  assert stds is not None
  return torch.tensor(stds, dtype=torch.float32, device=device)


def _env_rows(
  env_ids: torch.Tensor | slice | None, num_envs: int, device: str
) -> torch.Tensor:
  rows = torch.arange(num_envs, device=device)
  if env_ids is None:
    return rows
  if isinstance(env_ids, slice):
    return rows[env_ids]
  return torch.as_tensor(env_ids, device=device).long().reshape(-1)


def _uniform(shape: tuple[int, ...], bound: float, device: str) -> torch.Tensor:
  return (2.0 * torch.rand(shape, device=device) - 1.0) * bound


def _rotate(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  """``matrix @ vector`` over a batch: (B, 3, 3) and (B, 3)."""
  return torch.einsum("bij,bj->bi", matrix, vector)


class osl_actpack(ManagerTermBase):
  """The ActPack bus: ``[q_knee, q_ankle, qd_knee, qd_ankle, gyro (3), acc (3)]``,
  then projected gravity (3) and the applied knee and ankle torque (2) when switched
  on, so (B, 10) by default.

  The encoders read the joint angles plus a per-episode offset: an encoder's zero is
  calibrated by hand and is off a little every time the leg is assembled. The
  velocities carry no offset. The IMU vectors are expressed in a per-episode
  mis-mounted frame, then biased, then clamped to the ActPack's ranges, in that
  order: the mount and the bias are part of what the chip measures, so its
  saturation applies after them. That clamp is the IMU's only saturation (the MuJoCo
  sensors have no ``cutoff``). The accelerometer reads specific force, +9.81 m/s^2
  along y with the shank upright and still. Projected gravity,
  ``mount^T R_site^T (0, 0, -1)``, is what an orientation estimator on the same chip
  would report. White noise comes last, one Gaussian draw per component and sample
  (:func:`osl_noise_std`), so after the clamp and before the manager's delay and
  history.

  Params: ``entity_name``; ``projected_gravity`` and ``applied_torque`` (the
  extras); ``randomize`` (draw the per-episode errors at every reset and add the
  noise; off, the term reads the sensors exactly); the error bounds
  ``encoder_bias_rad``, ``gyro_bias``, ``acc_bias`` (uniform per component) and
  ``imu_mount_deg`` (uniform about each axis); and the noise stds
  ``encoder_pos_noise``, ``encoder_vel_noise``, ``gyro_noise``, ``acc_noise``,
  ``gravity_noise`` and ``applied_torque_noise``, the last two used with their
  extra only. Bounds and stds the params leave out take :class:`OslObsCfg`'s.

  Attributes:
    encoder_bias: (B, 2) the knee and ankle offsets (rad).
    gyro_bias: (B, 3) rad/s, in the IMU's frame.
    acc_bias: (B, 3) m/s^2, in the IMU's frame.
    mount: (B, 3, 3) the IMU's frame in the site's (site vector = mount @ IMU
      vector), the identity when not randomized.
    noise_std: (D,) the white noise's std per output component, added only when
      randomized.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    params = cfg.params
    defaults = OslObsCfg()
    entity = params.get("entity_name", "robot")
    self._projected_gravity = osl_switch(params, "projected_gravity")
    self._applied_torque = osl_switch(params, "applied_torque")
    self._randomize = bool(params.get("randomize", False))
    self._encoder_bias_rad = float(
      params.get("encoder_bias_rad", defaults.encoder_bias_rad)
    )
    self._gyro_bias = float(params.get("gyro_bias", defaults.gyro_bias))
    self._acc_bias = float(params.get("acc_bias", defaults.acc_bias))
    self._mount_rad = math.radians(
      float(params.get("imu_mount_deg", defaults.imu_mount_deg))
    )

    def sensors(*names: str) -> list[Any]:
      return [env.scene[scene_sensor_name(name, entity)] for name in names]

    self._positions = sensors(KNEE_POS, ANKLE_POS)
    self._velocities = sensors(KNEE_VEL, ANKLE_VEL)
    self._gyro, self._acc = sensors(IMU_GYRO, IMU_ACC)
    self._torques = sensors(KNEE_TORQUE, ANKLE_TORQUE) if self._applied_torque else []
    self._site_id = -1
    if self._projected_gravity:
      site = env.sim.mj_model.site(scene_sensor_name(OSL_IMU_SITE, entity))
      self._site_id = int(site.id)

    n, device = self.num_envs, self.device
    self.encoder_bias = torch.zeros(n, 2, device=device)
    self.gyro_bias = torch.zeros(n, 3, device=device)
    self.acc_bias = torch.zeros(n, 3, device=device)
    self.mount = torch.eye(3, device=device).repeat(n, 1, 1)
    self.noise_std = _noise_std("actpack", params, device)
    self._noisy = self._randomize and bool(self.noise_std.any())
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if not self._randomize:
      return
    rows = _env_rows(env_ids, self.num_envs, self.device)
    n = int(rows.numel())
    if n == 0:
      return
    self.encoder_bias[rows] = _uniform((n, 2), self._encoder_bias_rad, self.device)
    self.gyro_bias[rows] = _uniform((n, 3), self._gyro_bias, self.device)
    self.acc_bias[rows] = _uniform((n, 3), self._acc_bias, self.device)
    angles = _uniform((n, 3), self._mount_rad, self.device)
    self.mount[rows] = matrix_from_euler(angles, "XYZ")

  def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
    del params  # Resolved in __init__.
    to_imu = self.mount.transpose(-1, -2)
    gyro = _rotate(to_imu, self._gyro.data) + self.gyro_bias
    acc = _rotate(to_imu, self._acc.data) + self.acc_bias
    parts = [
      torch.cat([s.data for s in self._positions], dim=-1) + self.encoder_bias,
      torch.cat([s.data for s in self._velocities], dim=-1),
      gyro.clamp(-GYRO_RANGE, GYRO_RANGE),
      acc.clamp(-ACC_RANGE, ACC_RANGE),
    ]
    if self._projected_gravity:
      # The site's rotation maps site vectors to world ones, so world (0, 0, -1)
      # in the site's frame is minus the matrix's last row.
      site_to_world = env.sim.data.site_xmat[:, self._site_id].reshape(-1, 3, 3)
      parts.append(_rotate(to_imu, -site_to_world[:, 2, :]))
    if self._applied_torque:
      parts.append(torch.cat([s.data for s in self._torques], dim=-1))
    out = torch.cat(parts, dim=-1)
    if self._noisy:
      out = out + self.noise_std * torch.randn_like(out)
    return out


class osl_load_cells(ManagerTermBase):
  """The load-cell bus: the ankle cell's force (3) and torque (3) when ``ankle``, then
  the socket cell's when ``socket``, so (B, 12) by default and (B, 6) with one cell.

  Each reading is in its site's frame (OpenSim axes, y up the limb), minus its
  static tare: what the cell reads with the leg hanging still, computed once from the
  nominal model (:func:`~..config.sensors.static_load_cell_tare`). A tared cell reads
  zero in swing near the upright pose and minus the transmitted load in stance:
  about minus body weight along y at the socket in single support. The randomization
  adds a per-episode bias, the residual of an imperfect tare and of thermal drift,
  then white noise, one Gaussian draw per component and sample
  (:func:`osl_noise_std`).

  Params: ``entity_name``; ``ankle`` and ``socket`` (include that cell; at least one
  must be on); ``randomize`` (draw
  the biases at every reset and add the noise); ``force_bias`` (N) and
  ``torque_bias`` (N m), uniform per component; and the noise stds ``force_noise``
  and ``torque_noise``. Bounds and stds the params leave out take
  :class:`OslObsCfg`'s.

  Attributes:
    tare: (D,) the subtracted static readings.
    bias: (B, D) the per-episode biases, zero when not randomized.
    noise_std: (D,) the white noise's std per component, added only when
      randomized.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    params = cfg.params
    defaults = OslObsCfg()
    entity = params.get("entity_name", "robot")
    self._randomize = bool(params.get("randomize", False))
    names = load_cell_sensor_names(params)
    self._sensors = [env.scene[scene_sensor_name(name, entity)] for name in names]
    tare = static_load_cell_tare(env.sim.mj_model, prefix=f"{entity}/")
    self.tare = torch.as_tensor(
      np.concatenate([tare[name] for name in names]),
      dtype=torch.float32,
      device=self.device,
    )
    force = float(params.get("force_bias", defaults.force_bias))
    torque = float(params.get("torque_bias", defaults.torque_bias))
    bounds = [force] * 3 + [torque] * 3
    self._bias_bound = torch.tensor(
      bounds * (len(names) // 2), dtype=torch.float32, device=self.device
    )
    self.bias = torch.zeros(self.num_envs, len(self.tare), device=self.device)
    self.noise_std = _noise_std("loadcell", params, self.device)
    self._noisy = self._randomize and bool(self.noise_std.any())
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if not self._randomize:
      return
    rows = _env_rows(env_ids, self.num_envs, self.device)
    n = int(rows.numel())
    if n == 0:
      return
    self.bias[rows] = _uniform((n, len(self.tare)), 1.0, self.device) * self._bias_bound

  def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
    del env, params  # Resolved in __init__.
    readings = torch.cat([s.data for s in self._sensors], dim=-1)
    out = readings - self.tare + self.bias
    if self._noisy:
      out = out + self.noise_std * torch.randn_like(out)
    return out


class osl_foot_contact(ManagerTermBase):
  """The foot-contact bus: the OSL foot's heel contact force (1) when ``heel``, then
  its toe contact force (1) when ``toe``, so (B, 2) by default.

  Each reading is the model's own touch sensor, ``r_foot`` (heel) or ``r_toes``
  (toe), raw in N: the summed normal force of the floor contacts in its zone, >= 0,
  with no tare (it reads 0 unloaded), no scale and no clamp. The actor's running
  normalizer scales it, as it does the human's touch inputs. The ``r_foot`` zone
  covers the heel only under the extended touch sites (``osl_touch_sites``
  heel_extended or feet_extended): under ``original`` it never fires, so the env
  config refuses heel contact there. The randomization mirrors the load cells'
  force channels: a per-episode bias, then white noise, one Gaussian draw per
  component and sample (:func:`osl_noise_std`).

  Params: ``entity_name``; ``heel`` and ``toe`` (include that sensor; the env
  config leaves the term out when both are off); ``randomize`` (draw the biases at
  every reset and add the noise); ``contact_bias`` (N, uniform per channel) and the
  noise std ``contact_noise`` (N). Bounds and stds the params leave out take
  :class:`OslObsCfg`'s.

  Attributes:
    bias: (B, D) the per-episode biases, zero when not randomized.
    noise_std: (D,) the white noise's std per channel, added only when randomized.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    params = cfg.params
    defaults = OslObsCfg()
    entity = params.get("entity_name", "robot")
    self._randomize = bool(params.get("randomize", False))
    specs = [_SPECS[name] for name in contact_sensor_names(params)]
    self._sensors = [env.scene[scene_sensor_name(s.sensor, entity)] for s in specs]
    self._bias_bound = float(params.get("contact_bias", defaults.contact_bias))
    self.bias = torch.zeros(self.num_envs, len(specs), device=self.device)
    self.noise_std = _noise_std("contact", params, self.device)
    self._noisy = self._randomize and bool(self.noise_std.any())
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if not self._randomize:
      return
    rows = _env_rows(env_ids, self.num_envs, self.device)
    n = int(rows.numel())
    if n == 0:
      return
    self.bias[rows] = _uniform((n, self.bias.shape[1]), self._bias_bound, self.device)

  def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
    del env, params  # Resolved in __init__.
    out = torch.cat([s.data for s in self._sensors], dim=-1) + self.bias
    if self._noisy:
      out = out + self.noise_std * torch.randn_like(out)
    return out


def osl_last_action(
  env: ManagerBasedRlEnv, action_name: str = OSL_ACTION_TERM
) -> torch.Tensor:
  """(B, 2) the OSL's previous command in normalized units: the action term's raw
  input clamped to [-1, 1], which is what the controller commanded, since the term
  clips its scaled output at +-2.88 ctrl (raw +-1). Zero after a reset."""
  return env.action_manager.get_term(action_name).raw_action.clamp(-1.0, 1.0)


def osl_observation_group(
  mode: Mode, cfg: OslObsCfg | None = None
) -> ObservationGroupCfg:
  """The ``osl`` observation group: ``actpack``, ``loadcell``, ``contact`` and
  ``last_action``, less the sensor terms whose switches are all off.

  Each term keeps ``history_length`` frames, flattened, so the group is term-major
  and oldest to newest within a term: 500 dims by default (:func:`osl_obs_dim`).

  * ``"train"``: every bus lags by a draw from ``delay_steps``, per env, held for
    the episode; the class terms randomize (per-episode errors, per-component
    white noise).
  * ``"play"``: training's lags, clean readings.
  * ``"validation"``: clean readings at a fixed ``validation_delay_steps`` lag:
    :func:`osl_validation_group` of the training group.

  ``last_action`` is never delayed or corrupted. ``enable_corruption`` is on in
  training only, for any manager-level noise added later; no term has any, as the
  class terms add their own under ``randomize``.
  """
  if mode not in ("train", "validation", "play"):
    raise ValueError(f"Unknown mode {mode!r}.")
  cfg = cfg or OslObsCfg()
  group = _training_group(cfg)
  if mode == "validation":
    return osl_validation_group(group, cfg.validation_delay_steps)
  if mode == "play":
    return _without_randomization(group)
  return group


def osl_validation_group(
  train_group: ObservationGroupCfg,
  validation_delay_steps: int | Mapping[str, int] = 1,
) -> ObservationGroupCfg:
  """The validation ``osl`` group, derived from a training one.

  A copy of ``train_group`` (a config, such as the training env's
  ``cfg.observations["osl"]``) that reads its sensors exactly:
  ``enable_corruption`` off, ``randomize`` off on every term that has it (no
  per-episode errors, no white noise), no manager-level noise, and every bus term
  (:data:`OSL_BUS_TERMS`) at a fixed lag: ``validation_delay_steps``, one lag for
  every bus or one per bus present (:func:`osl_validation_lags` gives the middle of
  each one's training range). Everything else is the training group's however it
  was set, switches and history included, so validation observes what training
  does, at the same dims. For a group from ``osl_observation_group("train", cfg)``
  and ``cfg.validation_delay_steps``, this is ``osl_observation_group("validation",
  cfg)``.

  Raises:
    ValueError: For a negative lag, or a mapping missing a bus of the group.
  """
  present = [name for name in OSL_BUS_TERMS if train_group.terms.get(name) is not None]
  if isinstance(validation_delay_steps, Mapping):
    per_bus = cast(Mapping[str, int], validation_delay_steps)
    missing = [name for name in present if name not in per_bus]
    if missing:
      raise ValueError(f"No validation lag for the buses {missing}.")
    lags = {name: int(per_bus[name]) for name in present}
    values = list(lags.values())
  else:
    values = [int(validation_delay_steps)]
    lags = dict.fromkeys(present, values[0])
  if any(lag < 0 for lag in values):
    raise ValueError(f"validation_delay_steps must be >= 0: {validation_delay_steps}")
  group = _without_randomization(train_group)
  for name, lag in lags.items():
    term = group.terms[name]
    term.delay_min_lag = term.delay_max_lag = lag
  return group


def osl_validation_lags(group: ObservationGroupCfg) -> dict[str, int]:
  """Each bus term's validation lag: the middle of its training range, rounded down
  (``OslObsCfg.validation_delay_steps``'s rule and the bundle manifest's
  ``validation_steps``)."""
  return {
    name: (int(term.delay_min_lag) + int(term.delay_max_lag)) // 2
    for name in OSL_BUS_TERMS
    if (term := group.terms.get(name)) is not None
  }


def _without_randomization(group: ObservationGroupCfg) -> ObservationGroupCfg:
  """A copy of ``group`` whose terms read their sensors exactly, lags kept."""
  group = copy.deepcopy(group)
  group.enable_corruption = False
  for term in group.terms.values():
    if term is None:  # mjlab skips a term set to None.
      continue
    term.noise = None
    if "randomize" in term.params:
      term.params["randomize"] = False
  return group


def _training_group(cfg: OslObsCfg) -> ObservationGroupCfg:
  """The ``"train"`` group, which the other modes copy. Every term param carries in
  every mode; ``randomize`` alone turns the errors and the noise off. A sensor term
  whose switches are all off is left out."""
  history: dict[str, Any] = {
    "history_length": cfg.history_length,
    "flatten_history_dim": True,
  }
  bus: dict[str, Any] = {
    **history,
    "delay_min_lag": cfg.delay_steps[0],
    "delay_max_lag": cfg.delay_steps[1],
    "delay_per_env": True,
    # One lag per env per episode: the buffer draws a lag only for rows reset
    # since its last step.
    "delay_hold_prob": 1.0,
  }
  terms: dict[str, ObservationTermCfg] = {}
  terms["actpack"] = ObservationTermCfg(
    func=osl_actpack,
    params={
      "entity_name": "robot",
      "projected_gravity": cfg.projected_gravity,
      "applied_torque": cfg.applied_torque,
      "randomize": True,
      "encoder_bias_rad": cfg.encoder_bias_rad,
      "gyro_bias": cfg.gyro_bias,
      "acc_bias": cfg.acc_bias,
      "imu_mount_deg": cfg.imu_mount_deg,
      "encoder_pos_noise": cfg.encoder_pos_noise,
      "encoder_vel_noise": cfg.encoder_vel_noise,
      "gyro_noise": cfg.gyro_noise,
      "acc_noise": cfg.acc_noise,
      "gravity_noise": cfg.gravity_noise,
      "applied_torque_noise": cfg.applied_torque_noise,
    },
    **bus,
  )
  if cfg.ankle_load_cell or cfg.socket_load_cell:
    terms["loadcell"] = ObservationTermCfg(
      func=osl_load_cells,
      params={
        "entity_name": "robot",
        "ankle": cfg.ankle_load_cell,
        "socket": cfg.socket_load_cell,
        "randomize": True,
        "force_bias": cfg.force_bias,
        "torque_bias": cfg.torque_bias,
        "force_noise": cfg.force_noise,
        "torque_noise": cfg.torque_noise,
      },
      **bus,
    )
  if cfg.heel_contact or cfg.toe_contact:
    terms["contact"] = ObservationTermCfg(
      func=osl_foot_contact,
      params={
        "entity_name": "robot",
        "heel": cfg.heel_contact,
        "toe": cfg.toe_contact,
        "randomize": True,
        "contact_bias": cfg.contact_bias,
        "contact_noise": cfg.contact_noise,
      },
      **bus,
    )
  terms["last_action"] = ObservationTermCfg(
    func=osl_last_action, params={"action_name": OSL_ACTION_TERM}, **history
  )
  return ObservationGroupCfg(
    terms=dict(terms), concatenate_terms=True, enable_corruption=True
  )


def osl_sensor_group(group: ObservationGroupCfg) -> ObservationGroupCfg:
  """``group`` under the drop rule: without the sensor terms whose switches are all
  off (and terms set to None). ``group`` itself when nothing is dropped, a copy
  with a new terms dict otherwise.

  Raises:
    ValueError: For a term that is not one of :data:`OSL_OBS_TERMS`, or a group
      left without a sensor term.
  """
  unknown = [name for name in group.terms if name not in OSL_OBS_TERMS]
  if unknown:
    raise ValueError(
      f"Unknown osl observation terms {unknown}; the OSL actor's are "
      f"{list(OSL_OBS_TERMS)}."
    )
  terms = {
    name: term
    for name, term in group.terms.items()
    if term is not None
    and not (
      name in SENSOR_SWITCHES
      and not any(osl_switch(term.params, s) for s in SENSOR_SWITCHES[name])
    )
  }
  if not any(name in OSL_BUS_TERMS for name in terms):
    raise ValueError(
      "The osl group needs at least one sensor term: every sensor is switched off."
    )
  if list(terms) == list(group.terms):
    return group
  return replace(group, terms=terms)


##
# Records: a run's group as plain data, and the group rebuilt from one.
##

OSL_OBSERVATION_KEY = "osl_observation"
"""The top-level key of every stage-2 (and stage-1.5) checkpoint that records the
``osl`` group (since 2026-11-14): see :func:`osl_observation_entry`."""
OSL_OBSERVATION_FORMAT = "mjlab.osl_split.osl_observation/1"
TERM_SETTINGS: tuple[str, ...] = (
  "history_length",
  "flatten_history_dim",
  "delay_min_lag",
  "delay_max_lag",
  "delay_per_env",
  "delay_hold_prob",
  "delay_update_period",
  "delay_per_env_phase",
)
"""The ``osl`` term settings a record keeps and a rebuild restores, with the
params."""
GROUP_SETTINGS: tuple[str, ...] = (
  "concatenate_terms",
  "enable_corruption",
  "history_length",
  "flatten_history_dim",
)
"""The ``osl`` group settings a rebuild restores."""
TERM_FUNCS: Mapping[str, Any] = MappingProxyType(
  {
    "actpack": osl_actpack,
    "loadcell": osl_load_cells,
    "contact": osl_foot_contact,
    "last_action": osl_last_action,
  }
)
"""The registry :func:`osl_group_from_record` builds from."""
FLAG_PREFIX = "--env.observations.osl"
_EXTRAS = ("projected_gravity", "applied_torque")
"""The actpack switches that add channels."""


class OslTermLayout(NamedTuple):
  """One term of an ``osl`` group's input layout."""

  name: str
  channels: tuple[str, ...]
  history: int
  """Frames held (at least 1)."""


def osl_group_record(group: ObservationGroupCfg) -> dict[str, Any]:
  """``group`` as plain data, in the shape of a run's ``params/env.yaml``
  ``observations.osl`` read as data (functions as dotted paths)."""
  terms: dict[str, Any] = {}
  for name, term in group.terms.items():
    if term is None:
      continue
    func = term.func
    terms[name] = {
      "func": f"{func.__module__}.{func.__name__}",
      "params": _plain(term.params),
      "noise": None if term.noise is None else repr(term.noise),
      "clip": _plain(term.clip),
      "scale": _plain(term.scale),
      **{key: _plain(getattr(term, key)) for key in TERM_SETTINGS},
    }
  return {
    "terms": terms,
    **{key: _plain(getattr(group, key)) for key in GROUP_SETTINGS},
  }


def osl_group_from_record(
  record: Mapping[str, Any], train: bool = True
) -> ObservationGroupCfg:
  """The ``osl`` group a record describes (:func:`osl_group_record`, or a run's
  ``params/env.yaml`` ``observations.osl`` read as data), rebuilt from this
  checkout's terms (:data:`TERM_FUNCS`) in the record's order, with its params,
  history and latency. ``train`` keeps the record's randomization; otherwise the
  terms read their sensors exactly (lags kept), as play does. Params the record
  leaves out read as each term's defaults (a missing switch as
  :data:`_SWITCH_DEFAULTS` says), so old records rebuild what they trained.

  Raises:
    ValueError: For an unknown term, another function, manager-level noise, a clip
      or a scale (outside the bundle schema), a history that is not flattened, or a
      group that is not concatenated.
  """
  terms: dict[str, ObservationTermCfg] = {}
  for name, saved in record["terms"].items():
    if saved is None:
      continue
    func = TERM_FUNCS.get(name)
    if func is None:
      raise ValueError(
        f"observations.osl.terms has an unknown term {name!r}; this checkout knows "
        f"{list(TERM_FUNCS)}."
      )
    path = f"{func.__module__}.{func.__name__}"
    if saved["func"] != path:
      raise ValueError(f"observations.osl.{name}.func {saved['func']} -> {path}")
    processed = [k for k in ("noise", "clip", "scale") if saved.get(k) is not None]
    if processed:
      raise ValueError(f"observations.osl.{name} sets {processed}, outside the schema.")
    term = ObservationTermCfg(func=func, params=copy.deepcopy(dict(saved["params"])))
    for key in TERM_SETTINGS:
      if key in saved:
        setattr(term, key, saved[key])
    if term.history_length and not term.flatten_history_dim:
      raise ValueError(f"observations.osl.{name} keeps an unflattened history.")
    terms[name] = term
  group = ObservationGroupCfg(terms=terms)
  for key in GROUP_SETTINGS:
    if key in record:
      setattr(group, key, record[key])
  if not group.concatenate_terms or not group.flatten_history_dim:
    raise ValueError("The osl group must be concatenated and flattened.")
  group = osl_sensor_group(group)
  return group if train else _without_randomization(group)


def _as_record(group: ObservationGroupCfg | Mapping[str, Any]) -> Mapping[str, Any]:
  return osl_group_record(group) if isinstance(group, ObservationGroupCfg) else group


def osl_obs_layout(
  group: ObservationGroupCfg | Mapping[str, Any],
) -> tuple[OslTermLayout, ...]:
  """The input layout of a group or record: each term's channels and history, in
  order. Two groups feed a network alike exactly when their layouts are equal."""
  record = _as_record(group)
  shared = record.get("history_length")
  layout = []
  for name, term in record["terms"].items():
    if term is None:
      continue
    history = shared if shared is not None else term.get("history_length", 0)
    channels = tuple(c for c, _, _ in term_channels(name, term["params"]))
    layout.append(OslTermLayout(name, channels, max(int(history), 1)))
  return tuple(layout)


def osl_layout_dim(layout: tuple[OslTermLayout, ...]) -> int:
  """The input size of a layout."""
  return sum(len(term.channels) * term.history for term in layout)


def osl_sensor_set(group: ObservationGroupCfg | Mapping[str, Any]) -> dict[str, bool]:
  """Which of the leg's sensors a group or record reads (the bundle manifest's
  ``sensor_set``)."""
  terms = {
    name: term["params"]
    for name, term in _as_record(group)["terms"].items()
    if term is not None
  }

  def on(term: str, switch: str) -> bool:
    return term in terms and osl_switch(terms[term], switch)

  return {
    "encoders": "actpack" in terms,
    "imu": "actpack" in terms,
    "ankle_load_cell": on("loadcell", "ankle"),
    "socket_load_cell": on("loadcell", "socket"),
    "heel_contact": on("contact", "heel"),
    "toe_contact": on("contact", "toe"),
  }


def describe_osl_sensors(group: ObservationGroupCfg | Mapping[str, Any]) -> str:
  """A one-line summary of a group or record: its sensors, history and size."""
  sensors = osl_sensor_set(group)
  names = {
    "encoders": "encoders",
    "imu": "IMU",
    "ankle_load_cell": "ankle load cell",
    "socket_load_cell": "socket load cell",
    "heel_contact": "heel contact",
    "toe_contact": "toe contact",
  }
  layout = osl_obs_layout(group)
  histories = sorted({term.history for term in layout})
  frames = "/".join(str(h) for h in histories)
  on = ", ".join(names[key] for key, value in sensors.items() if value)
  return f"{on}; {frames} frames, {osl_layout_dim(layout)} inputs"


def osl_sensor_flags(
  record: ObservationGroupCfg | Mapping[str, Any],
  default: ObservationGroupCfg | Mapping[str, Any] | None = None,
) -> str:
  """The train flags that rebuild a record's input layout (sensors, extras,
  history) from the default group, e.g. a two-cell run without contacts:
  ``--env.observations.osl.terms.loadcell.params.socket True
  --env.observations.osl.terms.contact.params.heel False
  --env.observations.osl.terms.contact.params.toe False``. Empty for the default
  set. Lags, errors and noise are not part of the layout.

  Args:
    record: A group or record (:func:`osl_group_record`, or an env.yaml's).
    default: The group the flags apply to (default: the stage-2 training group,
      ``osl_observation_group("train")``).
  """
  wanted = _as_record(record)
  base = _as_record(
    default if default is not None else osl_observation_group("train", OslObsCfg())
  )

  def params(rec: Mapping[str, Any], term: str) -> Mapping[str, Any] | None:
    entry = rec["terms"].get(term)
    return None if entry is None else entry["params"]

  flags: list[str] = []
  unreachable: list[str] = []
  for (term, switch), _ in SWITCH_FIELDS.items():
    want, have = params(wanted, term), params(base, term)
    on = want is not None and osl_switch(want, switch)
    if on != (have is not None and osl_switch(have, switch)):
      if have is None:
        unreachable.append(term)
      flags.append(f"{FLAG_PREFIX}.terms.{term}.params.{switch} {on}")
  want, have = params(wanted, "actpack") or {}, params(base, "actpack") or {}
  for extra in _EXTRAS:
    if osl_switch(want, extra) != osl_switch(have, extra):
      flag = extra.replace("_", "-")
      flags.append(
        f"{FLAG_PREFIX}.terms.actpack.params.{flag} {osl_switch(want, extra)}"
      )
  wanted_layout = osl_obs_layout(wanted)
  base_history = {t.name: t.history for t in osl_obs_layout(base)}
  histories = {t.name: t.history for t in wanted_layout}
  if len(set(histories.values())) == 1:
    (history,) = set(histories.values())
    if any(base_history.get(name, history) != history for name in histories):
      flags.append(f"{FLAG_PREFIX}.history-length {history}")
  else:
    flags += [
      f"{FLAG_PREFIX}.terms.{name}.history-length {history}"
      for name, history in histories.items()
      if base_history.get(name) != history
    ]
  line = " ".join(flags)
  if unreachable:
    line += (
      f" (the default group has no {sorted(set(unreachable))} term, which no flag "
      "adds back)"
    )
  return line


def osl_observation_entry(group: ObservationGroupCfg) -> dict[str, Any]:
  """What a checkpoint records under :data:`OSL_OBSERVATION_KEY`: the training
  ``osl`` group as plain data (:func:`osl_group_record`), its sensor set and its
  input size."""
  record = osl_group_record(group)
  return {
    "format": OSL_OBSERVATION_FORMAT,
    "group": record,
    "sensor_set": osl_sensor_set(record),
    "dim": osl_layout_dim(osl_obs_layout(record)),
  }


def legacy_osl_record() -> dict[str, Any]:
  """The record of a run that records none: the legacy set's training group."""
  return osl_group_record(
    osl_observation_group("train", OslObsCfg(**LEGACY_OSL_SENSORS))
  )


def _plain(value: Any) -> Any:
  """Plain data: tuples become lists, tensors and arrays lists."""
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().tolist()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Mapping):
    return {str(k): _plain(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_plain(v) for v in value]
  return value
