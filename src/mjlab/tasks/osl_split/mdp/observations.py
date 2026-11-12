"""The OSL actor's observation: the prosthesis's own sensors, as the device reads them.

The stage-2 OSL actor is meant to run on the prosthesis, so it observes only what the
prosthesis measures (``config/sensors.py``): never the simulation state, the
reference, the commanded twist or the stride clock. Three terms, one per hardware
path:

* ``actpack``: the ActPack bus, the knee and ankle encoders and the shank IMU's gyro
  and accelerometer, plus two off-by-default extras: projected gravity (an
  orientation estimator's output rather than a raw channel) and the joints' applied
  torque (the drive's current-based estimate);
* ``loadcell``: the load-cell bus, the six-axis cell above the ankle and the one below
  the socket, both on by default (either can be switched off, not both), both tared;
* ``last_action``: the controller's own previous command, which crosses no bus.

Perception randomization models the hardware and applies in training only, from the
first stage-2 iteration:

* per-episode errors, drawn at every reset by the class terms: encoder offsets, IMU
  and load-cell biases, and a small IMU mount rotation;
* white noise on every sensor channel, one Gaussian std per component, added by the
  class terms as the last step of their output, so before the manager's delay and
  history;
* latency, one lag per bus, drawn per env at every reset and held for the episode
  (the manager's ``DelayBuffer`` with ``delay_hold_prob=1``, one buffer per term):
  the channels of a bus share their lag, and the two buses lag independently.

A class term reads its switches, error bounds and noise stds from its own params,
and its ``randomize`` param turns both the errors and the noise on. Its output and
its noise therefore follow the same params however they are set: through
:class:`OslObsCfg`, or one at a time on the train CLI (e.g.
``--env.observations.osl.terms.actpack.params.projected-gravity True``).

Validation reads the sensors exactly, at a fixed one-step lag; play reads them
exactly with training's lags. Both are the training group with its randomization
off (:func:`osl_validation_group` for validation), so they observe what training
does, at its dims. The previous command is always clean and immediate. Every term
keeps the same history, flattened term-major and oldest to newest.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
  OSL_IMU_SITE,
  SOCKET_FORCE,
  SOCKET_TORQUE,
  scene_sensor_name,
  static_load_cell_tare,
)
from mjlab.tasks.osl_split.rl.layout import OSL_ACTION_TERM
from mjlab.utils.lab_api.math import matrix_from_euler

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.osl_mimic.config.env_cfgs import Mode

OSL_OBS_TERMS: tuple[str, str, str] = ("actpack", "loadcell", "last_action")
"""The ``osl`` group's terms, in order."""
OSL_LAST_ACTION_DIM = len(OSL_TORQUE_JOINTS)
"""The knee and ankle commands."""
_BUS_TERMS = OSL_OBS_TERMS[:2]
"""The terms that cross a hardware bus, so lag."""
_SWITCH_DEFAULTS = {
  "projected_gravity": False,
  "applied_torque": False,
  "ankle": True,
  "socket": True,
}
"""A class term's switches when its params leave them out."""


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
  socket_load_cell: bool = True
  """The six-axis cell below the socket. At least one of the two cells must be on."""
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

  def __post_init__(self) -> None:
    lo, hi = self.delay_steps
    if not 0 <= lo <= hi:
      raise ValueError(f"delay_steps must satisfy 0 <= min <= max: {self.delay_steps}")
    if self.history_length < 0 or self.validation_delay_steps < 0:
      raise ValueError("history_length and validation_delay_steps must be >= 0.")
    if not (self.ankle_load_cell or self.socket_load_cell):
      raise ValueError("At least one load cell (ankle or socket) must be on.")
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


def osl_obs_frame_dim(cfg: OslObsCfg) -> int:
  """One frame of all three terms: 24 by default."""
  return osl_actpack_dim(cfg) + osl_loadcell_dim(cfg) + OSL_LAST_ACTION_DIM


def osl_obs_dim(cfg: OslObsCfg | None = None) -> int:
  """The whole ``osl`` group, history included: 600 by default."""
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
    if _switch(params, "projected_gravity"):
      stds += std("gravity_noise", 3)
    if _switch(params, "applied_torque"):
      stds += std("applied_torque_noise", 2)
    return stds
  if term == "loadcell":
    cell = std("force_noise", 3) + std("torque_noise", 3)
    return cell * (int(_switch(params, "ankle")) + int(_switch(params, "socket")))
  return None


def _switch(params: Mapping[str, Any], name: str) -> bool:
  return bool(params.get(name, _SWITCH_DEFAULTS[name]))


def load_cell_sensor_names(params: Mapping[str, Any]) -> list[str]:
  """The sensors the ``loadcell`` term reads under its params, in channel order: the
  ankle cell's force and torque when ``ankle``, then the socket cell's when
  ``socket``.

  Raises:
    ValueError: If both cells are switched off.
  """
  names: list[str] = []
  if _switch(params, "ankle"):
    names += [ANKLE_FORCE, ANKLE_LOAD_TORQUE]
  if _switch(params, "socket"):
    names += [SOCKET_FORCE, SOCKET_TORQUE]
  if not names:
    raise ValueError(
      "The loadcell term needs at least one load cell: ankle and socket are both off."
    )
  return names


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
    self._projected_gravity = _switch(params, "projected_gravity")
    self._applied_torque = _switch(params, "applied_torque")
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
  """The ``osl`` observation group: ``actpack``, ``loadcell`` and ``last_action``.

  Each term keeps ``history_length`` frames, flattened, so the group is term-major
  and oldest to newest within a term: 600 dims by default (:func:`osl_obs_dim`).

  * ``"train"``: both buses lag by a draw from ``delay_steps``, per env, held for
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
  train_group: ObservationGroupCfg, validation_delay_steps: int = 1
) -> ObservationGroupCfg:
  """The validation ``osl`` group, derived from a training one.

  A copy of ``train_group`` (a config, such as the training env's
  ``cfg.observations["osl"]``) that reads its sensors exactly:
  ``enable_corruption`` off, ``randomize`` off on every term that has it (no
  per-episode errors, no white noise), no manager-level noise, and ``actpack`` and
  ``loadcell`` at a fixed lag of ``validation_delay_steps``. Everything else is the
  training group's however it was set, switches and history included, so
  validation observes what training does, at the same dims. For a group from
  ``osl_observation_group("train", cfg)`` and ``cfg.validation_delay_steps``, this
  is ``osl_observation_group("validation", cfg)``.

  Raises:
    ValueError: If ``validation_delay_steps`` is negative.
  """
  if validation_delay_steps < 0:
    raise ValueError(f"validation_delay_steps must be >= 0: {validation_delay_steps}")
  group = _without_randomization(train_group)
  for name in _BUS_TERMS:
    term = group.terms.get(name)
    if term is not None:
      term.delay_min_lag = term.delay_max_lag = validation_delay_steps
  return group


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
  every mode; ``randomize`` alone turns the errors and the noise off."""
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
  actpack = ObservationTermCfg(
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
  loadcell = ObservationTermCfg(
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
  last_action = ObservationTermCfg(
    func=osl_last_action, params={"action_name": OSL_ACTION_TERM}, **history
  )
  terms = dict(zip(OSL_OBS_TERMS, (actpack, loadcell, last_action), strict=True))
  return ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=True
  )
