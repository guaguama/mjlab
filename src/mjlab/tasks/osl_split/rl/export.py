"""The OSL actor as a deployable artifact: ONNX plus a metadata JSON.

Of the two stage-2 actors only the OSL's is meant to leave simulation: it reads the
prosthesis's own sensors and drives its two motors. A controller on the device has
to rebuild its input exactly, so every export carries, next to the network:

* ``obs``: the ``osl`` observation's layout, term by term (offset, frame size,
  history length, the channel of every index and its unit). Terms are concatenated
  in order, each holding its frames oldest first: index ``offset + t * frame_dim +
  c`` is channel ``c`` of frame ``t``. The network normalizes its input itself
  (the frozen running statistics are part of the graph), so it takes raw values;
* ``sensors``: what each channel reads, where (site or joint), in which frame and
  unit, and the load cells' tare, which the observation subtracts;
* ``action``: how the two outputs become torque: clipped to [-1, 1], times the
  2.88 ctrl scale, times each motor's gain (N m per ctrl unit, read from the
  model), at the control period ``dt``;
* ``randomization``: the perception and actuation ranges the policy was trained
  under. They describe the robustness it was trained for; nothing in them is
  applied at inference;
* ``std``: the actor's sampling std (normalized action units) at export;
* ``lineage``: the stage-1 run stage 2 started from.

The ONNX graph (input ``obs`` (1, D), output ``actions`` (1, 2)) is exported the
way :meth:`~mjlab.rl.runner.MjlabOnPolicyRunner.export_policy_to_onnx` exports a
policy, and the same metadata is embedded in its ``metadata_props``, one JSON
string per top-level key, when ``onnx`` is importable.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OSL_TORQUE_JOINTS
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
  OSL_IMU_BODY,
  OSL_IMU_POS,
  OSL_IMU_SITE,
  OSL_SENSOR_SPECS,
  SOCKET_FORCE,
  SOCKET_TORQUE,
  scene_sensor_name,
)
from mjlab.tasks.osl_split.mdp.observations import osl_noise_std

from .layout import OSL_ACTION_TERM
from .ppo import OSL_OBS_GROUP, TwoAgentPPO

_onnx: ModuleType | None
try:
  _onnx = importlib.import_module("onnx")
except ImportError:  # onnxscript pulls onnx in; the JSON is written either way.
  _onnx = None

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

OSL_POLICY_FORMAT = "mjlab.osl_split.osl_policy/1"
"""Stored under ``"format"`` in every metadata JSON; bump it on layout changes."""
EXPORT_DIR = "exported"
"""The run directory's subdirectory the runner exports into."""
OBS_LAYOUT = "term-major, oldest to newest"
FRAMES = (
  "The OSL bodies' OpenSim axes: x anterior, y along the limb (up), z to the right. "
  "The knee assembly, tibial pylon and ankle assembly are one rigid shank, so the "
  "IMU and the ankle load cell share their axes."
)
ENTITY = "robot"
"""The entity the OSL belongs to in mjlab's scene (names are prefixed with it)."""

Channel = tuple[str, str, str | None]
"""``(name, unit, sensor)``: one observation channel and the sensor it reads (None
for derived channels)."""

_XYZ = ("x", "y", "z")
_UNITS = {spec.name: spec.unit for spec in OSL_SENSOR_SPECS}
_SATURATION = {IMU_GYRO: GYRO_RANGE, IMU_ACC: ACC_RANGE}
"""Per component: the ActPack's ranges, which the ``actpack`` term clamps the IMU to
after its mount rotation and bias (the MuJoCo sensors have no cutoff)."""
_JOINT_LABELS = ("knee", "ankle")
"""``OSL_TORQUE_JOINTS`` in order; the action term preserves it."""
_BIAS_PARAMS = {
  "actpack": ("encoder_bias_rad", "gyro_bias", "acc_bias", "imu_mount_deg"),
  "loadcell": ("force_bias", "torque_bias"),
}
"""The class terms' per-episode error bounds (``mdp.observations``)."""
_OSL_MOTOR_STRENGTH_EVENT = "osl_motor_strength"


def osl_policy_metadata(
  env: ManagerBasedRlEnv, alg: TwoAgentPPO, lineage: Mapping[str, Any] | None
) -> dict[str, Any]:
  """Everything a controller needs to run the exported OSL actor; see the module
  docstring.

  Args:
    env: The env the actor was trained in (its ``osl`` observation group, the OSL
      motor action term and the compiled model). A vec-env wrapper is unwrapped.
    alg: The two-agent algorithm (its OSL actor, std and action layout).
    lineage: The warm start's lineage (plain types), or None.

  Raises:
    ValueError: If the ``osl`` group does not match the OSL actor's input, or holds
      a term this exporter does not know.
  """
  env = getattr(env, "unwrapped", env)
  obs = _obs_metadata(env, alg)
  used = {sensor for term in obs["terms"] for sensor in term["sensors"]}
  std = [float(s) for s in alg.osl_std.detach().cpu().tolist()]
  return {
    "format": OSL_POLICY_FORMAT,
    "io": {
      "input": "obs",
      "input_shape": [1, obs["dim"]],
      "output": "actions",
      "output_shape": [1, len(std)],
    },
    "obs": obs,
    "sensors": _sensor_metadata(env, used),
    "action": _action_metadata(env, alg),
    "randomization": _randomization_metadata(env),
    "std": std,
    "lineage": dict(lineage) if lineage is not None else None,
  }


def export_osl_policy(
  alg: TwoAgentPPO,
  env: ManagerBasedRlEnv,
  out_dir: str | Path,
  name: str,
  lineage: Mapping[str, Any] | None,
) -> Path:
  """Write ``<out_dir>/<name>.onnx`` and ``<out_dir>/<name>.json``.

  The ONNX holds the OSL actor's deterministic output (its mean) behind its frozen
  observation normalizer, on CPU; the training actor is not touched (``as_onnx``
  works on a copy). Returns the ONNX path.
  """
  metadata = osl_policy_metadata(env, alg, lineage)
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  onnx_path = out / f"{name}.onnx"
  model = cast(Any, alg.osl_actor.as_onnx(verbose=False))
  model.to("cpu")
  model.eval()
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),
    str(onnx_path),
    export_params=True,
    opset_version=18,
    verbose=False,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={},
    dynamo=False,
  )
  text = {key: _dumps(value) for key, value in metadata.items()}
  (out / f"{name}.json").write_text(_dumps(metadata, indent=2) + "\n")
  if _onnx is not None:
    graph = _onnx.load(str(onnx_path))
    for key, value in text.items():
      entry = _onnx.StringStringEntryProto()
      entry.key = key
      entry.value = value
      graph.metadata_props.append(entry)
    _onnx.save(graph, str(onnx_path))
  return onnx_path


##
# Sections.
##


def _obs_metadata(env: ManagerBasedRlEnv, alg: TwoAgentPPO) -> dict[str, Any]:
  manager = env.observation_manager
  names = list(manager.active_terms[OSL_OBS_GROUP])
  dims = manager.group_obs_term_dim[OSL_OBS_GROUP]
  terms: list[dict[str, Any]] = []
  offset = 0
  for name, dim in zip(names, dims, strict=True):
    cfg = manager.get_term_cfg(OSL_OBS_GROUP, name)
    size = int(np.prod(dim))
    frames = max(int(cfg.history_length), 1)
    channels = term_channels(name, cfg.params)
    if len(channels) * frames != size:
      raise ValueError(
        f"OSL observation term {name!r} holds {size} values, not {len(channels)} "
        f"channels x {frames} frames."
      )
    terms.append(
      {
        "name": name,
        "offset": offset,
        "dim": size,
        "frame_dim": len(channels),
        "history_length": frames,
        "channels": [channel for channel, _, _ in channels],
        "units": [unit for _, unit, _ in channels],
        "sensors": [
          sensor
          for sensor in dict.fromkeys(sensor for _, _, sensor in channels)
          if sensor is not None
        ],
      }
    )
    offset += size
  actor = alg.osl_actor
  if list(actor.obs_groups) != [OSL_OBS_GROUP] or int(actor.obs_dim) != offset:
    raise ValueError(
      f"The OSL actor reads {list(actor.obs_groups)} ({actor.obs_dim} values), not "
      f"the {offset}-value {OSL_OBS_GROUP!r} group."
    )
  return {
    "group": OSL_OBS_GROUP,
    "dim": offset,
    "layout": OBS_LAYOUT,
    "index": "obs[offset + t * frame_dim + c] is channel c of frame t (t = 0 oldest)",
    "normalization": "inside the network (frozen running mean and variance)",
    "terms": terms,
  }


def term_channels(name: str, params: Mapping[str, Any]) -> list[Channel]:
  """One frame of an ``osl`` observation term, channel by channel, following the
  term's own params (``mdp.observations``)."""
  if name == "actpack":
    channels = [_scalar(s) for s in (KNEE_POS, ANKLE_POS, KNEE_VEL, ANKLE_VEL)]
    channels += _vector(IMU_GYRO) + _vector(IMU_ACC)
    if params.get("projected_gravity", False):
      # The IMU's orientation estimate, not a raw channel.
      channels += [(f"projected_gravity_{axis}", "1", None) for axis in _XYZ]
    if params.get("applied_torque", False):
      channels += [_scalar(KNEE_TORQUE), _scalar(ANKLE_TORQUE)]
    return channels
  if name == "loadcell":
    channels = _vector(ANKLE_FORCE) + _vector(ANKLE_LOAD_TORQUE)
    if params.get("socket", True):
      channels += _vector(SOCKET_FORCE) + _vector(SOCKET_TORQUE)
    return channels
  if name == "last_action":
    # The previous command, in normalized units: what the controller sent.
    return [(f"osl_{joint}_command", "1", None) for joint in _JOINT_LABELS]
  raise ValueError(f"Unknown OSL observation term {name!r}.")


def _scalar(sensor: str) -> Channel:
  return sensor, _UNITS[sensor], sensor


def _vector(sensor: str) -> list[Channel]:
  return [(f"{sensor}_{axis}", _UNITS[sensor], sensor) for axis in _XYZ]


def _sensor_metadata(env: ManagerBasedRlEnv, used: set[str]) -> dict[str, Any]:
  tare = _load_cell_tare(env)
  sensors = []
  for spec in OSL_SENSOR_SPECS:
    entry: dict[str, Any] = {
      "name": spec.name,
      "scene_name": scene_sensor_name(spec.name, ENTITY),
      "type": spec.type.name,
      "object_type": spec.objtype.name,
      "object": spec.objname,
      "dim": spec.dim,
      "unit": spec.unit,
      "saturation": _SATURATION.get(spec.name),
      "observed": spec.name in used,
    }
    if spec.name in tare:
      entry["tare"] = tare[spec.name]
    sensors.append(entry)
  return {
    "frames": FRAMES,
    "imu": {
      "site": OSL_IMU_SITE,
      "body": OSL_IMU_BODY,
      "pos": list(OSL_IMU_POS),
      "gyro_range": GYRO_RANGE,
      "acc_range": ACC_RANGE,
      "accelerometer": "specific force: +9.81 m/s^2 along y, shank upright and still",
    },
    "load_cell_tare": {
      "values": tare,
      "meaning": (
        "the static reading with the leg hanging still (the cell carries what lies "
        "below it); the observation is the reading minus this"
      ),
    },
    "sensors": sensors,
  }


def _load_cell_tare(env: ManagerBasedRlEnv) -> dict[str, list[float]]:
  """The tare the ``loadcell`` term subtracts, by sensor (empty without it)."""
  manager = env.observation_manager
  if "loadcell" not in manager.active_terms[OSL_OBS_GROUP]:
    return {}
  cfg = manager.get_term_cfg(OSL_OBS_GROUP, "loadcell")
  names = [ANKLE_FORCE, ANKLE_LOAD_TORQUE]
  if cfg.params.get("socket", True):
    names += [SOCKET_FORCE, SOCKET_TORQUE]
  values = [float(v) for v in torch.as_tensor(cfg.func.tare).reshape(-1).tolist()]
  if len(values) != 3 * len(names):
    raise ValueError(f"The load-cell tare has {len(values)} values for {names}.")
  return {name: values[3 * i : 3 * i + 3] for i, name in enumerate(names)}


def _action_metadata(env: ManagerBasedRlEnv, alg: TwoAgentPPO) -> dict[str, Any]:
  term = cast(Any, env.action_manager.get_term(OSL_ACTION_TERM))
  joints = [str(j) for j in term.target_names]
  scale = _per_joint(term.scale, len(joints))
  gains, ctrl_range = _motor_gains(env.sim.mj_model, joints)
  dt = float(env.step_dt)
  return {
    "term": OSL_ACTION_TERM,
    "joints": joints,
    "env_rows": alg.layout.rows("osl"),
    "normalized_clip": [-1.0, 1.0],
    "scale": scale,
    "ctrl_range": ctrl_range,
    "gains": gains,
    "gain_unit": "N m per ctrl unit",
    "torque_limit": [g * hi for g, (_, hi) in zip(gains, ctrl_range, strict=True)],
    "torque": "tau = gain * clip(scale * clip(a, -1, 1), ctrl_lo, ctrl_hi)",
    "dt": dt,
    "control_hz": 1.0 / dt,
  }


def _per_joint(value: Any, n: int) -> list[float]:
  if isinstance(value, torch.Tensor):
    return [float(v) for v in value.reshape(-1, n)[0].tolist()]
  return [float(value)] * n


def _motor_gains(
  model: mujoco.MjModel, joints: list[str]
) -> tuple[list[float], list[list[float]]]:
  """Each joint's motor: its force per ctrl unit (gear times gain, so the robot
  model's gear and the musculoskeletal model's gain read alike) and ctrl range."""
  gains: list[float] = []
  ranges: list[list[float]] = []
  for joint in joints:
    joint_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, scene_sensor_name(joint, ENTITY)
    )
    actuators = [
      a
      for a in range(model.nu)
      if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
      and int(model.actuator_trnid[a, 0]) == joint_id
    ]
    if joint_id < 0 or len(actuators) != 1:
      raise ValueError(f"Expected one motor on joint {joint!r}, got {actuators}.")
    a = actuators[0]
    gains.append(float(model.actuator_gear[a, 0] * model.actuator_gainprm[a, 0]))
    ranges.append([float(x) for x in model.actuator_ctrlrange[a]])
  return gains, ranges


def _randomization_metadata(env: ManagerBasedRlEnv) -> dict[str, Any]:
  manager = env.observation_manager
  observation: dict[str, Any] = {}
  for name in manager.active_terms[OSL_OBS_GROUP]:
    cfg = manager.get_term_cfg(OSL_OBS_GROUP, name)
    std = getattr(getattr(cfg, "noise", None), "std", None)
    if cfg.params.get("randomize", False):
      # The class terms add their own white noise (``mdp.observations``).
      std = osl_noise_std(name, cfg.params) or std
    entry: dict[str, Any] = {
      "delay_steps": [int(cfg.delay_min_lag), int(cfg.delay_max_lag)],
      "delay_held_per_episode": float(cfg.delay_hold_prob) >= 1.0,
      "noise_std": _plain(std),
    }
    if cfg.params.get("randomize", False):
      entry["per_episode_bias"] = {
        key: float(cfg.params[key])
        for key in _BIAS_PARAMS.get(name, ())
        if key in cfg.params
      }
    observation[name] = entry
  out: dict[str, Any] = {
    "applies_to": "simulation only; the deployed policy reads the raw sensors",
    "observation": observation,
  }
  delays = _osl_actuator_delay_s(env)
  if delays is not None:
    out["osl_actuator_delay_s"] = delays
  strength = _osl_motor_strength(env)
  if strength is not None:
    out["osl_motor_strength"] = strength
  return out


def _osl_actuator_delay_s(env: ManagerBasedRlEnv) -> list[float] | None:
  """The OSL motors' command delay range (s), drawn per episode in training."""
  entities = getattr(getattr(getattr(env, "cfg", None), "scene", None), "entities", {})
  entity = entities.get(ENTITY) if isinstance(entities, Mapping) else None
  articulation = getattr(entity, "articulation", None)
  for actuator in getattr(articulation, "actuators", ()) or ():
    if tuple(getattr(actuator, "target_names_expr", ())) == OSL_TORQUE_JOINTS:
      dt = float(env.physics_dt)
      return [actuator.delay_min_lag * dt, actuator.delay_max_lag * dt]
  return None


def _osl_motor_strength(env: ManagerBasedRlEnv) -> list[float] | None:
  events = getattr(getattr(env, "cfg", None), "events", None)
  if not isinstance(events, Mapping) or _OSL_MOTOR_STRENGTH_EVENT not in events:
    return None
  scale = events[_OSL_MOTOR_STRENGTH_EVENT].params.get("scale_range")
  return _plain(scale) if scale is not None else None


##
# JSON.
##


def _plain(value: Any) -> Any:
  """Plain JSON types: tensors and arrays become lists, tuples lists."""
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
  if isinstance(value, Path):
    return str(value)
  return value


def _dumps(value: Any, indent: int | None = None) -> str:
  return json.dumps(_plain(value), indent=indent, allow_nan=False)
