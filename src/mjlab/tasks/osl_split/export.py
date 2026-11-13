"""Export a stage-2 split checkpoint as a two-controller deployment bundle, on CPU.

Stage 2 trains two actors that drive one body together: the human actor (the
stage-1 unified actor minus its OSL rows, reading the privileged stage-1
observation) and the OSL actor (a small network on the prosthesis's own sensors,
with an observation history). A bundle carries both, so a consumer such as
osl_deploy can run them side by side in plain CPU MuJoCo:

* ``human_actor.onnx``, ``osl_actor.onnx``: each actor's raw action mean behind
  its frozen observation statistics (input ``observation``, output ``mean``);
* ``manifest.json``: schema 2, kind ``osl_split_dual``. Shared: the model, timing,
  reference settings, solver caps, provenance and checksums. Per controller
  (``controllers.human``, ``controllers.osl``): its ONNX file and I/O, its
  observation contract, its action blocks and its training delays. The human's
  observation and action sections have exactly the single-policy (schema 1)
  format, for the muscle block only; the OSL's action section has that format for
  the motor block, and its observation (``osl_split_sensors_v1``) is the sensor
  contract of :mod:`.rl.export`: channels, sensors, tare, saturation, history,
  latency and reset;
* ``model.mjb``: the compiled stage-2 scene (with the OSL's IMU site and sensors)
  at the deployment solver caps, with the OSL foot touch sites the run trained with
  (the manifest's ``touch_sites`` declares them, read from this file);
* ``reference.npz``: the library as the training command preprocessed it;
* ``parity.npz``: each actor's ONNX outputs on inputs drawn from its own
  statistics, and an OSL observation conformance sequence: CPU MuJoCo states and
  the ``osl`` observation mjlab's own terms and buffers build from them at each
  training latency (see :func:`osl_conformance`);
* ``README.md``: a summary for people (not checksummed).

The human runs on its raw mean by default (stage-2 human actors keep the stage-1
recipe's std of 0.07; ``auto`` picks ``expected`` only for a muscle std above
0.3), the OSL on its mean. The two controllers' action blocks together drive
every actuator exactly once::

  uv run --no-sync python -m mjlab.tasks.osl_split.export \\
    logs/rsl_rl/<split experiment>/<run>/model_<N>.pt \\
    [--library <clip folder>] [--hf-repo <user>/<repo>]

The bundle goes to ``<run>/deploy/model_<N>`` unless ``--output`` says otherwise.
The run's ``params/`` decide the model, socket and reward. The task is rebuilt
from this checkout with the OSL foot touch sites the checkpoint was trained with
(its ``osl_touch_sites`` key, else ``original``), which its warm start's stage 1
must share; its env.yaml must match the rebuilt task (touch sites, simulation,
decimation, the human's observation, actions, actuator delays), and its ``osl``
observation settings (switches, error bounds, history, latency) are taken from the
run.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import numpy as np
import torch
import tyro

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket, TouchSites
from mjlab.tasks.osl_gait_library import export as gl
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.rl.export import (
  Inference,
  PolicyParts,
  action_metadata,
  action_std,
  attach_metadata,
  export_actor_onnx,
  normalizer_samples,
  onnx_parity,
  policy_metadata,
  training_metadata,
)
from mjlab.tasks.osl_mimic.touch_sites import (
  run_touch_sites,
  set_touch_sites,
  touch_sites_manifest,
  touch_sites_readme_line,
)

from .config.env_cfgs import osl_split_env_cfg
from .mdp.observations import osl_validation_group
from .rl.export import DEFAULT_INFERENCE as OSL_INFERENCE
from .rl.export import osl_policy_metadata
from .rl.layout import OSL_ACTION_TERM, ActionLayout, Agent
from .rl.ppo import OSL_OBS_GROUP
from .rl.warmstart import lineage_touch_sites

if TYPE_CHECKING:
  from mjlab.envs.mdp.actions.actions import BaseAction
  from mjlab.managers.observation_manager import ObservationGroupCfg

SCHEMA_VERSION = 2
BUNDLE_KIND = "osl_split_dual"
OSL_OBS_SCHEMA = "osl_split_sensors_v1"
HUMAN: Agent = "human"
OSL: Agent = "osl"
CONTROLLERS: tuple[Agent, Agent] = (HUMAN, OSL)
ONNX_FILES = {HUMAN: "human_actor.onnx", OSL: "osl_actor.onnx"}
CHECKED_FILES = (
  "model.mjb",
  "reference.npz",
  ONNX_FILES[HUMAN],
  ONNX_FILES[OSL],
  "parity.npz",
)
ALGORITHM_CLASS = "mjlab.tasks.osl_split.rl.ppo:TwoAgentPPO"
STATE_KEYS = {HUMAN: "human_actor_state_dict", OSL: "osl_actor_state_dict"}
"""Each actor's state dict in a stage-2 checkpoint (``TwoAgentPPO.save``)."""
MODEL_KEYS = {HUMAN: "actor", OSL: "osl_actor"}
"""Each actor's model config in ``params/agent.yaml`` (also its obs-group key)."""
OBS_GROUPS = {HUMAN: "actor", OSL: OSL_OBS_GROUP}
SOURCE_FOLDERS = (*gl.SOURCE_FOLDERS, "tasks/osl_split")
OSL_SPEC_FN = "mjlab.tasks.osl_split.config.sensors.spec_with_osl_sensors"
"""The run's robot spec_fn: the OSL sensors on top of stage 1's edits."""
_OSL_TERM_SETTINGS = (
  "history_length",
  "flatten_history_dim",
  "delay_min_lag",
  "delay_max_lag",
  "delay_per_env",
  "delay_hold_prob",
  "delay_update_period",
  "delay_per_env_phase",
)
"""The ``osl`` term settings taken from the run, with its params."""
CONFORMANCE_STEPS = 40
CONFORMANCE_TWIST = (1.0, 0.0, 0.5)
"""The conformance sequence plays the clip nearest this (vx, vy, wz): a turning
walk, so every IMU axis moves and both feet load and unload."""
CONFORMANCE_LAST_ACTION = 1.3
"""The previous commands of the conformance sequence are uniform in +-this, so
the observation's clamp to [-1, 1] is exercised."""
LATENCY_NOTE = (
  "Per bus, in control steps. At control step k after a reset (k = 0 is the "
  "reset's own observation) a bus term with lag L holds the frame read at step "
  "max(k - L, 0); its history then stacks these delayed frames. Training draws L "
  "uniformly from training_steps per episode and holds it; mjlab's validation "
  "plays validation_steps, which a deployment should default to."
)
PREVIOUS_COMMAND = (
  "last_action is the OSL controller's own previous output (raw policy units), "
  "clamped to [-1, 1]: what it commanded at the previous control step, before any "
  "actuator delay. It is never delayed and reads 0 at a reset."
)


##
# The run.
##


@dataclass(frozen=True)
class SplitRunSpec:
  """What a stage-2 run trained: from its ``params/agent.yaml``."""

  model: Model
  socket: Socket
  reward: Reward
  experiment: str
  run: str

  @property
  def task_id(self) -> str:
    from .config import split_task_id

    return split_task_id(self.model, self.socket, self.reward)


def split_run_spec(checkpoint: Path, agent: Mapping[str, Any]) -> SplitRunSpec:
  """The run's model, socket and reward, checked to be a stage-2 run whose actors
  this bundle schema covers.

  Raises:
    ValueError: For anything else.
  """
  algorithm = agent.get("algorithm") or {}
  if algorithm.get("class_name") != ALGORITHM_CLASS:
    raise ValueError(
      f"{checkpoint.parent / 'params/agent.yaml'} is not a stage-2 split run "
      f"(algorithm {algorithm.get('class_name')!r}). Export a stage-1 run with "
      "python -m mjlab.tasks.osl_gait_library.export."
    )
  spec = gl.run_spec(checkpoint, dict(agent))
  if spec.variant != "unicycle":
    raise ValueError(f"Stage 2 runs on the unicycle library, not {spec.variant!r}.")
  if agent.get("clip_actions") is not None:
    raise ValueError("Wrapper action clipping is outside the bundle schema.")
  groups = agent.get("obs_groups") or {}
  for name in CONTROLLERS:
    key = MODEL_KEYS[name]
    if list(groups.get(key) or ()) != [OBS_GROUPS[name]]:
      raise ValueError(
        f"The {name} actor must read the {OBS_GROUPS[name]!r} observation group "
        f"alone (obs_groups.{key} = {groups.get(key)})."
      )
  return SplitRunSpec(spec.model, spec.socket, spec.reward, spec.experiment, spec.run)


def load_payload(checkpoint: Path) -> dict[str, Any]:
  """The stage-2 checkpoint, memory-mapped (only the actors are read).

  Raises:
    ValueError: If it is not a stage-2 checkpoint.
  """
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
  missing = [
    STATE_KEYS[name] for name in CONTROLLERS if STATE_KEYS[name] not in payload
  ]
  if missing:
    hint = (
      " It is a stage-1 checkpoint: export it with python -m "
      "mjlab.tasks.osl_gait_library.export."
      if "actor_state_dict" in payload
      else ""
    )
    raise ValueError(f"{checkpoint} is not a stage-2 checkpoint ({missing}).{hint}")
  return payload


def split_config(
  spec: SplitRunSpec, saved: dict[str, Any], *, touch_sites: TouchSites
) -> Any:
  """The run's stage-2 env config, rebuilt from this checkout in train mode, with
  the OSL foot ``touch_sites`` the run trained with.

  The touch sites, the human's observation, the actions, the simulation and the
  actuator delays must match the run's (:func:`~mjlab.tasks.osl_gait_library.
  export.check_actor_config`); the ``osl`` group's terms must be the run's, and
  their params, history and latency are taken from it, so a run trained with OSL
  observation switches flipped on the command line exports as trained.

  Raises:
    ValueError: On drift, or ``osl`` processing outside the schema.
  """
  cfg = osl_split_env_cfg(spec.model, spec.socket, spec.reward, "train")
  set_touch_sites(cfg, touch_sites)
  gl.check_actor_config(cfg, saved)
  spec_fn = str(saved["scene"]["entities"]["robot"].get("spec_fn"))
  if spec_fn != OSL_SPEC_FN:
    raise ValueError(f"The run's robot spec_fn is {spec_fn!r}, not {OSL_SPEC_FN!r}.")
  apply_osl_group(cfg.observations[OSL_OBS_GROUP], saved["observations"][OSL_OBS_GROUP])
  return cfg


def apply_osl_group(group: ObservationGroupCfg, saved: Mapping[str, Any]) -> None:
  """Give ``group`` (the rebuilt ``osl`` group) the run's term params, history and
  latency, in place, after checking it has the run's terms.

  Raises:
    ValueError: For other terms or functions, a clip or scale on a term, or a
      group that is not concatenated and flattened.
  """
  names, saved_names = list(group.terms), list(saved["terms"])
  if names != saved_names:
    raise ValueError(f"observations.osl.terms {saved_names} -> {names}")
  for name, term in group.terms.items():
    original = saved["terms"][name]
    func = f"{term.func.__module__}.{term.func.__name__}"
    if original["func"] != func:
      raise ValueError(f"observations.osl.{name}.func {original['func']} -> {func}")
    processed = [k for k in ("clip", "scale") if original.get(k) is not None]
    if processed:
      raise ValueError(f"observations.osl.{name} sets {processed}, outside the schema.")
    term.params = dict(original["params"])
    for key in _OSL_TERM_SETTINGS:
      setattr(term, key, original[key])
    if term.history_length and not term.flatten_history_dim:
      raise ValueError(f"observations.osl.{name} keeps an unflattened history.")
  for key in ("concatenate_terms", "history_length", "flatten_history_dim"):
    setattr(group, key, saved[key])
  if not group.concatenate_terms or not group.flatten_history_dim:
    raise ValueError("The osl group must be concatenated and flattened.")


##
# CPU stand-in for the env the OSL observation reads.
##


class _Sensor:
  """A scene sensor over a CPU MjData: ``data`` is its reading, (1, dim) float32,
  as mujoco_warp's single-precision sensordata gives it to the terms."""

  def __init__(self, holder: SimpleNamespace, name: str) -> None:
    sensor = holder.model.sensor(name)
    self._holder = holder
    self._slice = slice(int(sensor.adr[0]), int(sensor.adr[0] + sensor.dim[0]))

  @property
  def data(self) -> torch.Tensor:
    values = self._holder.data.sensordata[self._slice]
    return torch.as_tensor(values, dtype=torch.float32).reshape(1, -1)


class _Data:
  def __init__(self, holder: SimpleNamespace) -> None:
    self._holder = holder

  @property
  def site_xmat(self) -> torch.Tensor:
    values = self._holder.data.site_xmat
    return torch.as_tensor(values, dtype=torch.float32).reshape(1, -1, 9)


class CpuOslEnv:
  """What mjlab's ``osl`` observation terms read from an env, over CPU MuJoCo.

  One env. ``scene[name]`` is a sensor of :attr:`data`, ``sim.mj_model`` the
  model, ``action_manager.get_term`` the task's action terms (the OSL term's raw
  action is the previous command), plus the config and timing the export reads.
  With a real :class:`~mjlab.managers.observation_manager.ObservationManager` on
  it, the training code itself builds the ``osl`` observation from CPU state.
  """

  num_envs = 1
  device = "cpu"

  def __init__(
    self,
    cfg: Any,
    model: mujoco.MjModel,
    actions: Mapping[str, BaseAction],
    step_dt: float,
  ) -> None:
    self.cfg = cfg
    self.data = mujoco.MjData(model)
    holder = SimpleNamespace(model=model, data=self.data)
    self._holder = holder
    self.sim = SimpleNamespace(mj_model=model, data=_Data(holder))
    self.actions = dict(actions)
    self.action_manager = SimpleNamespace(get_term=self.actions.__getitem__)
    self.step_dt = float(step_dt)
    self.physics_dt = float(cfg.sim.mujoco.timestep)
    self.observation_manager: Any = None

  @property
  def scene(self) -> Mapping[str, _Sensor]:
    holder = self._holder

    class _Scene:
      def __getitem__(self, name: str) -> _Sensor:
        return _Sensor(holder, name)

    return _Scene()  # type: ignore[return-value]

  def set_state(
    self,
    qpos: np.ndarray,
    qvel: np.ndarray,
    act: np.ndarray,
    ctrl: np.ndarray,
    last_action: np.ndarray,
  ) -> None:
    """Reset the data, set the state and the previous OSL command, and run
    ``mj_forward`` (sensors included)."""
    model, data = self._holder.model, self.data
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:] = qpos, qvel, act, ctrl
    mujoco.mj_forward(model, data)
    command = torch.as_tensor(last_action, dtype=torch.float32).reshape(1, -1)
    self.actions[OSL_ACTION_TERM].process_actions(command)


def observation_manager(env: CpuOslEnv, group: ObservationGroupCfg) -> Any:
  """mjlab's observation manager over ``env`` for the ``osl`` group alone."""
  from mjlab.managers.observation_manager import ObservationManager

  return ObservationManager({OSL_OBS_GROUP: group}, env)


##
# Conformance.
##


def _root_joint(model: mujoco.MjModel) -> tuple[int, int]:
  joint = model.joint("robot/root")
  if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError("robot/root is not a free joint.")
  return int(model.jnt_qposadr[joint.id]), int(model.jnt_dofadr[joint.id])


def reference_states(
  model: mujoco.MjModel,
  command: Any,
  joint_names: list[str],
  clip: int,
  steps: int,
) -> tuple[np.ndarray, np.ndarray]:
  """``steps`` consecutive reference frames of ``clip`` as (qpos, qvel): the root
  pose, its world linear and body angular velocity, and the joints."""
  data = command.data
  frames = int(command.num_frames[clip])
  if frames < steps:
    raise ValueError(f"Clip {clip} has {frames} frames, fewer than {steps}.")
  q0, v0 = _root_joint(model)
  qpos_ids = [int(model.joint(f"robot/{j}").qposadr[0]) for j in joint_names]
  dof_ids = [int(model.joint(f"robot/{j}").dofadr[0]) for j in joint_names]
  qpos = np.tile(model.qpos0, (steps, 1))
  qvel = np.zeros((steps, model.nv))
  for t in range(steps):
    root_quat = data["root_quat"][clip, t].cpu().double().numpy()
    qpos[t, q0 : q0 + 3] = data["root_pos"][clip, t].cpu().double().numpy()
    qpos[t, q0 + 3 : q0 + 7] = root_quat
    qpos[t, qpos_ids] = data["joint_pos"][clip, t].cpu().double().numpy()
    qvel[t, v0 : v0 + 3] = data["root_lin_vel"][clip, t].cpu().double().numpy()
    inverse = np.empty(4)
    mujoco.mju_negQuat(inverse, root_quat)
    omega = data["root_ang_vel_w"][clip, t].cpu().double().numpy()
    mujoco.mju_rotVecQuat(qvel[t, v0 + 3 : v0 + 6], omega, inverse)
    qvel[t, dof_ids] = data["joint_vel"][clip, t].cpu().double().numpy()
  return qpos, qvel


def osl_conformance(
  cfg: Any,
  model: mujoco.MjModel,
  actions: Mapping[str, BaseAction],
  step_dt: float,
  command: Any,
  joint_names: list[str],
  steps: int = CONFORMANCE_STEPS,
  seed: int = 0,
) -> dict[str, np.ndarray]:
  """The ``osl`` observation mjlab builds from a sequence of CPU MuJoCo states.

  The states are ``steps`` consecutive reference frames of a turning clip, each
  with random muscle activations and controls and a random previous OSL command
  (0 at the first step, a reset). At each step the data is reset, the state set,
  ``mj_forward`` run, and mjlab's own observation manager computes the ``osl``
  group with randomization off, as validation does, once per bus latency in the
  training range: the reset's observation first (``compute(update_history=True,
  env_ids=[0])`` after ``reset()``), then one per step. A consumer reproduces each
  ``obs_lag<L>`` from the states alone: it is the whole input contract, sensors,
  tare, saturation, channel order, latency, history and reset.

  Returns:
    ``qpos`` (S, nq), ``qvel`` (S, nv), ``act`` (S, na), ``ctrl`` (S, nu),
    ``last_action`` (S, 2) the raw previous command before the clamp, ``frames``
    (S, F) each step's undelayed frame (the terms' channels concatenated),
    ``lags`` (n,) and ``obs_lag<L>`` (S, D) for each L, ``clip`` the clip index.
  """
  rng = np.random.default_rng(seed)
  twists = command.twists.cpu().double().numpy()
  clip = int(np.argmin(np.linalg.norm(twists - np.asarray(CONFORMANCE_TWIST), axis=1)))
  qpos, qvel = reference_states(model, command, joint_names, clip, steps)
  act = rng.uniform(0.0, 0.3, (steps, model.na))
  low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
  ctrl = rng.uniform(low, high, (steps, model.nu))
  osl_dim = int(actions[OSL_ACTION_TERM].action_dim)
  last_action = rng.uniform(
    -CONFORMANCE_LAST_ACTION, CONFORMANCE_LAST_ACTION, (steps, osl_dim)
  )
  last_action[0] = 0.0
  train = cfg.observations[OSL_OBS_GROUP]
  bus = [t for t in train.terms.values() if t is not None and t.delay_max_lag > 0]
  low_lag = min((int(t.delay_min_lag) for t in bus), default=0)
  high_lag = max((int(t.delay_max_lag) for t in bus), default=0)
  lags = list(range(low_lag, high_lag + 1))
  out: dict[str, np.ndarray] = {
    "qpos": qpos,
    "qvel": qvel,
    "act": act,
    "ctrl": ctrl,
    "last_action": last_action,
    "lags": np.asarray(lags),
    "clip": np.asarray(clip),
  }
  # Lag 0 always runs: its newest frames are the undelayed ones.
  for lag in sorted({0, *lags}):
    env = CpuOslEnv(cfg, model, actions, step_dt)
    env.set_state(qpos[0], qvel[0], act[0], ctrl[0], last_action[0])
    manager = observation_manager(env, osl_validation_group(train, lag))
    rows = []
    for t in range(steps):
      env.set_state(qpos[t], qvel[t], act[t], ctrl[t], last_action[t])
      if t == 0:
        manager.reset(None)
        obs = manager.compute(update_history=True, env_ids=torch.tensor([0]))
      else:
        obs = manager.compute(update_history=True)
      rows.append(obs[OSL_OBS_GROUP][0].double().numpy())
    observations = np.stack(rows)
    if lag == 0:
      out["frames"] = _newest_frames(manager, observations)
    if lag in lags:
      out[f"obs_lag{lag}"] = observations
  return out


def _newest_frames(manager: Any, observations: np.ndarray) -> np.ndarray:
  """Each step's newest frame of every term, concatenated in term order."""
  parts = []
  offset = 0
  for name, dims in zip(
    manager.active_terms[OSL_OBS_GROUP],
    manager.group_obs_term_dim[OSL_OBS_GROUP],
    strict=True,
  ):
    size = int(np.prod(dims))
    frames = max(int(manager.get_term_cfg(OSL_OBS_GROUP, name).history_length), 1)
    width = size // frames
    parts.append(observations[:, offset + size - width : offset + size])
    offset += size
  return np.concatenate(parts, axis=1)


##
# The controllers.
##


def _plain(value: Any) -> Any:
  """JSON types: tensors and arrays become lists, enums their values."""
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
  return getattr(value, "value", value)


def _parts(
  c: SimpleNamespace, actions: Mapping[str, BaseAction], actor_group: bool
) -> PolicyParts:
  """The policy parts of a controller driving ``actions``; with the human's
  observation group when ``actor_group``, without an observation otherwise."""
  actor = c.cfg.observations["actor"]
  return PolicyParts(
    model=c.model,
    joint_names=list(c.joint_names),
    obs_terms=list(actor.terms) if actor_group else [],
    obs_term_cfgs=dict(actor.terms) if actor_group else {},
    actions=actions,
    site_bodies=list(c.command.cfg.site_bodies),
    lookahead_steps=int(c.command.cfg.lookahead_steps),
    decimation=int(c.cfg.decimation),
    step_dt=float(c.step_dt),
    entity_cfg=c.cfg.scene.entities["robot"],
    obs_group_history=actor.history_length if actor_group else None,
  )


def human_controller(
  c: SimpleNamespace, actor: torch.nn.Module, obs_dim: int, inference: Inference | None
) -> dict[str, Any]:
  """The human controller's section: schema-1 ``io``, ``observation``, ``action``
  and ``training`` over its own action blocks."""
  actions = {name: c.actions[name] for name in c.layout.human_terms()}
  parts = _parts(c, actions, actor_group=True)
  std = action_std(actor, c.layout.dim(HUMAN))
  meta = policy_metadata(parts, std, obs_dim, inference, complete=False)
  return {
    "onnx": ONNX_FILES[HUMAN],
    "io": meta["io"],
    "observation": meta["observation"],
    "action": meta["action"],
    "training": meta["training"],
    "memory": None,
  }


def osl_controller(
  c: SimpleNamespace, actor: torch.nn.Module, obs_dim: int, lineage: Any
) -> dict[str, Any]:
  """The OSL controller's section: ``io``, the ``osl_split_sensors_v1``
  observation contract (:mod:`.rl.export`'s, built by mjlab's own observation
  manager over the bundle's model), a schema-1 ``action`` section for the motor
  block, the training delays and randomization, and the motors' torque map."""
  actions = {OSL_ACTION_TERM: c.actions[OSL_ACTION_TERM]}
  parts = _parts(c, actions, actor_group=False)
  std = action_std(actor, c.layout.dim(OSL))
  env = CpuOslEnv(c.cfg, c.model, c.actions, c.step_dt)
  mujoco.mj_forward(c.model, env.data)
  env.observation_manager = observation_manager(env, c.cfg.observations[OSL_OBS_GROUP])
  alg = SimpleNamespace(osl_actor=actor, osl_std=torch.as_tensor(std), layout=c.layout)
  device = osl_policy_metadata(env, alg, lineage)  # type: ignore[arg-type]
  obs = device["obs"]
  if obs["dim"] != obs_dim:
    raise ValueError(
      f"The OSL actor reads {obs_dim} values; the group has {obs['dim']}."
    )
  manager = env.observation_manager
  buses: dict[str, Any] = {}
  for term in obs["terms"]:
    cfg = manager.get_term_cfg(OSL_OBS_GROUP, term["name"])
    lo, hi = int(cfg.delay_min_lag), int(cfg.delay_max_lag)
    if hi > 0:
      buses[term["name"]] = {
        "training_steps": [lo, hi],
        "held_per_episode": float(cfg.delay_hold_prob) >= 1.0
        and int(cfg.delay_update_period) == 0,
        "validation_steps": (lo + hi) // 2,
      }
  randomization = device["randomization"]
  training = training_metadata(parts)
  training["observation_randomization"] = randomization
  return {
    "onnx": ONNX_FILES[OSL],
    "io": {
      "input": "observation",
      "input_shape": [1, obs_dim],
      "output": "mean",
      "output_shape": [1, len(std)],
    },
    "observation": {
      "schema": OSL_OBS_SCHEMA,
      "group": OSL_OBS_GROUP,
      "size": obs_dim,
      "terms": [t["name"] for t in obs["terms"]],
      "term_sizes": {t["name"]: t["dim"] for t in obs["terms"]},
      "layout": obs["layout"],
      "index": obs["index"],
      "normalization": "Frozen checkpoint RunningMeanStd, baked into osl_actor.onnx",
      "term_layout": obs["terms"],
      "latency": {
        "unit": "control steps",
        "buses": buses,
        "default": "validation_steps",
        "note": LATENCY_NOTE,
      },
      "previous_command": PREVIOUS_COMMAND,
      "reset": device["reset"],
      "sensors": device["sensors"],
      "format": device["format"],
    },
    "action": action_metadata(parts, std, OSL_INFERENCE, complete=False),
    "training": training,
    "device": device["action"],
    "memory": {
      "kind": "observation_history",
      "frames": max(int(t["history_length"]) for t in obs["terms"]),
      "owner": (
        "the controller: the ONNX graph is stateless; the consumer keeps each "
        "term's history and each bus's latency buffer and clears them at a reset"
      ),
    },
  }


def check_coverage(model: mujoco.MjModel, sections: Mapping[str, Any]) -> None:
  """The controllers' action blocks drive every actuator exactly once.

  Raises:
    ValueError: On an overlap or a gap.
  """
  ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    for section in sections.values()
    for name in section["action"]["actuator_names"]
  ]
  if len(set(ids)) != len(ids) or sorted(ids) != list(range(model.nu)):
    raise ValueError("The controllers must drive every actuator exactly once.")


##
# The bundle.
##


def _readme(manifest: dict[str, Any]) -> str:
  run, human, osl = (
    manifest["run"],
    manifest["controllers"][HUMAN],
    manifest["controllers"][OSL],
  )
  lineage = manifest.get("lineage") or {}
  validation = manifest["validation"]
  solver = manifest["deployment"]["solver"]
  trained = manifest["deployment"]["training_solver"]

  def delays(section: dict[str, Any]) -> str:
    return ", ".join(
      f"`{name}` {d['seconds'][0] * 1e3:g}-{d['seconds'][1] * 1e3:g} ms"
      if d
      else f"`{name}` unknown"
      for name, d in section["training"]["actuator_delay"].items()
    )

  buses = osl["observation"]["latency"]["buses"]
  latency = ", ".join(
    f"`{name}` {b['training_steps'][0]}-{b['training_steps'][1]} steps "
    f"(validation {b['validation_steps']})"
    for name, b in buses.items()
  )
  stage1 = lineage.get("stage1_checkpoint")
  return f"""# {run["task"]}: {run["run"]} / {run["checkpoint"]}

Two-controller deployment bundle (schema {manifest["schema_version"]},
`{manifest["kind"]}`) exported by mjlab (`mjlab.tasks.osl_split.export`) for
osl_deploy (plain CPU MuJoCo {manifest["mujoco_version"]}).

- Model: {run["model"]}, socket: {run["socket"]}, reward: {run["reward"]}, library:
  {run["variant"]} ({len(manifest["reference"]["clip_names"])} clips)
{touch_sites_readme_line(manifest["touch_sites"])}
- Checkpoint: `{run["checkpoint"]}` (iteration {run["iteration"]}, stage-2
  iteration {run["stage_iteration"]}); warm start: `{stage1}`
- Control: {1.0 / manifest["policy_dt"]:g} Hz for both controllers,
  {manifest["decimation"]} physics steps per action
- Solver caps of `model.mjb`: {solver["iterations"]} Newton,
  {solver["ls_iterations"]} line search (training:
  {trained["iterations"]}/{trained["ls_iterations"]} in mujoco_warp)

## Human controller (`{human["onnx"]}`)

- Observation: {human["observation"]["size"]} values, schema
  `{human["observation"]["schema"]}` (stage 1's), terms
  {", ".join(human["observation"]["terms"])}
- Action: {human["action"]["size"]} outputs, blocks
  {", ".join(f"{b['name']} ({b['size']})" for b in human["action"]["blocks"])};
  default inference **{human["action"]["default_inference"]}**
- Actuator command delay in training: {delays(human)}
- ONNX vs PyTorch max abs error: {validation["human_onnx_max_abs_error"]:.3g}

## OSL controller (`{osl["onnx"]}`)

- Observation: {osl["observation"]["size"]} values, schema
  `{osl["observation"]["schema"]}`: the prosthesis's own sensors, terms
  {", ".join(osl["observation"]["terms"])}, each with
  {osl["memory"]["frames"]} frames of history (term-major, oldest first)
- Sensor latency in training: {latency}
- Action: {osl["action"]["size"]} outputs, blocks
  {", ".join(f"{b['name']} ({b['size']})" for b in osl["action"]["blocks"])};
  default inference **{osl["action"]["default_inference"]}**
- Actuator command delay in training: {delays(osl)}
- ONNX vs PyTorch max abs error: {validation["osl_onnx_max_abs_error"]:.3g}

The OSL network is stateless; the controller keeps the history and latency buffers
(`controllers.osl.observation`: channels, sensors, tare, saturation, latency,
reset). `parity.npz` holds an observation conformance sequence (`osl_*` keys):
CPU MuJoCo states and the observation mjlab's own terms build from them at each
latency.

Compare against the stage-1 teacher by exporting the warm start itself:

```
uv run --no-sync python -m mjlab.tasks.osl_gait_library.export {stage1}
```
"""


def export_bundle(
  checkpoint: str | Path,
  output: str | Path | None = None,
  library: str | Path | None = None,
  human_inference: Inference | None = None,
  overwrite: bool = False,
  solver_iterations: int | None = None,
  ls_solver_iterations: int | None = None,
) -> Path:
  """Export ``checkpoint`` as a two-controller bundle (module docstring).

  Args:
    checkpoint: ``model_<N>.pt`` of a stage-2 split run.
    output: The bundle folder (default ``<run>/deploy/<checkpoint stem>``).
    library: The clip folder, when the run's is not on this machine.
    human_inference: The human's default inference; None picks ``expected`` for a
      median muscle std above 0.3 and ``mean`` otherwise. The OSL's is ``mean``.
    overwrite: Replace an existing bundle (once the new one is complete).
    solver_iterations, ls_solver_iterations: The bundle model's solver caps;
      default the run's, at least 10 and 20.
  """
  checkpoint = Path(checkpoint).resolve()
  output = Path(output or checkpoint.parent / "deploy" / checkpoint.stem).resolve()
  if output.exists() and not overwrite:
    raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
  params = checkpoint.parent / "params"
  agent = gl.read_config(params / "agent.yaml")
  saved = gl.read_config(params / "env.yaml")
  spec = split_run_spec(checkpoint, agent)
  payload = load_payload(checkpoint)
  split = dict((payload.get("infos") or {}).get("split") or {})
  lineage = _plain(split.get("lineage"))
  if lineage and (lineage.get("model"), lineage.get("socket")) != (
    spec.model,
    spec.socket,
  ):
    raise ValueError(f"The lineage is a {lineage.get('model')} run, not {spec.model}.")
  touch = run_touch_sites(checkpoint, payload, saved)
  if lineage and lineage_touch_sites(lineage) != touch.value:
    raise ValueError(
      f"The run trained with the OSL touch sites {touch.value!r}, but its warm "
      f"start's stage 1 with {lineage_touch_sites(lineage)!r}."
    )
  print(f"[INFO] OSL touch sites: {touch.value} (from {touch.source}).")

  cfg = split_config(spec, saved, touch_sites=touch.value)
  env, command, actions, solver = gl.compile_task(
    cfg, saved, library, solver_iterations, ls_solver_iterations
  )
  layout = ActionLayout.from_list(
    [(name, int(term.action_dim)) for name, term in actions.items()]
  )
  if lineage and [list(t) for t in lineage.get("action_terms", [])] != (
    layout.to_list()
  ):
    raise ValueError(
      f"The run's action terms {lineage.get('action_terms')} differ from the "
      f"rebuilt task's {layout.to_list()}."
    )
  c = SimpleNamespace(
    cfg=cfg,
    model=env.sim.mj_model,
    joint_names=list(env.scene["robot"].joint_names),
    command=command,
    actions=actions,
    layout=layout,
    step_dt=float(env.step_dt),
  )
  actors = {
    name: gl.build_actor(
      agent[MODEL_KEYS[name]],
      payload[STATE_KEYS[name]],
      OBS_GROUPS[name],
      layout.dim(name),
    )
    for name in CONTROLLERS
  }

  output.parent.mkdir(parents=True, exist_ok=True)
  tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
  umask = os.umask(0)
  os.umask(umask)
  tmp.chmod(0o777 & ~umask)  # mkdtemp's 0700 would hide the bundle from others.
  try:
    dims = {
      name: export_actor_onnx(actors[name], tmp / ONNX_FILES[name])
      for name in CONTROLLERS
    }
    sections = {
      HUMAN: human_controller(c, actors[HUMAN], dims[HUMAN], human_inference),
      OSL: osl_controller(c, actors[OSL], dims[OSL], lineage),
    }
    check_coverage(c.model, sections)
    iteration = payload.get("iter")
    manifest: dict[str, Any] = {
      "schema_version": SCHEMA_VERSION,
      "kind": BUNDLE_KIND,
      "mujoco_version": mujoco.__version__,
      "decimation": int(cfg.decimation),
      "policy_dt": c.step_dt,
      "root_joint": "robot/root",
      "joint_names": [f"robot/{j}" for j in c.joint_names],
      "socket": spec.socket,
      "touch_sites": touch_sites_manifest(c.model),
      "run": {
        "task": spec.task_id,
        "stage": 2,
        "model": spec.model,
        "socket": spec.socket,
        "reward": spec.reward,
        "variant": "unicycle",
        "experiment": spec.experiment,
        "run": spec.run,
        "checkpoint": checkpoint.name,
        "iteration": int(iteration) if iteration is not None else None,
        "stage_iteration": payload.get("stage_iteration"),
        "stage_start": split.get("stage_start"),
      },
      "reference": {
        "clip_names": list(command.clip_names),
        "policy_dt": c.step_dt,
        "lookahead_steps": command.cfg.lookahead_steps,
        "lookahead_stride": command.cfg.lookahead_stride,
        "blend_time_s": command.cfg.blend_time_s,
        "twist_metric_weights": list(command.cfg.twist_metric_weights),
        "used_by": [HUMAN],
      },
      "deployment": {
        "solver": solver["deployment"],
        "training_solver": solver["training"],
        "note": (
          "Solver caps of model.mjb. Training ran mujoco_warp at training_solver, "
          "where its solves converge; CPU MuJoCo needs a higher cap to converge "
          "(it truncates the contact solve at 4 iterations under the elliptic cone "
          "at impratio 3)."
        ),
      },
      "composition": {
        "controllers": list(CONTROLLERS),
        "env_action_terms": layout.to_list(),
        "actuators": (
          "Each controller maps its own output through its own action blocks "
          "(block starts count within that controller's output) to the actuators "
          "it names; together they drive every actuator exactly once."
        ),
        "timing": (
          "Both controllers act at every control step (policy_dt) on the same "
          "instant: the state after the step's last physics tick, refreshed by "
          "mj_forward, and their actions start together."
        ),
      },
      "controllers": sections,
      "lineage": lineage,
    }
    if manifest["touch_sites"]["version"] != touch.value:
      raise RuntimeError(
        f"model.mjb carries the {manifest['touch_sites']['version']!r} touch sites, "
        f"but the run trained with {touch.value!r}."
      )
    manifest["provenance"] = gl._provenance(
      checkpoint,
      Path(command.cfg.library_dir),
      int(iteration) if iteration is not None else None,
      SOURCE_FOLDERS,
    )
    mujoco.mj_saveModel(c.model, str(tmp / "model.mjb"))
    arrays: dict[str, Any] = {
      key: command.data[key].cpu().numpy() for key in gl.REFERENCE_FIELDS
    }
    arrays["num_frames"] = command.num_frames.cpu().numpy()
    arrays["twists"] = command.twists.cpu().numpy()
    arrays["strides"] = command.strides.cpu().numpy()
    np.savez_compressed(tmp / "reference.npz", **arrays)

    parity: dict[str, Any] = {}
    errors: dict[str, float] = {}
    rtol, atol = gl.PARITY_TOLERANCE
    for name in CONTROLLERS:
      actor = actors[name]
      observations = normalizer_samples(actor, gl.PARITY_SAMPLES)
      means, errors[name] = onnx_parity(tmp / ONNX_FILES[name], actor, observations)
      with torch.inference_mode():
        reference = actor.as_onnx(verbose=False)(torch.from_numpy(observations))  # type: ignore[operator]
      np.testing.assert_allclose(means, reference.numpy(), rtol=rtol, atol=atol)
      parity[f"{name}_observations"] = observations
      parity[f"{name}_means"] = means
    conformance = osl_conformance(
      cfg, c.model, actions, c.step_dt, command, c.joint_names
    )
    parity.update({f"osl_conformance_{k}": v for k, v in conformance.items()})
    np.savez_compressed(tmp / "parity.npz", **parity)
    manifest["validation"] = {
      "human_onnx_max_abs_error": errors[HUMAN],
      "osl_onnx_max_abs_error": errors[OSL],
      "parity_inputs": (
        "<controller>_observations: the actor's observation mean, then mean + std "
        "* z (N(0, 1)); <controller>_means: their ONNX outputs"
      ),
      "osl_conformance": {
        "keys": sorted(f"osl_conformance_{k}" for k in conformance),
        "steps": int(conformance["qpos"].shape[0]),
        "lags": conformance["lags"].tolist(),
        "clip": int(conformance["clip"]),
        "procedure": (
          "For each step t: mj_resetData; set qpos, qvel, act and ctrl from "
          "osl_conformance_*[t]; mj_forward; the previous OSL command is "
          "osl_conformance_last_action[t] (0 at t = 0, a reset). Build the osl "
          "frame, push it through the latency and history buffers with a reset "
          "at t = 0, and compare with osl_conformance_obs_lag<L>[t] (float32 "
          "terms: about 1e-5 relative). osl_conformance_frames[t] is the "
          "undelayed frame."
        ),
      },
      "physics_transfer_verified": False,
    }
    manifest["files"] = {name: gl.sha256(tmp / name) for name in CHECKED_FILES}
    header = {k: manifest[k] for k in ("schema_version", "kind", "run")}
    for name in CONTROLLERS:
      attach_metadata(
        tmp / ONNX_FILES[name], _plain({**header, "controller": name, **sections[name]})
      )
      manifest["files"][ONNX_FILES[name]] = gl.sha256(tmp / ONNX_FILES[name])
    (tmp / "manifest.json").write_text(json.dumps(_plain(manifest), indent=2) + "\n")
    (tmp / "README.md").write_text(_readme(_plain(manifest)))
    if output.exists():
      shutil.rmtree(output)
    os.rename(tmp, output)
  except BaseException:
    shutil.rmtree(tmp, ignore_errors=True)
    raise
  return output


##
# Command line.
##


@dataclass
class SplitExportConfig:
  checkpoint: tyro.conf.Positional[Path]
  """``model_<N>.pt`` of a stage-2 split run (its run directory holds
  ``params/``)."""
  output: Path | None = None
  """Bundle folder; default ``<run>/deploy/<checkpoint stem>``."""
  library: Path | None = None
  """Clip folder, when the run's ``library_dir`` is not on this machine."""
  human_inference: Literal["auto", "mean", "expected"] = "auto"
  """The human's default inference: ``auto`` picks ``expected`` for a median
  muscle std above 0.3 and ``mean`` otherwise. The OSL runs on its mean."""
  overwrite: bool = False
  """Replace an existing bundle."""
  hf_repo: str | None = None
  """Upload the bundle to this Hugging Face model repo (``<user>/<name>``)."""
  hf_path: str | None = None
  """Folder inside the repo; default ``<run>/<checkpoint stem>``."""
  hf_public: bool = False
  """Create a missing repo public (default private)."""
  threads: int = 2
  """CPU threads for PyTorch."""
  solver_iterations: int | None = None
  """The bundle model's Newton iteration cap; default the run's, at least 10."""
  ls_solver_iterations: int | None = None
  """Its line-search cap; default the run's, at least 20."""


def main() -> None:
  import mjlab

  cfg = tyro.cli(SplitExportConfig, config=mjlab.TYRO_FLAGS)
  torch.set_num_threads(cfg.threads)
  inference = None if cfg.human_inference == "auto" else cfg.human_inference
  bundle = export_bundle(
    cfg.checkpoint,
    cfg.output,
    cfg.library,
    inference,
    cfg.overwrite,
    cfg.solver_iterations,
    cfg.ls_solver_iterations,
  )
  manifest = json.loads((bundle / "manifest.json").read_text())
  controllers = manifest["controllers"]
  print(
    f"[INFO] Bundle: {bundle}\n"
    f"[INFO] {manifest['run']['task']}: human {controllers[HUMAN]['action']['size']} "
    f"outputs ({controllers[HUMAN]['action']['default_inference']}), OSL "
    f"{controllers[OSL]['action']['size']} outputs "
    f"({controllers[OSL]['action']['default_inference']}); ONNX max abs error "
    f"{manifest['validation']['human_onnx_max_abs_error']:.3g} / "
    f"{manifest['validation']['osl_onnx_max_abs_error']:.3g}"
  )
  if cfg.hf_repo:
    try:
      url = gl.upload_bundle(
        bundle, cfg.hf_repo, cfg.hf_path, private=not cfg.hf_public
      )
    except Exception as err:
      print(f"[WARN] Upload failed: {err}", file=sys.stderr)
      sys.exit(1)
    print(f"[INFO] Uploaded: {url}")


if __name__ == "__main__":
  main()
