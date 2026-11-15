"""Export a stairs checkpoint as a deployment bundle, on CPU.

A stairs policy follows one reference clip up the clip's own terrain, at the
terrain's world coordinates. Its bundle is the gait-library bundle
(:mod:`mjlab.tasks.osl_gait_library.export`, schema 1, one ``actor.onnx`` for the
whole body) with the reference, terrain and checks a single clip needs:

* ``actor.onnx``: the raw action mean behind the frozen observation statistics;
* ``manifest.json``: the policy contract (:mod:`mjlab.tasks.osl_mimic.rl.export`)
  with the observation schema ``osl_mimic_v1`` (the gait-library terms without
  ``twist`` and ``phase``), and the sections a single clip adds: ``reference``
  (``kind`` ``single_clip``: how the frame advances, clamps and ends),
  ``terrain`` (the boxes, as compiled), ``initial_state`` and ``failure``;
  provenance and checksums of every file;
* ``model.mjb``: the compiled scene with the terrain boxes (geoms
  ``terrain_box_<i>`` on the body ``terrain``, with the floor plane), the task's
  simulation options at the deployment solver caps, and the OSL foot touch sites
  the run trained with (the manifest's ``touch_sites`` declares them);
* ``terrain.json``: the TERRA terrain file the run trained on (the run's
  ``params/terrain.json``), from which ``model.mjb``'s boxes were built;
* ``reference.npz``: the clip as the training command preprocessed it, (T, ...)
  float32 in the terrain's world frame, never shifted;
* ``parity.npz``: the actor's ONNX outputs on inputs drawn from its statistics,
  and an observation conformance set (:func:`actor_conformance`): CPU MuJoCo
  states at reference frames and the observation mjlab's own terms build there;
* ``README.md``: a summary for people (not checksummed).

The run's ``params/`` decide the model, socket and reward, and its
``params/terrain.json`` the terrain; the clip is the run's ``motion_file`` (or
``--motion``, when the run's path is not on this machine). The task is rebuilt
from code with the OSL foot touch sites the checkpoint was trained with and must
match the run's ``params/env.yaml`` (touch sites, simulation, decimation,
observation terms, actions, actuator delays). No Warp simulation or GPU is
involved::

  uv run --no-sync python -m mjlab.tasks.osl_stairs.export \\
    logs/rsl_rl/<stairs experiment>/<run>/model_<N>.pt \\
    [--motion <clip.npz>] [--hf-repo <user>/<repo>]

The bundle goes to ``<run>/deploy/model_<N>`` unless ``--output`` says otherwise.
Stage-2 stairs runs export with :mod:`mjlab.tasks.osl_split_stairs.export`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import mujoco
import numpy as np
import torch
import tyro

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.entity.data import EntityData
from mjlab.tasks.osl_gait_library import export as gl
from mjlab.tasks.osl_mimic.config.env_cfgs import VALIDATION_SITE_THRESHOLD, Reward
from mjlab.tasks.osl_mimic.rl.export import (
  Inference,
  PolicyParts,
  action_std,
  attach_metadata,
  export_actor_onnx,
  normalizer_samples,
  onnx_parity,
  policy_metadata,
)
from mjlab.tasks.osl_mimic.touch_sites import (
  run_touch_sites,
  set_touch_sites,
  touch_sites_manifest,
  touch_sites_readme_line,
)
from mjlab.tasks.osl_stairs.config.env_cfgs import osl_stairs_env_cfg, sync_terrain
from mjlab.tasks.osl_stairs.mdp import StairsMotionCommand, StairsMotionCommandCfg
from mjlab.tasks.osl_stairs.rl.runner import TERRAIN_COPY
from mjlab.tasks.osl_stairs.terrain import (
  BOX_PREFIX,
  TERRAIN_BODY,
  scene_boxes,
  terrain_sidecar,
)

if TYPE_CHECKING:
  from mjlab.envs.mdp.actions.actions import BaseAction
  from mjlab.managers.observation_manager import ObservationGroupCfg

REFERENCE_KIND = "single_clip"
TERRAIN_FILE = "terrain.json"
"""The terrain's file in the bundle (a copy of the run's ``params/terrain.json``)."""
CHECKED_FILES = ("model.mjb", "reference.npz", TERRAIN_FILE, "actor.onnx", "parity.npz")
REFERENCE_FIELDS = {
  "root_pos": "ref_root_pos",
  "root_quat": "ref_root_quat",
  "root_lin_vel": "ref_root_lin_vel",
  "root_ang_vel_w": "ref_root_ang_vel_w",
  "joint_pos": "ref_joint_pos",
  "joint_vel": "ref_joint_vel",
  "site_pos": "ref_site_pos",
  "rpos": "ref_rpos",
}
"""``reference.npz`` key -> the command's (T, ...) array: the gait-library bundle's
fields, for one clip and without its clip axis."""
REFERENCE_UNITS = {
  "root_pos": "(T, 3) m, world: the root body 'Full Body'",
  "root_quat": "(T, 4) wxyz, world",
  "root_lin_vel": "(T, 3) m/s, world",
  "root_ang_vel_w": "(T, 3) rad/s, world (qvel holds it in the root body frame)",
  "joint_pos": "(T, joints) in joint_names order; joints absent from the clip "
  "(the compliant socket's) are 0",
  "joint_vel": "(T, joints), as joint_pos",
  "site_pos": "(T, sites, 3) m, world: the site bodies' origins",
  "rpos": "(T, sites - 1, 3) m, world: site_pos[:, 1:] - site_pos[:, :1]",
  "num_frames": "() int64: T",
}
COMMAND_SETTINGS = ("site_bodies", "lookahead_steps", "lookahead_stride")
"""Reference settings taken from the run: they shape the observation."""
SOURCE_FOLDERS = (*gl.SOURCE_FOLDERS, "tasks/osl_stairs")
CONFORMANCE_SEED = 0
CONFORMANCE_ACTIVATION = (0.0, 0.3)
"""The conformance states' muscle activations are uniform in this range."""
CONFORMANCE_KEYS = ("frames", "qpos", "qvel", "act", "ctrl", "obs", "means")
STAIRS_ALGORITHM = "mjlab.tasks.osl_mimic.rl.ppo:MuscleMimicPPO"
SPLIT_ALGORITHM = "mjlab.tasks.osl_split.rl.ppo:TwoAgentPPO"
DEPLOYMENT_NOTE = (
  "Solver caps of model.mjb. Training ran mujoco_warp at training_solver, where its "
  "solves converge; CPU MuJoCo needs a higher cap to converge (it truncates the "
  "contact solve at 4 iterations under the elliptic cone at impratio 3)."
)


##
# The run.
##


@dataclass(frozen=True)
class StairsRunSpec:
  """What a stairs stage-1 run trained: from its ``params/``."""

  model: Model
  socket: Socket
  reward: Reward
  experiment: str
  run: str

  @property
  def task_id(self) -> str | None:
    from .config import _VARIANTS

    for task, (model, socket, reward, _) in _VARIANTS.items():
      if (model, socket, reward) == (self.model, self.socket, self.reward):
        return task
    return None


def model_socket_reward(
  checkpoint: Path, agent: Mapping[str, Any]
) -> tuple[Model, Socket, Reward]:
  """The run's model, socket and reward, from ``params/agent.yaml``'s
  ``validation``.

  Raises:
    ValueError: If any is missing or unknown.
  """
  v = agent.get("validation") or {}
  missing = [k for k in ("model", "socket", "reward") if k not in v]
  if missing:
    raise ValueError(
      f"{checkpoint.parent / 'params/agent.yaml'} has no validation.{missing}: not "
      "an OSL run."
    )
  if v["model"] not in ("robot", "msk") or v["socket"] not in ("rigid", "compliant"):
    raise ValueError(f"Unknown model or socket in the run: {v}")
  if v["reward"] not in ("mimic", "clf"):
    raise ValueError(f"Unknown reward in the run: {v['reward']}")
  return v["model"], v["socket"], v["reward"]


def check_stairs_run(checkpoint: Path, saved: Mapping[str, Any]) -> Path:
  """Check that the run played a stairs clip; returns its terrain copy
  (``params/terrain.json``).

  Raises:
    ValueError: If its ``params/env.yaml`` records no terrain (a gait-library or
      flat-ground run) or the run has no ``params/terrain.json``.
  """
  motion = (saved.get("commands") or {}).get("motion") or {}
  if "terrain_file" not in motion:
    trained = (
      f"on the motion library {motion['library_dir']}: export it with python -m "
      "mjlab.tasks.osl_gait_library.export (stage 1) or mjlab.tasks.osl_split."
      "export (stage 2)"
      if "library_dir" in motion
      else "without a terrain"
    )
    raise ValueError(
      f"{checkpoint.parent / 'params/env.yaml'} is not a stairs run: it trained "
      f"{trained}."
    )
  terrain = checkpoint.parent / "params" / TERRAIN_COPY
  if not terrain.is_file():
    raise ValueError(
      f"The run has no params/{TERRAIN_COPY}, the terrain every stairs run copies "
      "as it starts; the bundle's terrain must be the one the policy trained on."
    )
  return terrain


def stairs_run_spec(
  checkpoint: Path, agent: Mapping[str, Any], saved: Mapping[str, Any]
) -> StairsRunSpec:
  """The stage-1 run's model, socket and reward, checked to be a stairs run.

  Raises:
    ValueError: For a stage-2 run, a run that is not on the stairs, or an unknown
      model, socket or reward.
  """
  algorithm = (agent.get("algorithm") or {}).get("class_name")
  if algorithm == SPLIT_ALGORITHM:
    raise ValueError(
      f"{checkpoint} is a stage-2 run: export it with python -m "
      "mjlab.tasks.osl_split_stairs.export."
    )
  if algorithm != STAIRS_ALGORITHM:
    raise ValueError(f"Unsupported algorithm {algorithm!r}: not an OSL stairs run.")
  model, socket, reward = model_socket_reward(checkpoint, agent)
  check_stairs_run(checkpoint, saved)
  return StairsRunSpec(
    model=model,
    socket=socket,
    reward=reward,
    experiment=str(agent.get("experiment_name", checkpoint.parent.parent.name)),
    run=checkpoint.parent.name,
  )


def resolve_motion(
  checkpoint: Path, saved: Mapping[str, Any], motion: str | Path | None
) -> Path:
  """The clip to export: ``motion`` if given, else the run's ``motion_file``.

  A ``motion`` with another file name than the run's draws a warning: clips are
  told apart by name only.

  Raises:
    ValueError: If the file is missing.
  """
  recorded = str(saved["commands"]["motion"].get("motion_file") or "")
  path = Path(motion or recorded).expanduser()
  if not path.is_file():
    raise ValueError(
      f"No motion file at {path}. Pass --motion <the run's clip.npz>."
      if not motion
      else f"No motion file at {path}."
    )
  if recorded and path.name != Path(recorded).name:
    print(
      f"[WARN] The run trained on the clip {Path(recorded).name}; exporting "
      f"{path.name}.",
      flush=True,
    )
  return path.resolve()


def warn_on_refit_sidecar(motion: Path, terrain: Path) -> None:
  """Warn when the clip's terrain sidecar differs from the run's copy: the boxes
  were refit after training. The bundle keeps the run's terrain."""
  sidecar = terrain_sidecar(motion)
  if sidecar.is_file() and gl.sha256(sidecar) != gl.sha256(terrain):
    print(
      f"[WARN] {sidecar} differs from the run's {terrain}: the bundle uses the "
      "terrain the policy trained on (the run's copy).",
      flush=True,
    )


def point_at_clip(
  cfg: Any, saved: Mapping[str, Any], motion: Path, terrain: Path
) -> None:
  """Make ``cfg``'s motion command the run's (its reference settings, ``motion``
  and ``terrain``) and its scene's terrain follow it, in place."""
  command_cfg = cfg.commands["motion"]
  assert isinstance(command_cfg, StairsMotionCommandCfg)
  saved_motion = saved["commands"]["motion"]
  for key in COMMAND_SETTINGS:
    value = saved_motion[key]
    setattr(command_cfg, key, tuple(value) if isinstance(value, list) else value)
  command_cfg.motion_file = str(motion)
  command_cfg.terrain_file = str(terrain)
  sync_terrain(cfg)


def compile_stairs_task(
  cfg: Any,
  solver_iterations: int | None = None,
  ls_solver_iterations: int | None = None,
) -> tuple[SimpleNamespace, StairsMotionCommand, dict[str, BaseAction], dict]:
  """Compile a stairs task config on CPU, its motion command already pointed at
  the clip and terrain (:func:`point_at_clip`): the deployment solver caps, the
  scene with the terrain boxes, the command (which checks the boxes against its
  terrain file) and the actions.

  Returns the CPU env namespace, the command, the action terms in action-manager
  order and the solver caps (``training`` and ``deployment``).
  """
  solver = gl.deployment_solver(cfg, solver_iterations, ls_solver_iterations)
  env = gl.cpu_scene_env(cfg)
  command = cfg.commands["motion"].build(env)
  env.command_manager = SimpleNamespace(get_term=lambda _: command)
  return env, command, gl.build_actions(cfg, env), solver


##
# The reference and the terrain.
##


def reference_arrays(command: StairsMotionCommand) -> dict[str, Any]:
  """``reference.npz``: the command's preprocessed clip, (T, ...) float32."""
  arrays: dict[str, Any] = {
    key: getattr(command, attr).detach().cpu().numpy()
    for key, attr in REFERENCE_FIELDS.items()
  }
  arrays["num_frames"] = np.asarray(command.num_frames, dtype=np.int64)
  return arrays


def reference_section(
  command: StairsMotionCommand, motion: Path, step_dt: float
) -> dict[str, Any]:
  """The manifest's ``reference``: the clip and how a single clip plays."""
  cfg = command.cfg
  frames = int(command.num_frames)
  return {
    "kind": REFERENCE_KIND,
    "clip": motion.stem,
    "clip_file": motion.name,
    "clip_names": [motion.stem],
    "num_frames": frames,
    "policy_dt": step_dt,
    "duration_s": (frames - 1) * step_dt,
    "lookahead_steps": int(cfg.lookahead_steps),
    "lookahead_stride": int(cfg.lookahead_stride),
    "site_bodies": [f"robot/{b}" for b in cfg.site_bodies],
    "start_frame": 0,
    "training_start_frames": "uniform in [0, num_frames - 1]",
    "fields": dict(REFERENCE_UNITS),
    "frame": (
      "At control step k after the start (k = 0: the start's own observation) the "
      "observation reads frame f = min(start_frame + k, num_frames - 1): the "
      "frame saturates at the last one."
    ),
    "lookahead": (
      "The goal's lookahead frames are min(f + i * lookahead_stride, num_frames "
      "- 1) for i = 1 .. lookahead_steps - 1: clamped at the last frame, never "
      "wrapped."
    ),
    "phase": "The goal's last value is f / num_frames.",
    "positions": (
      "World coordinates of the terrain, unshifted: no re-centring at the start, "
      "no rebase, no clip switching, no twist command. The goal itself reads "
      "only relative positions and differences; the start state and the failure "
      "check read world positions."
    ),
    "episode_end": (
      "Training ends an episode as a time-out after the step whose next frame is "
      "the last (min(f + 1, num_frames - 1) >= num_frames - 1): from start_frame "
      "0, after num_frames - 1 control steps, the last observation acted on "
      "reading frame num_frames - 2. Training starts at the last frame are the "
      "only place the policy saw frame num_frames - 1."
    ),
  }


def terrain_section(
  command: StairsMotionCommand,
  model: mujoco.MjModel,
  source: Path,
) -> dict[str, Any]:
  """The manifest's ``terrain``: the terrain file and every box as compiled.

  Raises:
    ValueError: If the compiled boxes are not the file's (the command checks the
      same at construction).
  """
  ids = scene_boxes(model)
  if len(ids) != len(command.terrain):
    raise ValueError("The compiled scene's boxes differ from the terrain file.")
  boxes = []
  for i, (gid, box) in enumerate(zip(ids, command.terrain, strict=True)):
    boxes.append(
      {
        "geom": model.geom(gid).name,
        "name": box.name,
        "pos": list(box.pos),
        "size": list(box.size),
        "yaw": box.yaw,
        "pitch": box.pitch,
        "quat": box.quat.tolist(),
        "compiled": {
          "pos": model.geom_pos[gid].tolist(),
          "quat": model.geom_quat[gid].tolist(),
          "size": model.geom_size[gid].tolist(),
        },
      }
    )
    if model.geom(gid).name != f"{BOX_PREFIX}{i}":
      raise ValueError(f"Terrain box {i} compiled as {model.geom(gid).name}.")
  body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TERRAIN_BODY)
  return {
    "file": TERRAIN_FILE,
    "format": "TERRA TerrainSpec JSON (boxes: world pos, half-extents size, yaw, "
    "pitch; orientation Rz(yaw) @ Ry(pitch))",
    "sha256": gl.sha256(source),
    "source": str(source),
    "box_count": len(boxes),
    "in_model": True,
    "body": TERRAIN_BODY,
    "body_id": int(body),
    "geom_prefix": BOX_PREFIX,
    "boxes": boxes,
    "floor": "mjlab's terrain plane at z = 0 on the same body",
    "contact": "the floor plane's parameters (MuJoCo defaults, friction 1.0)",
    "check": (
      "model.mjb holds exactly box_count geoms terrain_box_<i> (i in file order) "
      "on body terrain at the world origin, each with the file's pos and size "
      "(atol 1e-6) and quat (|<q_model, q_file>| > 1 - 1e-9): mjlab's "
      "check_scene_terrain."
    ),
  }


def initial_state_section() -> dict[str, Any]:
  return {
    "frame": "reference.start_frame",
    "qpos": (
      "root: root_pos[f0], root_quat[f0] (world, unshifted); non-root joints: "
      "joint_pos[f0]"
    ),
    "qvel": (
      "root: root_lin_vel[f0] (world) and root_ang_vel_w[f0] rotated into the "
      "root body frame; non-root joints: joint_vel[f0]"
    ),
    "act": "zero (MuJoCo's reset)",
    "servo_targets": (
      "position servo targets (robot model) = joint_pos[f0], also in a reset "
      "actuator delay history, as the gait-library bundles"
    ),
  }


def failure_section() -> dict[str, Any]:
  return {
    "site_deviation": {
      "threshold_m": VALIDATION_SITE_THRESHOLD,
      "rule": (
        "After the step from frame f: the mean over the site bodies 1.. of "
        "||(p_i - p_0) - rpos[min(f + 1, num_frames - 1)]_i|| exceeds "
        "threshold_m (p: the site bodies' world positions); mjlab's validation "
        "failure"
      ),
    }
  }


##
# Observation conformance on CPU MuJoCo.
##


class _LiveData:
  """CPU MjData fields as mujoco_warp's single-precision (1, ...) tensors, read
  afresh at each access."""

  def __init__(self, data: mujoco.MjData) -> None:
    self._data = data

  def __getattr__(self, name: str) -> torch.Tensor:
    if name.startswith("_"):
      raise AttributeError(name)
    value = np.asarray(getattr(self._data, name))
    return torch.as_tensor(value, dtype=torch.float32).reshape(1, *value.shape)


class _Sensor:
  """A scene sensor of the CPU data: ``data`` is its reading, (1, dim) float32."""

  def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str) -> None:
    sensor = model.sensor(name)
    self._data = data
    self._slice = slice(int(sensor.adr[0]), int(sensor.adr[0] + sensor.dim[0]))

  @property
  def data(self) -> torch.Tensor:
    values = self._data.sensordata[self._slice]
    return torch.as_tensor(values, dtype=torch.float32).reshape(1, -1)


def cpu_entity(entity: Any, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
  """A stand-in for the scene entity over CPU MuJoCo: its indexing, and mjlab's
  own :class:`~mjlab.entity.data.EntityData` reading ``data`` (only the state
  reads; targets, limits and defaults are empty)."""
  indexing = entity._compute_indexing(model, "cpu")
  empty = torch.zeros(1, 0)
  entity_data = EntityData(
    indexing=indexing,
    data=_LiveData(data),  # type: ignore[arg-type]
    model=None,  # type: ignore[arg-type]
    device="cpu",
    default_root_state=empty,
    default_joint_pos=empty,
    default_joint_vel=empty,
    default_joint_pos_limits=empty,
    joint_pos_limits=empty,
    soft_joint_pos_limits=empty,
    gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]),
    forward_vec_b=torch.tensor([[1.0, 0.0, 0.0]]),
    is_fixed_base=bool(entity.is_fixed_base),
    is_articulated=bool(entity.is_articulated),
    is_actuated=bool(entity.is_actuated),
    joint_pos_target=empty,
    joint_vel_target=empty,
    joint_effort_target=empty,
    tendon_len_target=empty,
    tendon_vel_target=empty,
    tendon_effort_target=empty,
    site_effort_target=empty,
    encoder_bias=empty,
  )
  return SimpleNamespace(
    data=entity_data,
    indexing=indexing,
    joint_names=list(entity.joint_names),
    body_names=list(entity.body_names),
  )


class CpuActorEnv:
  """What mjlab's full-body ``actor`` terms read from an env, over CPU MuJoCo.

  One env: ``scene["robot"]`` an entity stand-in whose
  :class:`~mjlab.entity.data.EntityData` reads :attr:`data`, other scene names
  its sensors, ``sim`` the model and data, ``command_manager`` the motion command
  (its robot swapped for the stand-in while the env is in use). With a real
  :class:`~mjlab.managers.observation_manager.ObservationManager` on it, the
  training code itself builds the observation from CPU state.
  """

  num_envs = 1
  device = "cpu"

  def __init__(
    self, cfg: Any, model: mujoco.MjModel, entity: Any, command: Any, step_dt: float
  ) -> None:
    self.cfg = cfg
    self.model = model
    self.data = mujoco.MjData(model)
    self.robot = cpu_entity(entity, model, self.data)
    self.sim = SimpleNamespace(mj_model=model, data=_LiveData(self.data))
    self.command = command
    self.command_manager = SimpleNamespace(get_term=lambda _: command)
    self.step_dt = float(step_dt)
    self.physics_dt = float(cfg.sim.mujoco.timestep)
    self.observation_manager: Any = None

  @property
  def scene(self) -> Mapping[str, Any]:
    env = self

    class _Scene:
      def __getitem__(self, name: str) -> Any:
        if name == "robot":
          return env.robot
        return _Sensor(env.model, env.data, name)

    return _Scene()  # type: ignore[return-value]

  def set_state(
    self, qpos: np.ndarray, qvel: np.ndarray, act: np.ndarray, ctrl: np.ndarray
  ) -> None:
    """Reset the data, set the state and run ``mj_forward`` (sensors included)."""
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:], self.data.qvel[:] = qpos, qvel
    self.data.act[:], self.data.ctrl[:] = act, ctrl
    mujoco.mj_forward(self.model, self.data)


def conformance_frames(num_frames: int, lookahead_steps: int, stride: int) -> list[int]:
  """The reference frames the conformance set observes: the start, three across
  the clip, and the last lookahead span densely enough that each lookahead
  entry is seen just before and after it clamps, up to the last frame."""
  last = num_frames - 1
  span = (lookahead_steps - 1) * stride
  candidates = [0, 1, 2, num_frames // 4, num_frames // 2, 3 * num_frames // 4]
  for i in range(1, lookahead_steps):
    edge = last - i * stride
    candidates += [edge, edge + 1]
  candidates += [last - span - 1, last - 2, last - 1, last]
  return sorted({f for f in candidates if 0 <= f <= last})


def reference_qpos_qvel(
  model: mujoco.MjModel,
  command: Any,
  joint_names: Sequence[str],
  frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
  """The reference state at each of ``frames`` as (qpos, qvel): the root's world
  pose (unshifted), its world linear and body angular velocity, and the joints."""
  root = model.joint("robot/root")
  if model.jnt_type[root.id] != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError("robot/root is not a free joint.")
  q0, v0 = int(root.qposadr[0]), int(root.dofadr[0])
  qpos_ids = [int(model.joint(f"robot/{j}").qposadr[0]) for j in joint_names]
  dof_ids = [int(model.joint(f"robot/{j}").dofadr[0]) for j in joint_names]
  qpos = np.tile(model.qpos0, (len(frames), 1))
  qvel = np.zeros((len(frames), model.nv))

  def at(name: str, f: int) -> np.ndarray:
    return getattr(command, name)[f].detach().cpu().double().numpy()

  for row, f in enumerate(frames):
    quat = at("ref_root_quat", f)
    qpos[row, q0 : q0 + 3] = at("ref_root_pos", f)
    qpos[row, q0 + 3 : q0 + 7] = quat
    qpos[row, qpos_ids] = at("ref_joint_pos", f)
    qvel[row, v0 : v0 + 3] = at("ref_root_lin_vel", f)
    inverse = np.empty(4)
    mujoco.mju_negQuat(inverse, quat)
    mujoco.mju_rotVecQuat(
      qvel[row, v0 + 3 : v0 + 6], at("ref_root_ang_vel_w", f), inverse
    )
    qvel[row, dof_ids] = at("ref_joint_vel", f)
  return qpos, qvel


def validation_group(group: ObservationGroupCfg) -> ObservationGroupCfg:
  """A copy of an observation group with corruption off, as validation reads it."""
  group = deepcopy(group)
  group.enable_corruption = False
  return group


def actor_conformance(
  cfg: Any,
  model: mujoco.MjModel,
  command: StairsMotionCommand,
  group: ObservationGroupCfg,
  step_dt: float,
  seed: int = CONFORMANCE_SEED,
) -> dict[str, np.ndarray]:
  """The full-body observation mjlab builds at a set of reference frames.

  At each frame of :func:`conformance_frames`: the data is reset, the reference
  state set (:func:`reference_qpos_qvel`) with random muscle activations
  (uniform in ``CONFORMANCE_ACTIVATION``) and random controls (uniform in each
  actuator's ctrlrange), ``mj_forward`` run, the command put at that frame, and
  mjlab's own observation manager computes ``group`` (the full-body actor
  group, without corruption) over CPU MuJoCo with the training terms and
  :class:`~mjlab.entity.data.EntityData`. A consumer reproduces each ``obs`` row
  from its state and frame alone: the whole observation contract, including the
  reference's frame, lookahead clamp and phase.

  Returns:
    ``frames`` (S,), ``qpos`` (S, nq), ``qvel`` (S, nv), ``act`` (S, na),
    ``ctrl`` (S, nu) and ``obs`` (S, D).
  """
  from mjlab.managers.observation_manager import ObservationManager

  frames = conformance_frames(
    int(command.num_frames),
    int(command.cfg.lookahead_steps),
    int(command.cfg.lookahead_stride),
  )
  entity = command.robot
  env = CpuActorEnv(cfg, model, entity, command, step_dt)
  qpos, qvel = reference_qpos_qvel(model, command, env.robot.joint_names, frames)
  rng = np.random.default_rng(seed)
  act = rng.uniform(*CONFORMANCE_ACTIVATION, (len(frames), model.na))
  low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
  ctrl = rng.uniform(low, high, (len(frames), model.nu))
  saved_frame = command.frame.clone()
  command.robot = env.robot
  try:
    manager = ObservationManager({"actor": validation_group(group)}, env)
    rows = []
    for row, f in enumerate(frames):
      env.set_state(qpos[row], qvel[row], act[row], ctrl[row])
      command.frame[:] = f
      # As the env computes it each step: without update_history the manager
      # returns its cached observation.
      obs = cast(torch.Tensor, manager.compute(update_history=True)["actor"])
      rows.append(obs[0].double().numpy())
  finally:
    command.robot = entity
    command.frame[:] = saved_frame
  if len({r.tobytes() for r in rows}) != len(rows):
    raise RuntimeError("Two conformance observations are identical.")
  return {
    "frames": np.asarray(frames, dtype=np.int64),
    "qpos": qpos,
    "qvel": qvel,
    "act": act,
    "ctrl": ctrl,
    "obs": np.stack(rows),
  }


def conformance_procedure(prefix: str) -> str:
  return (
    f"For each row t: mj_resetData; set qpos, qvel, act and ctrl from "
    f"{prefix}_*[t]; mj_forward; the reference at frame {prefix}_frames[t] "
    f"(reference.frame/lookahead/phase). Build the observation and compare with "
    f"{prefix}_obs[t] (float32 terms: about 1e-5 relative); {prefix}_means[t] is "
    "the ONNX output on it."
  )


##
# The bundle.
##


def stairs_provenance(
  checkpoint: Path,
  iteration: int | None,
  motion: Path,
  terrain: Path,
  source_folders: tuple[str, ...] = SOURCE_FOLDERS,
) -> dict[str, Any]:
  """Where the bundle comes from: the checkpoint and its configs, the clip and
  terrain files, the training sources and the mjlab commit."""
  import mjlab

  root = Path(mjlab.__file__).resolve().parent
  status = gl._git(root, "status", "--porcelain", "--", *source_folders)
  return {
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": gl.sha256(checkpoint),
    "iteration": iteration,
    "agent_config_sha256": gl.sha256(checkpoint.parent / "params/agent.yaml"),
    "env_config_sha256": gl.sha256(checkpoint.parent / "params/env.yaml"),
    "motion_file": str(motion),
    "motion_sha256": gl.sha256(motion),
    "terrain_file": str(terrain),
    "terrain_sha256": gl.sha256(terrain),
    "training_sources": {
      str(p.relative_to(root)): gl.sha256(p)
      for folder in source_folders
      for p in sorted((root / folder).rglob("*"))
      if p.is_file() and p.suffix in (".py", ".xml")
    },
    "mjlab_head": gl._git(root, "rev-parse", "HEAD"),
    "mjlab_sources_modified": bool(status) if status is not None else None,
    "note": (
      "Scene and reference rebuilt from this checkout, the clip and the run's "
      "terrain copy; the checkpoint records no model or clip hashes, so their "
      "identity at training time is checked by config (and, for stage 2, by the "
      "clip's name and the terrain's digest) only."
    ),
  }


def _readme(manifest: dict[str, Any]) -> str:
  run, action = manifest["run"], manifest["action"]
  reference, terrain = manifest["reference"], manifest["terrain"]
  delays = manifest["training"]["actuator_delay"]
  delay_lines = "\n".join(
    f"- `{name}`: {d['seconds'][0] * 1e3:g}-{d['seconds'][1] * 1e3:g} ms"
    if d
    else f"- `{name}`: unknown"
    for name, d in delays.items()
  )
  solver = manifest["deployment"]["solver"]
  trained = manifest["deployment"]["training_solver"]
  return f"""# {run["task"] or "OSL stairs policy"}: {run["run"]} / {run["checkpoint"]}

Stairs deployment bundle exported by mjlab (`mjlab.tasks.osl_stairs.export`) for
osl_deploy (plain CPU MuJoCo {manifest["mujoco_version"]}).

- Model: {run["model"]}, socket: {run["socket"]}, reward: {run["reward"]}
- Clip: `{reference["clip_file"]}`, {reference["num_frames"]} frames
  ({reference["duration_s"]:g} s), reference kind `{reference["kind"]}` (one clip,
  started at frame {reference["start_frame"]}, never re-centred, lookahead clamped
  at the last frame)
- Terrain: {terrain["box_count"]} boxes in `model.mjb` (body `{terrain["body"]}`,
  geoms `{terrain["geom_prefix"]}<i>`), from `{terrain["file"]}`
{touch_sites_readme_line(manifest["touch_sites"])}
- Checkpoint: `{run["checkpoint"]}` (iteration {run["iteration"]})
- Control: {1.0 / manifest["policy_dt"]:g} Hz, {manifest["decimation"]} physics steps
  per action
- Observation: {manifest["observation"]["size"]} values, schema
  `{manifest["observation"]["schema"]}`, terms
  {", ".join(manifest["observation"]["terms"])}
- Action: {action["size"]} outputs, blocks
  {", ".join(f"{b['name']} ({b['size']})" for b in action["blocks"])}
- Default inference: **{action["default_inference"]}**
- ONNX vs PyTorch max abs error: {manifest["validation"]["onnx_max_abs_error"]:.3g}
- Solver caps of `model.mjb`: {solver["iterations"]} Newton, {solver["ls_iterations"]}
  line search (training: {trained["iterations"]}/{trained["ls_iterations"]} in
  mujoco_warp)
- Failure (validation's): mean relative site deviation above
  {manifest["failure"]["site_deviation"]["threshold_m"]:g} m

Actuator command delays in training (deployment must add them itself):
{delay_lines}

`parity.npz` holds an observation conformance set (`actor_conformance_*`): CPU
MuJoCo states at reference frames and the observation mjlab's own terms build
there (manifest `validation.actor_conformance`).
"""


def export_bundle(
  checkpoint: str | Path,
  output: str | Path | None = None,
  motion: str | Path | None = None,
  inference: Inference | None = None,
  overwrite: bool = False,
  solver_iterations: int | None = None,
  ls_solver_iterations: int | None = None,
) -> Path:
  """Export a stairs stage-1 ``checkpoint`` as a bundle (module docstring).

  Args:
    checkpoint: ``model_<N>.pt`` of a stairs run.
    output: The bundle folder (default ``<run>/deploy/<checkpoint stem>``).
    motion: The clip, when the run's ``motion_file`` is not on this machine.
    inference: The manifest's default inference; None picks ``expected`` for MSK
      policies whose median muscle std exceeds 0.3 and ``mean`` for the others.
    overwrite: Replace an existing bundle (once the new one is complete).
    solver_iterations, ls_solver_iterations: The bundle model's solver caps;
      default the run's, at least 10 and 20.
  """
  checkpoint = Path(checkpoint).resolve()
  output = Path(output or checkpoint.parent / "deploy" / checkpoint.stem).resolve()
  if output.exists() and not overwrite:
    raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
  params = checkpoint.parent / "params"
  agent, saved = (
    gl.read_config(params / "agent.yaml"),
    gl.read_config(params / "env.yaml"),
  )
  spec = stairs_run_spec(checkpoint, agent, saved)
  terrain = check_stairs_run(checkpoint, saved)
  clip = resolve_motion(checkpoint, saved, motion)
  warn_on_refit_sidecar(clip, terrain)

  cfg = osl_stairs_env_cfg(spec.model, spec.socket, spec.reward, "train")
  touch = run_touch_sites(checkpoint, saved_env=saved)
  set_touch_sites(cfg, touch.value)
  print(f"[INFO] OSL touch sites: {touch.value} (from {touch.source}).")
  point_at_clip(cfg, saved, clip, terrain)
  gl.check_actor_config(cfg, saved)
  env, command, actions, solver = compile_stairs_task(
    cfg, solver_iterations, ls_solver_iterations
  )
  model = env.sim.mj_model
  actor_group = cfg.observations["actor"]
  parts = PolicyParts(
    model=model,
    joint_names=list(env.scene["robot"].joint_names),
    obs_terms=list(actor_group.terms),
    obs_term_cfgs=dict(actor_group.terms),
    actions=actions,
    site_bodies=list(command.cfg.site_bodies),
    lookahead_steps=int(command.cfg.lookahead_steps),
    decimation=int(cfg.decimation),
    step_dt=float(env.step_dt),
    entity_cfg=cfg.scene.entities["robot"],
    obs_group_history=actor_group.history_length,
  )
  action_dim = sum(term.action_dim for term in actions.values())
  actor = gl.load_actor(checkpoint, action_dim)
  std = action_std(actor, action_dim)
  saved_iter = torch.load(checkpoint, map_location="meta", weights_only=False).get(
    "iter"
  )
  iteration = int(saved_iter) if saved_iter is not None else None

  output.parent.mkdir(parents=True, exist_ok=True)
  tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
  umask = os.umask(0)
  os.umask(umask)
  tmp.chmod(0o777 & ~umask)  # mkdtemp's 0700 would hide the bundle from others.
  try:
    obs_dim = export_actor_onnx(actor, tmp / "actor.onnx")
    manifest: dict[str, Any] = policy_metadata(parts, std, obs_dim, inference)
    manifest["socket"] = spec.socket
    manifest["touch_sites"] = touch_sites_manifest(model)
    if manifest["touch_sites"]["version"] != touch.value:
      raise RuntimeError(
        f"model.mjb carries the {manifest['touch_sites']['version']!r} touch sites, "
        f"but the run trained with {touch.value!r}."
      )
    manifest["run"] = {
      "task": spec.task_id,
      "stage": 1,
      "model": spec.model,
      "socket": spec.socket,
      "reward": spec.reward,
      "variant": "stairs",
      "experiment": spec.experiment,
      "run": spec.run,
      "checkpoint": checkpoint.name,
      "iteration": iteration,
    }
    manifest["reference"] = reference_section(command, clip, parts.step_dt)
    manifest["terrain"] = terrain_section(command, model, terrain)
    manifest["initial_state"] = initial_state_section()
    manifest["failure"] = failure_section()
    manifest["deployment"] = {
      "solver": solver["deployment"],
      "training_solver": solver["training"],
      "note": DEPLOYMENT_NOTE,
    }
    manifest["provenance"] = stairs_provenance(checkpoint, iteration, clip, terrain)

    mujoco.mj_saveModel(model, str(tmp / "model.mjb"))
    shutil.copyfile(terrain, tmp / TERRAIN_FILE)
    np.savez_compressed(tmp / "reference.npz", **reference_arrays(command))
    observations = normalizer_samples(actor, gl.PARITY_SAMPLES)
    means, error = onnx_parity(tmp / "actor.onnx", actor, observations)
    rtol, atol = gl.PARITY_TOLERANCE
    with torch.inference_mode():
      reference_means = actor.as_onnx(verbose=False)(torch.from_numpy(observations))  # type: ignore[operator]
    np.testing.assert_allclose(means, reference_means.numpy(), rtol=rtol, atol=atol)
    conformance = actor_conformance(cfg, model, command, actor_group, parts.step_dt)
    if conformance["obs"].shape[1] != obs_dim:
      raise RuntimeError(
        f"The conformance observation has {conformance['obs'].shape[1]} values; "
        f"the actor reads {obs_dim}."
      )
    conformance["means"], conformance_error = onnx_parity(
      tmp / "actor.onnx", actor, conformance["obs"].astype(np.float32)
    )
    parity: dict[str, Any] = {
      "observation": observations[0],
      "mean": means[0],
      "observations": observations,
      "means": means,
      **{f"actor_conformance_{k}": conformance[k] for k in CONFORMANCE_KEYS},
    }
    np.savez_compressed(tmp / "parity.npz", **parity)
    manifest["validation"] = {
      "onnx_max_abs_error": max(error, conformance_error),
      "parity_inputs": "the actor's observation mean, then mean + std * z (N(0, 1))",
      "actor_conformance": {
        "keys": [f"actor_conformance_{k}" for k in CONFORMANCE_KEYS],
        "frames": conformance["frames"].tolist(),
        "seed": CONFORMANCE_SEED,
        "procedure": conformance_procedure("actor_conformance"),
      },
      "physics_transfer_verified": False,
    }
    manifest["files"] = {name: gl.sha256(tmp / name) for name in CHECKED_FILES}
    attach_metadata(
      tmp / "actor.onnx", {k: v for k, v in manifest.items() if k != "files"}
    )
    manifest["files"]["actor.onnx"] = gl.sha256(tmp / "actor.onnx")
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (tmp / "README.md").write_text(_readme(manifest))
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
class StairsExportConfig:
  checkpoint: tyro.conf.Positional[Path]
  """``model_<N>.pt`` of a stairs stage-1 run (its run directory holds
  ``params/``)."""
  output: Path | None = None
  """Bundle folder; default ``<run>/deploy/<checkpoint stem>``."""
  motion: Path | None = None
  """The clip, when the run's ``motion_file`` is not on this machine."""
  inference: Literal["auto", "mean", "expected"] = "auto"
  """Default inference written to the manifest: ``auto`` picks ``expected`` for MSK
  policies whose median muscle std exceeds 0.3 and ``mean`` for the others."""
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


def upload(bundle: Path, repo: str, path: str | None, public: bool) -> None:
  """Upload a bundle (:func:`~mjlab.tasks.osl_gait_library.export.upload_bundle`);
  exits with status 1 on failure."""
  try:
    url = gl.upload_bundle(bundle, repo, path, private=not public)
  except Exception as err:
    print(f"[WARN] Upload failed: {err}", file=sys.stderr)
    sys.exit(1)
  print(f"[INFO] Uploaded: {url}")


def main() -> None:
  import mjlab

  cfg = tyro.cli(StairsExportConfig, config=mjlab.TYRO_FLAGS)
  torch.set_num_threads(cfg.threads)
  bundle = export_bundle(
    cfg.checkpoint,
    cfg.output,
    cfg.motion,
    None if cfg.inference == "auto" else cfg.inference,
    cfg.overwrite,
    cfg.solver_iterations,
    cfg.ls_solver_iterations,
  )
  manifest = json.loads((bundle / "manifest.json").read_text())
  print(
    f"[INFO] Bundle: {bundle}\n"
    f"[INFO] {manifest['run']['task']} on {manifest['reference']['clip_file']} "
    f"({manifest['terrain']['box_count']} boxes), default inference "
    f"{manifest['action']['default_inference']}, ONNX max abs error "
    f"{manifest['validation']['onnx_max_abs_error']:.3g}"
  )
  if cfg.hf_repo:
    upload(bundle, cfg.hf_repo, cfg.hf_path, cfg.hf_public)


if __name__ == "__main__":
  main()
