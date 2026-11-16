"""Export a stairs stage-2 checkpoint as a two-controller deployment bundle, on CPU.

The unicycle stage-2 bundle (:mod:`mjlab.tasks.osl_split.export`: schema 2, kind
``osl_split_dual``, ``human_actor.onnx`` + ``osl_actor.onnx`` over one model)
with the clip and terrain of the stairs stage-1 bundle
(:mod:`mjlab.tasks.osl_stairs.export`):

* the human controller reads the stairs observation, schema ``osl_mimic_v1``
  (the gait-library terms without ``twist`` and ``phase``); the OSL controller's
  section is unicycle stage 2's (``osl_split_sensors_v1``: sensors, tare,
  saturation, history, latency, previous command, reset);
* ``reference``, ``terrain``, ``initial_state`` and ``failure`` are the stairs
  stage-1 bundle's sections (``reference.kind`` ``single_clip``), the reference
  used by the human alone; ``terrain.json`` and ``reference.npz`` are its files;
* ``model.mjb`` is the stage-2 scene (with the OSL's sensors) and the terrain
  boxes, at the deployment solver caps, with the run's OSL foot touch sites;
* ``parity.npz`` holds each actor's ONNX outputs on inputs drawn from its own
  statistics, the human's observation conformance set at reference frames
  (``human_conformance_*``, as stage 1's ``actor_conformance_*``) and the OSL
  observation conformance sequence over consecutive reference frames of the climb
  (``osl_conformance_*``, unicycle stage 2's procedure).

The run's ``params/`` decide the model, socket and reward, its
``params/terrain.json`` the terrain (the checkpoint's recorded terrain digest must
match it) and its ``motion_file`` the clip (or ``--motion``; the checkpoint's
recorded clip name must match). The task is rebuilt from this checkout with the
OSL foot touch sites the checkpoint was trained with, which its warm start's
stage 1 must share, and its ``osl`` group from the run's record::

  uv run --no-sync python -m mjlab.tasks.osl_split_stairs.export \\
    logs/rsl_rl/<stairs split experiment>/<run>/model_<N>.pt \\
    [--motion <clip.npz>] [--hf-repo <user>/<repo>]

The bundle goes to ``<run>/deploy/model_<N>`` unless ``--output`` says otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import mujoco
import numpy as np
import torch
import tyro

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket, TouchSites
from mjlab.tasks.osl_gait_library import export as gl
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.rl.export import (
  Inference,
  attach_metadata,
  export_actor_onnx,
  normalizer_samples,
  onnx_parity,
)
from mjlab.tasks.osl_mimic.touch_sites import (
  run_touch_sites,
  set_touch_sites,
  touch_sites_manifest,
  touch_sites_readme_line,
)
from mjlab.tasks.osl_split import export as split
from mjlab.tasks.osl_split.config.env_cfgs import sync_osl_group
from mjlab.tasks.osl_split.mdp.observations import (
  describe_osl_sensors,
  osl_group_from_record,
  osl_sensor_set,
)
from mjlab.tasks.osl_split.rl.layout import ActionLayout
from mjlab.tasks.osl_split.rl.osl_observation import run_osl_observation
from mjlab.tasks.osl_split.rl.ppo import OSL_OBS_GROUP
from mjlab.tasks.osl_split.rl.warmstart import lineage_touch_sites
from mjlab.tasks.osl_stairs import export as stairs
from mjlab.tasks.osl_stairs.mdp import StairsMotionCommand

from .config import stairs_split_task_id
from .config.env_cfgs import osl_split_stairs_env_cfg
from .config.rl_cfg import VARIANT
from .rl.runner import STAIRS_INFO_KEY

HUMAN, OSL = split.HUMAN, split.OSL
CONTROLLERS = split.CONTROLLERS
CHECKED_FILES = (
  "model.mjb",
  "reference.npz",
  stairs.TERRAIN_FILE,
  split.ONNX_FILES[HUMAN],
  split.ONNX_FILES[OSL],
  "parity.npz",
)
SOURCE_FOLDERS = (*stairs.SOURCE_FOLDERS, "tasks/osl_split", "tasks/osl_split_stairs")
CONFORMANCE_STEPS = split.CONFORMANCE_STEPS
"""Consecutive reference frames of the OSL conformance sequence."""


##
# The run.
##


@dataclass(frozen=True)
class StairsSplitRunSpec:
  """What a stairs stage-2 run trained: from its ``params/``."""

  model: Model
  socket: Socket
  reward: Reward
  experiment: str
  run: str

  @property
  def task_id(self) -> str:
    return stairs_split_task_id(self.model, self.socket, self.reward)


def stairs_split_run_spec(
  checkpoint: Path, agent: Mapping[str, Any], saved: Mapping[str, Any]
) -> StairsSplitRunSpec:
  """The run's model, socket and reward, checked to be a stairs stage-2 run whose
  actors this bundle schema covers.

  Raises:
    ValueError: For anything else.
  """
  algorithm = (agent.get("algorithm") or {}).get("class_name")
  if algorithm != split.ALGORITHM_CLASS:
    raise ValueError(
      f"{checkpoint.parent / 'params/agent.yaml'} is not a stage-2 split run "
      f"(algorithm {algorithm!r}). Export a stairs stage-1 run with python -m "
      "mjlab.tasks.osl_stairs.export."
    )
  model, socket, reward = stairs.model_socket_reward(checkpoint, agent)
  variant = (agent.get("validation") or {}).get("variant")
  if variant != VARIANT:
    raise ValueError(
      f"The run's validation.variant is {variant!r}, not {VARIANT!r}: export a "
      "unicycle stage-2 run with python -m mjlab.tasks.osl_split.export."
    )
  stairs.check_stairs_run(checkpoint, saved)
  if agent.get("clip_actions") is not None:
    raise ValueError("Wrapper action clipping is outside the bundle schema.")
  groups = agent.get("obs_groups") or {}
  for name in CONTROLLERS:
    key = split.MODEL_KEYS[name]
    if list(groups.get(key) or ()) != [split.OBS_GROUPS[name]]:
      raise ValueError(
        f"The {name} actor must read the {split.OBS_GROUPS[name]!r} observation "
        f"group alone (obs_groups.{key} = {groups.get(key)})."
      )
  return StairsSplitRunSpec(
    model=model,
    socket=socket,
    reward=reward,
    experiment=str(agent.get("experiment_name", checkpoint.parent.parent.name)),
    run=checkpoint.parent.name,
  )


def check_recorded_clip(
  payload: Mapping[str, Any], motion: Path, terrain: Path
) -> dict[str, Any]:
  """The clip and terrain the checkpoint records (``infos["stairs"]``), checked
  against the files the bundle is built from; returns the record (empty when the
  checkpoint has none).

  Raises:
    ValueError: If the checkpoint names another clip or another terrain digest.
  """
  recorded = dict((payload.get("infos") or {}).get(STAIRS_INFO_KEY) or {})
  clip = recorded.get("clip")
  if clip is not None and clip != motion.name:
    raise ValueError(
      f"The checkpoint trained on the clip {clip}, not {motion.name}. Pass "
      "--motion <that clip>."
    )
  digest = recorded.get("terrain_sha256")
  if digest is not None and digest != gl.sha256(terrain):
    raise ValueError(
      f"The checkpoint trained on another terrain (SHA-256 {str(digest)[:12]}) "
      f"than the run's {terrain} ({gl.sha256(terrain)[:12]})."
    )
  if not recorded:
    print(
      "[WARN] The checkpoint records no clip or terrain; the bundle uses the run's.",
      flush=True,
    )
  return recorded


def stairs_split_config(
  spec: StairsSplitRunSpec,
  saved: Mapping[str, Any],
  *,
  touch_sites: TouchSites,
  osl_record: Mapping[str, Any],
  motion: Path,
  terrain: Path,
) -> Any:
  """The run's stairs stage-2 env config, rebuilt from this checkout in train
  mode: the OSL foot ``touch_sites`` the run trained with, the clip and terrain,
  and the ``osl`` group ``osl_record`` describes.

  The touch sites, the human's observation, the actions, the simulation and the
  actuator delays must match the run's (:func:`~mjlab.tasks.osl_gait_library.
  export.check_actor_config`), and its robot must carry the OSL sensors.

  Raises:
    ValueError: On drift, an ``osl`` group outside the schema, or one
      :func:`~mjlab.tasks.osl_split.config.env_cfgs.sync_osl_group` refuses.
  """
  cfg = osl_split_stairs_env_cfg(spec.model, spec.socket, spec.reward, "train")
  set_touch_sites(cfg, touch_sites)
  stairs.point_at_clip(cfg, saved, motion, terrain)
  gl.check_actor_config(cfg, dict(saved))
  spec_fn = str(saved["scene"]["entities"]["robot"].get("spec_fn"))
  if spec_fn != split.OSL_SPEC_FN:
    raise ValueError(
      f"The run's robot spec_fn is {spec_fn!r}, not {split.OSL_SPEC_FN!r}."
    )
  cfg.observations[OSL_OBS_GROUP] = osl_group_from_record(osl_record, train=True)
  sync_osl_group(cfg)
  return cfg


##
# Conformance.
##


def climb_window(command: StairsMotionCommand, steps: int) -> int:
  """The first of the ``steps`` consecutive frames over which the reference root
  moves the most: the climb, so the OSL's IMU turns and its foot loads and
  unloads (the clip starts and ends standing).

  Raises:
    ValueError: If the clip has fewer frames.
  """
  speed = command.ref_root_lin_vel.norm(dim=-1).detach().cpu().double().numpy()
  if speed.shape[0] < steps:
    raise ValueError(f"The clip has {speed.shape[0]} frames, fewer than {steps}.")
  return int(np.argmax(np.convolve(speed, np.ones(steps), mode="valid")))


def stairs_osl_conformance(
  cfg: Any, c: SimpleNamespace, steps: int = CONFORMANCE_STEPS
) -> dict[str, np.ndarray]:
  """Unicycle stage 2's OSL conformance sequence
  (:func:`~mjlab.tasks.osl_split.export.osl_conformance`) over ``steps``
  consecutive reference frames of the climb (:func:`climb_window`), labelled
  ``start_frame``."""
  start = climb_window(c.command, steps)
  qpos, qvel = stairs.reference_qpos_qvel(
    c.model, c.command, c.joint_names, range(start, start + steps)
  )
  return split.osl_conformance_sequence(
    cfg, c.model, c.actions, c.step_dt, qpos, qvel, {"start_frame": np.asarray(start)}
  )


##
# The bundle.
##


def _readme(manifest: dict[str, Any]) -> str:
  run, reference, terrain = manifest["run"], manifest["reference"], manifest["terrain"]
  human, osl = manifest["controllers"][HUMAN], manifest["controllers"][OSL]
  validation = manifest["validation"]
  solver = manifest["deployment"]["solver"]
  trained = manifest["deployment"]["training_solver"]
  lineage = manifest.get("lineage") or {}

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
  sensors = split.sensor_names(osl["observation"].get("sensor_set") or {})
  stage1 = lineage.get("stage1_checkpoint")
  return f"""# {run["task"]}: {run["run"]} / {run["checkpoint"]}

Two-controller stairs bundle (schema {manifest["schema_version"]},
`{manifest["kind"]}`) exported by mjlab (`mjlab.tasks.osl_split_stairs.export`) for
osl_deploy (plain CPU MuJoCo {manifest["mujoco_version"]}).

- Model: {run["model"]}, socket: {run["socket"]}, reward: {run["reward"]}
- Clip: `{reference["clip_file"]}`, {reference["num_frames"]} frames
  ({reference["duration_s"]:g} s), reference kind `{reference["kind"]}` (one clip,
  started at frame {reference["start_frame"]}, never re-centred, lookahead clamped
  at the last frame), read by the human controller
- Terrain: {terrain["box_count"]} boxes in `model.mjb` (body `{terrain["body"]}`,
  geoms `{terrain["geom_prefix"]}<i>`), from `{terrain["file"]}`
{touch_sites_readme_line(manifest["touch_sites"])}
- Checkpoint: `{run["checkpoint"]}` (iteration {run["iteration"]}, stage-2
  iteration {run["stage_iteration"]}); warm start: `{stage1}`
- Control: {1.0 / manifest["policy_dt"]:g} Hz for both controllers,
  {manifest["decimation"]} physics steps per action
- Solver caps of `model.mjb`: {solver["iterations"]} Newton,
  {solver["ls_iterations"]} line search (training:
  {trained["iterations"]}/{trained["ls_iterations"]} in mujoco_warp)
- Failure (validation's): mean relative site deviation above
  {manifest["failure"]["site_deviation"]["threshold_m"]:g} m

## Human controller (`{human["onnx"]}`)

- Observation: {human["observation"]["size"]} values, schema
  `{human["observation"]["schema"]}` (stairs stage 1's), terms
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
- Sensors: {sensors} (manifest `sensor_set`)
- Sensor latency in training: {latency}
- Action: {osl["action"]["size"]} outputs, blocks
  {", ".join(f"{b['name']} ({b['size']})" for b in osl["action"]["blocks"])};
  default inference **{osl["action"]["default_inference"]}**
- Actuator command delay in training: {delays(osl)}
- ONNX vs PyTorch max abs error: {validation["osl_onnx_max_abs_error"]:.3g}

`parity.npz` holds the human's observation conformance set
(`human_conformance_*`: CPU MuJoCo states at reference frames and the observation
mjlab's own terms build there) and the OSL's conformance sequence
(`osl_conformance_*`: consecutive reference states of the climb and the OSL
observation at each training latency).

Compare against the stage-1 teacher by exporting the warm start itself:

```
uv run --no-sync python -m mjlab.tasks.osl_stairs.export {stage1}
```
"""


def export_bundle(
  checkpoint: str | Path,
  output: str | Path | None = None,
  motion: str | Path | None = None,
  human_inference: Inference | None = None,
  overwrite: bool = False,
  solver_iterations: int | None = None,
  ls_solver_iterations: int | None = None,
) -> Path:
  """Export a stairs stage-2 ``checkpoint`` as a two-controller bundle (module
  docstring).

  Args:
    checkpoint: ``model_<N>.pt`` of a stairs stage-2 run.
    output: The bundle folder (default ``<run>/deploy/<checkpoint stem>``).
    motion: The clip, when the run's ``motion_file`` is not on this machine.
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
  spec = stairs_split_run_spec(checkpoint, agent, saved)
  terrain = stairs.check_stairs_run(checkpoint, saved)
  clip = stairs.resolve_motion(checkpoint, saved, motion)
  payload = split.load_payload(checkpoint)
  recorded = check_recorded_clip(payload, clip, terrain)
  stairs.warn_on_refit_sidecar(clip, terrain)
  split_info = dict((payload.get("infos") or {}).get("split") or {})
  lineage = split._plain(split_info.get("lineage"))
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
  osl_obs = run_osl_observation(checkpoint, payload, saved)
  print(
    f"[INFO] OSL sensors: {describe_osl_sensors(osl_obs.record)} (from "
    f"{osl_obs.source})."
  )

  cfg = stairs_split_config(
    spec,
    saved,
    touch_sites=touch.value,
    osl_record=osl_obs.record,
    motion=clip,
    terrain=terrain,
  )
  env, command, actions, solver = stairs.compile_stairs_task(
    cfg, solver_iterations, ls_solver_iterations
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
      agent[split.MODEL_KEYS[name]],
      payload[split.STATE_KEYS[name]],
      split.OBS_GROUPS[name],
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
      name: export_actor_onnx(actors[name], tmp / split.ONNX_FILES[name])
      for name in CONTROLLERS
    }
    sections = {
      HUMAN: split.human_controller(c, actors[HUMAN], dims[HUMAN], human_inference),
      OSL: split.osl_controller(c, actors[OSL], dims[OSL], lineage),
    }
    split.check_coverage(c.model, sections)
    iteration = int(payload["iter"]) if payload.get("iter") is not None else None
    reference = stairs.reference_section(command, clip, c.step_dt)
    reference["used_by"] = [HUMAN]
    manifest: dict[str, Any] = {
      "schema_version": split.SCHEMA_VERSION,
      "kind": split.BUNDLE_KIND,
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
        "variant": VARIANT,
        "experiment": spec.experiment,
        "run": spec.run,
        "checkpoint": checkpoint.name,
        "iteration": iteration,
        "stage_iteration": payload.get("stage_iteration"),
        "stage_start": split_info.get("stage_start"),
        "recorded_clip": recorded or None,
      },
      "reference": reference,
      "terrain": stairs.terrain_section(command, c.model, terrain),
      "initial_state": stairs.initial_state_section(),
      "failure": stairs.failure_section(),
      "deployment": {
        "solver": solver["deployment"],
        "training_solver": solver["training"],
        "note": stairs.DEPLOYMENT_NOTE,
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
    manifest["provenance"] = stairs.stairs_provenance(
      checkpoint,
      iteration,
      clip,
      terrain,
      SOURCE_FOLDERS,
    )
    mujoco.mj_saveModel(c.model, str(tmp / "model.mjb"))
    shutil.copyfile(terrain, tmp / stairs.TERRAIN_FILE)
    np.savez_compressed(tmp / "reference.npz", **stairs.reference_arrays(command))

    parity: dict[str, Any] = {}
    errors: dict[str, float] = {}
    rtol, atol = gl.PARITY_TOLERANCE
    for name in CONTROLLERS:
      actor = actors[name]
      observations = normalizer_samples(actor, gl.PARITY_SAMPLES)
      means, errors[name] = onnx_parity(
        tmp / split.ONNX_FILES[name], actor, observations
      )
      with torch.inference_mode():
        expected = actor.as_onnx(verbose=False)(torch.from_numpy(observations))  # type: ignore[operator]
      np.testing.assert_allclose(means, expected.numpy(), rtol=rtol, atol=atol)
      parity[f"{name}_observations"] = observations
      parity[f"{name}_means"] = means
    human = stairs.actor_conformance(
      cfg, c.model, command, cfg.observations["actor"], c.step_dt
    )
    if human["obs"].shape[1] != dims[HUMAN]:
      raise RuntimeError(
        f"The human conformance observation has {human['obs'].shape[1]} values; "
        f"the human actor reads {dims[HUMAN]}."
      )
    human["means"], human_error = onnx_parity(
      tmp / split.ONNX_FILES[HUMAN], actors[HUMAN], human["obs"].astype(np.float32)
    )
    errors[HUMAN] = max(errors[HUMAN], human_error)
    parity.update({f"human_conformance_{k}": human[k] for k in stairs.CONFORMANCE_KEYS})
    conformance = stairs_osl_conformance(cfg, c)
    parity.update({f"osl_conformance_{k}": v for k, v in conformance.items()})
    np.savez_compressed(tmp / "parity.npz", **parity)
    manifest["validation"] = {
      "human_onnx_max_abs_error": errors[HUMAN],
      "osl_onnx_max_abs_error": errors[OSL],
      "parity_inputs": (
        "<controller>_observations: the actor's observation mean, then mean + std "
        "* z (N(0, 1)); <controller>_means: their ONNX outputs"
      ),
      "human_conformance": {
        "keys": [f"human_conformance_{k}" for k in stairs.CONFORMANCE_KEYS],
        "frames": human["frames"].tolist(),
        "seed": stairs.CONFORMANCE_SEED,
        "procedure": stairs.conformance_procedure("human_conformance"),
      },
      "osl_conformance": {
        "keys": sorted(f"osl_conformance_{k}" for k in conformance),
        "steps": int(conformance["qpos"].shape[0]),
        "lags": conformance["lags"].tolist(),
        "start_frame": int(conformance["start_frame"]),
        "procedure": (
          "For each step t: mj_resetData; set qpos, qvel, act and ctrl from "
          "osl_conformance_*[t] (reference frame start_frame + t); mj_forward; "
          "the previous OSL command is osl_conformance_last_action[t] (0 at "
          "t = 0, a reset). Build the osl frame, push it through the latency and "
          "history buffers with a reset at t = 0, and compare with "
          "osl_conformance_obs_lag<L>[t] (float32 terms: about 1e-5 relative). "
          "osl_conformance_frames[t] is the undelayed frame."
        ),
      },
      "physics_transfer_verified": False,
    }
    manifest["files"] = {name: gl.sha256(tmp / name) for name in CHECKED_FILES}
    header = {k: manifest[k] for k in ("schema_version", "kind", "run")}
    for name in CONTROLLERS:
      attach_metadata(
        tmp / split.ONNX_FILES[name],
        split._plain({**header, "controller": name, **sections[name]}),
      )
      manifest["files"][split.ONNX_FILES[name]] = gl.sha256(
        tmp / split.ONNX_FILES[name]
      )
    sections[OSL]["observation"]["sensor_set"] = osl_sensor_set(
      cfg.observations[OSL_OBS_GROUP]
    )
    plain = split._plain(manifest)
    (tmp / "manifest.json").write_text(json.dumps(plain, indent=2) + "\n")
    (tmp / "README.md").write_text(_readme(plain))
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
class StairsSplitExportConfig:
  checkpoint: tyro.conf.Positional[Path]
  """``model_<N>.pt`` of a stairs stage-2 run (its run directory holds
  ``params/``)."""
  output: Path | None = None
  """Bundle folder; default ``<run>/deploy/<checkpoint stem>``."""
  motion: Path | None = None
  """The clip, when the run's ``motion_file`` is not on this machine."""
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

  cfg = tyro.cli(StairsSplitExportConfig, config=mjlab.TYRO_FLAGS)
  torch.set_num_threads(cfg.threads)
  bundle = export_bundle(
    cfg.checkpoint,
    cfg.output,
    cfg.motion,
    None if cfg.human_inference == "auto" else cfg.human_inference,
    cfg.overwrite,
    cfg.solver_iterations,
    cfg.ls_solver_iterations,
  )
  manifest = json.loads((bundle / "manifest.json").read_text())
  controllers = manifest["controllers"]
  print(
    f"[INFO] Bundle: {bundle}\n"
    f"[INFO] {manifest['run']['task']} on {manifest['reference']['clip_file']} "
    f"({manifest['terrain']['box_count']} boxes): human "
    f"{controllers[HUMAN]['action']['size']} outputs "
    f"({controllers[HUMAN]['action']['default_inference']}), OSL "
    f"{controllers[OSL]['action']['size']} outputs "
    f"({controllers[OSL]['action']['default_inference']}); ONNX max abs error "
    f"{manifest['validation']['human_onnx_max_abs_error']:.3g} / "
    f"{manifest['validation']['osl_onnx_max_abs_error']:.3g}"
  )
  if cfg.hf_repo:
    stairs.upload(bundle, cfg.hf_repo, cfg.hf_path, cfg.hf_public)


if __name__ == "__main__":
  main()
