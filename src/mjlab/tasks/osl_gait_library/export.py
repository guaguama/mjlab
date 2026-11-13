"""Export a gait-library checkpoint as a deployment bundle, on CPU, and upload it.

A bundle is the self-contained folder osl_deploy runs in plain CPU MuJoCo:

* ``actor.onnx``: the raw action mean behind the frozen observation statistics;
* ``manifest.json``: the policy contract (:mod:`mjlab.tasks.osl_mimic.rl.export`),
  the reference settings, provenance and checksums of every file;
* ``model.mjb``: the compiled scene with the task's simulation options and the OSL
  foot touch sites the run trained with (the manifest's ``touch_sites`` declares
  them, read from this file);
* ``reference.npz``: the library as the training command preprocessed it;
* ``parity.npz``: observations and their ONNX means, for a consumer's own check;
* ``README.md``: a summary for people (not checksummed).

Robot and MSK checkpoints of every gait-library task (forward and unicycle, either
reward and socket) export alike. The run's ``params/agent.yaml`` names its model,
socket, reward and variant; the task's config is rebuilt from code, with the OSL
touch sites the checkpoint was trained with (its ``osl_touch_sites`` key, else its
run's ``params/env.yaml`` field, else ``original``), and checked against the run's
``params/env.yaml`` (touch sites, simulation, decimation, observation terms, action
maps, actuator delays), and the reference settings are taken from the run.
No Warp simulation or GPU is involved::

  uv run --no-sync python -m mjlab.tasks.osl_gait_library.export \\
    logs/rsl_rl/<experiment>/<run>/model_<N>.pt \\
    [--library <clip folder>] [--hf-repo <user>/<repo>]

The bundle goes to ``<run>/deploy/model_<N>`` unless ``--output`` says otherwise.
``--hf-repo`` uploads it, by default to ``<run>/model_<N>`` in that repo (created
private when missing), through ``huggingface_hub`` when it is importable or the
``hf`` command line tool otherwise.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import mujoco
import numpy as np
import torch
import tyro

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  LEGACY_TOUCH_SITES,
  Model,
  Socket,
)
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.rl.export import (
  Inference,
  PolicyParts,
  action_std,
  attach_metadata,
  export_actor_onnx,
  normalizer_samples,
  onnx_parity,
  policy_metadata,
  unsupported_processing,
)
from mjlab.tasks.osl_mimic.touch_sites import (
  ENV_FIELD,
  run_touch_sites,
  saved_env_touch_sites,
  set_touch_sites,
  touch_sites_manifest,
  touch_sites_readme_line,
)

from .config.env_cfgs import Variant, osl_gait_library_env_cfg
from .mdp import GaitLibraryCommand, GaitLibraryCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.envs.mdp.actions.actions import BaseAction

ACTOR_CLASS = "mjlab.tasks.osl_mimic.rl.models:MuscleMimicMLPModel"
REFERENCE_FIELDS = (
  "root_pos",
  "root_quat",
  "root_lin_vel",
  "root_ang_vel_w",
  "joint_pos",
  "joint_vel",
  "site_pos",
  "rpos",
)
"""The command's preprocessed per-frame arrays a consumer replays (C, T_max, ...)."""
COMMAND_SETTINGS = (
  "site_bodies",
  "lookahead_steps",
  "lookahead_stride",
  "blend_time_s",
  "twist_metric_weights",
)
"""Reference settings taken from the run: they shape the observation."""
CHECKED_FILES = ("model.mjb", "reference.npz", "actor.onnx", "parity.npz")
PARITY_SAMPLES = 8
PARITY_TOLERANCE = (3e-4, 3e-5)
"""(rtol, atol) between onnxruntime and torch."""
SOURCE_FOLDERS = (
  "tasks/osl_mimic",
  "tasks/osl_gait_library",
  "asset_zoo/robots/osl_fullbody",
)


##
# The run.
##


def read_config(path: str | Path) -> dict[str, Any]:
  """A saved ``params/*.yaml`` as plain data: Python tags are never constructed."""
  import yaml

  class DataLoader(yaml.SafeLoader):
    pass

  def python_data(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
      return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
      return loader.construct_sequence(node, deep=True)
    value = loader.construct_scalar(cast(yaml.ScalarNode, node))
    return suffix.removeprefix("name:") if suffix.startswith("name:") else value

  DataLoader.add_multi_constructor("tag:yaml.org,2002:python/", python_data)
  return yaml.load(Path(path).read_text(), Loader=DataLoader)


def plain(value: Any) -> Any:
  """JSON types: tuples become lists, enums their values."""
  return json.loads(json.dumps(value, default=lambda x: x.value))


DEPLOY_SOLVER_ITERATIONS = 10
DEPLOY_LS_ITERATIONS = 20
"""The smallest solver caps a bundle gets (the run's own when higher). The OSL tasks
train at mujoco_warp's 4 Newton / 8 line-search iterations, where a policy behaves the
same as at 10 (the solves converge). CPU MuJoCo's Newton solver needs up to ~7
iterations at impratio 3 under the elliptic cone: capped at 4 it truncates the contact
solve, and a policy that validates in warp falls within a second in osl_deploy (the
first converged MSK run: 0 of 12 test clips survived 20 s at 4, 9 of 12 at 10, and 10
and 50 iterations were identical). Solves stop at convergence, so the higher cap only
costs time where it is needed."""


@dataclass(frozen=True)
class RunSpec:
  """What a gait-library run trained: from its ``params/agent.yaml``."""

  model: Model
  socket: Socket
  reward: Reward
  variant: Variant
  experiment: str
  run: str

  @property
  def task_id(self) -> str | None:
    from .config import _VARIANTS

    for task, (model, socket, reward, variant, _) in _VARIANTS.items():
      if (model, socket, reward, variant) == (
        self.model,
        self.socket,
        self.reward,
        self.variant,
      ):
        return task
    return None


def run_spec(checkpoint: Path, agent: dict[str, Any]) -> RunSpec:
  """The run's model, socket, reward and variant (forward runs predate
  ``variant``)."""
  v = agent.get("validation") or {}
  missing = [k for k in ("model", "socket", "reward") if k not in v]
  if missing:
    raise ValueError(
      f"{checkpoint.parent / 'params/agent.yaml'} has no validation.{missing}: not "
      "a gait-library run."
    )
  if v["model"] not in ("robot", "msk") or v["socket"] not in ("rigid", "compliant"):
    raise ValueError(f"Unknown model or socket in the run: {v}")
  if v["reward"] not in ("mimic", "clf"):
    raise ValueError(f"Unknown reward in the run: {v['reward']}")
  variant = v.get("variant", "walk_fwd")
  if variant not in ("walk_fwd", "unicycle"):
    raise ValueError(f"Unknown gait-library variant in the run: {variant}")
  return RunSpec(
    model=v["model"],
    socket=v["socket"],
    reward=v["reward"],
    variant=variant,
    experiment=str(agent.get("experiment_name", checkpoint.parent.parent.name)),
    run=checkpoint.parent.name,
  )


def config_mismatches(cfg: Any, saved: dict[str, Any]) -> list[str]:
  """Every setting the deployed policy depends on where the config rebuilt from
  this checkout differs from the run's ``env.yaml``: the OSL touch sites (a run
  without the field trained with ``original``), the simulation, decimation, actor
  terms, actions and actuator delays."""
  issues: list[str] = []

  def check(label: str, current: Any, original: Any) -> None:
    if plain(current) != original:
      issues.append(label)

  # A run's env.yaml without the field trained with the model XML's touch sites.
  touch_now = getattr(cfg, ENV_FIELD, LEGACY_TOUCH_SITES)
  touch_then = saved_env_touch_sites(saved) or LEGACY_TOUCH_SITES
  if touch_now != touch_then:
    issues.append(f"{ENV_FIELD} ({touch_then} -> {touch_now})")

  mujoco_now = plain(dataclasses.asdict(cfg.sim.mujoco))
  mujoco_then = saved["sim"]["mujoco"]
  for key in sorted(set(mujoco_now) | set(mujoco_then)):
    if mujoco_now.get(key) != mujoco_then.get(key):
      issues.append(f"sim.mujoco.{key}")
  check("decimation", cfg.decimation, saved["decimation"])

  actor = cfg.observations["actor"]
  saved_actor = saved["observations"]["actor"]
  if list(actor.terms) != list(saved_actor["terms"]):
    issues.append(
      f"observations.actor.terms {list(saved_actor['terms'])} -> {list(actor.terms)}"
    )
  else:
    for name, term in actor.terms.items():
      original = saved_actor["terms"][name]
      if original["func"] != f"{term.func.__module__}.{term.func.__name__}":
        issues.append(f"observations.actor.{name}.func")
      for key in ("params", "scale", "clip", "history_length", "flatten_history_dim"):
        if key in original:
          check(f"observations.actor.{name}.{key}", getattr(term, key), original[key])

  if list(cfg.actions) != list(saved["actions"]):
    issues.append(f"actions {list(saved['actions'])} -> {list(cfg.actions)}")
  else:
    for name, action in cfg.actions.items():
      for key in ("actuator_names", "scale", "offset", "clip", "preserve_order"):
        check(
          f"actions.{name}.{key}", getattr(action, key), saved["actions"][name][key]
        )

  now = cfg.scene.entities["robot"].articulation.actuators
  then = saved["scene"]["entities"]["robot"]["articulation"]["actuators"]
  if len(now) != len(then):
    issues.append("scene.entities.robot.articulation.actuators")
  else:
    for i, (a, b) in enumerate(zip(now, then, strict=True)):
      for key in ("target_names_expr", "delay_min_lag", "delay_max_lag"):
        check(f"actuators[{i}].{key}", getattr(a, key), b.get(key))
  return issues


@dataclass
class Components:
  """A run's task, rebuilt on CPU: config, compiled model, command and actions."""

  spec: RunSpec
  cfg: Any
  env: SimpleNamespace
  command: GaitLibraryCommand
  parts: PolicyParts
  solver: dict[str, Any] = field(default_factory=dict)
  """The compiled model's solver caps and the run's (``deployment``/``training``)."""


def build_components(
  checkpoint: str | Path,
  library: str | Path | None = None,
  solver_iterations: int | None = None,
  ls_solver_iterations: int | None = None,
) -> Components:
  """Rebuild the run's task on CPU, without a Warp simulation.

  Args:
    checkpoint: ``model_<N>.pt`` inside a run directory with ``params/``.
    library: The clip folder, when the run's ``library_dir`` is not on this machine.
    solver_iterations, ls_solver_iterations: The compiled model's solver caps;
      default the run's own, raised to ``DEPLOY_SOLVER_ITERATIONS`` and
      ``DEPLOY_LS_ITERATIONS`` (see there).

  The model builds the OSL touch sites the checkpoint was trained with
  (``touch_sites.run_touch_sites``).

  Raises:
    ValueError: If the run is not a gait-library run, the config rebuilt from this
      checkout differs from the run's in a setting the policy depends on, the
      checkpoint and its run's env.yaml record different touch sites, or the
      library folder is missing.
  """
  checkpoint = Path(checkpoint).resolve()
  params = checkpoint.parent / "params"
  agent, saved = read_config(params / "agent.yaml"), read_config(params / "env.yaml")
  spec = run_spec(checkpoint, agent)
  cfg = osl_gait_library_env_cfg(
    spec.model, spec.socket, spec.reward, "train", spec.variant
  )
  touch = run_touch_sites(checkpoint, saved_env=saved)
  set_touch_sites(cfg, touch.value)
  print(f"[INFO] OSL touch sites: {touch.value} (from {touch.source}).")
  check_actor_config(cfg, saved)
  env, command, actions, solver = compile_task(
    cfg, saved, library, solver_iterations, ls_solver_iterations
  )
  command_cfg = command.cfg
  actor = cfg.observations["actor"]
  parts = PolicyParts(
    model=env.sim.mj_model,
    joint_names=list(env.scene["robot"].joint_names),
    obs_terms=list(actor.terms),
    obs_term_cfgs=dict(actor.terms),
    actions=actions,
    site_bodies=list(command_cfg.site_bodies),
    lookahead_steps=int(command_cfg.lookahead_steps),
    decimation=int(cfg.decimation),
    step_dt=float(env.step_dt),
    entity_cfg=cfg.scene.entities["robot"],
    obs_group_history=actor.history_length,
  )
  return Components(spec, cfg, env, command, parts, solver)


def check_actor_config(cfg: Any, saved: dict[str, Any]) -> None:
  """Refuse a run whose actor input is processed, or whose config rebuilt from
  this checkout differs from its ``env.yaml`` (:func:`config_mismatches`).

  Raises:
    ValueError: On either.
  """
  saved_actor = saved["observations"]["actor"]
  processed = unsupported_processing(
    saved_actor["terms"], saved_actor.get("history_length")
  )
  if processed:
    raise ValueError(f"The run's actor input is processed ({processed}).")
  issues = config_mismatches(cfg, saved)
  if issues:
    raise ValueError(
      "The task rebuilt from this checkout differs from the run in: "
      + ", ".join(issues)
    )


def compile_task(
  cfg: ManagerBasedRlEnvCfg,
  saved: dict[str, Any],
  library: str | Path | None = None,
  solver_iterations: int | None = None,
  ls_solver_iterations: int | None = None,
) -> tuple[SimpleNamespace, GaitLibraryCommand, dict[str, BaseAction], dict]:
  """Compile a gait-library task config on CPU: the run's reference settings and
  library, the deployment solver caps, the scene, the command and the actions.

  ``cfg`` is edited in place (its command settings and solver caps). Returns the
  CPU env namespace, the command, the action terms in action-manager order and
  the solver caps (``training`` and ``deployment``).

  Raises:
    ValueError: If the run's command cannot be reproduced, the library folder is
      missing or a solver cap is not positive.
  """
  from mjlab.scene import Scene

  command_cfg = cfg.commands["motion"]
  assert isinstance(command_cfg, GaitLibraryCommandCfg)
  saved_motion = saved["commands"]["motion"]
  if not saved_motion.get("rebase_on_resample", True):
    raise ValueError("The bundle's reference schema needs rebase_on_resample=True.")
  for key in COMMAND_SETTINGS:
    value = saved_motion[key]
    setattr(command_cfg, key, tuple(value) if isinstance(value, list) else value)
  folder = Path(library or saved_motion["library_dir"]).expanduser()
  if not folder.is_dir() or not any(folder.glob("*.npz")):
    raise ValueError(
      f"No gait library at {folder}. Pass --library <folder of the run's clips>."
    )
  command_cfg.library_dir = str(folder.resolve())

  mj = cfg.sim.mujoco
  solver = {
    "training": {"iterations": mj.iterations, "ls_iterations": mj.ls_iterations},
  }
  mj.iterations = (
    max(mj.iterations, DEPLOY_SOLVER_ITERATIONS)
    if solver_iterations is None
    else int(solver_iterations)
  )
  mj.ls_iterations = (
    max(mj.ls_iterations, DEPLOY_LS_ITERATIONS)
    if ls_solver_iterations is None
    else int(ls_solver_iterations)
  )
  if mj.iterations < 1 or mj.ls_iterations < 1:
    raise ValueError("Solver iteration caps must be positive.")
  solver["deployment"] = {
    "iterations": mj.iterations,
    "ls_iterations": mj.ls_iterations,
  }

  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, "cpu")
  model = scene.compile()
  cfg.sim.mujoco.apply(model)
  env = SimpleNamespace(
    scene=scene,
    device="cpu",
    num_envs=1,
    step_dt=cfg.sim.mujoco.timestep * cfg.decimation,
    common_step_counter=0,
    cfg=cfg,
  )
  env.sim = SimpleNamespace(mj_model=model)
  command = command_cfg.build(env)  # type: ignore[arg-type]
  env.command_manager = SimpleNamespace(get_term=lambda _: command)
  actions = {
    name: cast("BaseAction", action.build(env))  # type: ignore[arg-type]
    for name, action in cfg.actions.items()
  }
  return env, command, actions, solver


def load_actor(checkpoint: str | Path, action_dim: int) -> torch.nn.Module:
  """The checkpoint's actor, loaded strictly, normalizer frozen, in eval mode."""
  checkpoint = Path(checkpoint)
  agent = read_config(checkpoint.parent / "params" / "agent.yaml")
  if agent.get("clip_actions") is not None:
    raise ValueError("Wrapper action clipping is outside the bundle schema.")
  if list(agent["obs_groups"]["actor"]) != ["actor"]:
    raise ValueError("The actor must read the 'actor' observation group alone.")
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  return build_actor(agent["actor"], payload["actor_state_dict"], "actor", action_dim)


def build_actor(
  model_cfg: dict[str, Any],
  state: dict[str, torch.Tensor],
  obs_group: str,
  action_dim: int,
) -> torch.nn.Module:
  """A :data:`ACTOR_CLASS` actor from its saved model config and state dict,
  loaded strictly, normalizer frozen, in eval mode.

  Args:
    model_cfg: The actor's section of ``params/agent.yaml``.
    state: Its state dict from the checkpoint.
    obs_group: The observation group it reads (its input width comes from the
      state's normalizer).
    action_dim: Its output width.

  Raises:
    ValueError: For another model class, or a convolutional or recurrent actor.
  """
  from tensordict import TensorDict

  from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel

  actor_cfg = dict(model_cfg)
  if actor_cfg.pop("class_name") != ACTOR_CLASS:
    raise ValueError(f"Unsupported actor model; expected {ACTOR_CLASS}.")
  for key in ("cnn_cfg", "rnn_type"):
    if actor_cfg.pop(key, None) is not None:
      raise ValueError("The bundle schema supports feed-forward actors only.")
  for key in ("rnn_hidden_dim", "rnn_num_layers"):
    actor_cfg.pop(key, None)
  obs_dim = int(state["obs_normalizer.mean"].numel())
  # Shapes on the meta device: an orthogonal init of 25M weights costs seconds.
  with torch.device("meta"):
    obs = TensorDict({obs_group: torch.zeros(1, obs_dim)}, batch_size=[1])
    actor = MuscleMimicMLPModel(
      obs, {"actor": [obs_group]}, "actor", action_dim, **actor_cfg
    )
  actor.load_state_dict(state, strict=True, assign=True)
  actor.set_norm_mode("frozen")
  actor.eval()
  return actor


##
# The bundle.
##


def sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
  try:
    return subprocess.check_output(
      ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def _provenance(
  checkpoint: Path,
  library: Path,
  iteration: int | None,
  source_folders: tuple[str, ...] = SOURCE_FOLDERS,
) -> dict:
  import mjlab

  root = Path(mjlab.__file__).resolve().parent
  status = _git(root, "status", "--porcelain", "--", *source_folders)
  return {
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "iteration": iteration,
    "agent_config_sha256": sha256(checkpoint.parent / "params/agent.yaml"),
    "env_config_sha256": sha256(checkpoint.parent / "params/env.yaml"),
    "library_dir": str(library),
    "library": {p.name: sha256(p) for p in sorted(library.glob("*.npz"))},
    "training_sources": {
      str(p.relative_to(root)): sha256(p)
      for folder in source_folders
      for p in sorted((root / folder).rglob("*"))
      if p.is_file() and p.suffix in (".py", ".xml")
    },
    "mjlab_head": _git(root, "rev-parse", "HEAD"),
    "mjlab_sources_modified": bool(status) if status is not None else None,
    "note": (
      "Scene and reference rebuilt from this checkout and the library above; the "
      "checkpoint records neither model nor clip hashes, so their identity at "
      "training time is checked by config only."
    ),
  }


def _readme(manifest: dict[str, Any]) -> str:
  run, action = manifest["run"], manifest["action"]
  delays = manifest["training"]["actuator_delay"]
  delay_lines = "\n".join(
    f"- `{name}`: {d['seconds'][0] * 1e3:g}-{d['seconds'][1] * 1e3:g} ms"
    if d
    else f"- `{name}`: unknown"
    for name, d in delays.items()
  )
  validation = manifest["validation"]
  deployment = manifest.get("deployment") or {}
  solver = deployment.get("solver") or {"iterations": "?", "ls_iterations": "?"}
  trained = deployment.get("training_solver") or {
    "iterations": "?",
    "ls_iterations": "?",
  }
  return f"""# {run["task"] or "OSL gait-library policy"}: {run["run"]} / {run["checkpoint"]}

Deployment bundle exported by mjlab (`mjlab.tasks.osl_gait_library.export`) for
osl_deploy (plain CPU MuJoCo {manifest["mujoco_version"]}).

- Model: {run["model"]}, socket: {run["socket"]}, reward: {run["reward"]}, library:
  {run["variant"]} ({len(manifest["reference"]["clip_names"])} clips)
{touch_sites_readme_line(manifest["touch_sites"])}
- Checkpoint: `{run["checkpoint"]}` (iteration {manifest["provenance"]["iteration"]})
- Control: {1.0 / manifest["policy_dt"]:g} Hz, {manifest["decimation"]} physics steps
  per action
- Observation: {manifest["observation"]["size"]} values,
  terms {", ".join(manifest["observation"]["terms"])}
- Action: {action["size"]} outputs, blocks
  {", ".join(f"{b['name']} ({b['size']})" for b in action["blocks"])}
- Default inference: **{action["default_inference"]}**
- ONNX vs PyTorch max abs error: {validation["onnx_max_abs_error"]:.3g}
- Solver caps of `model.mjb`: {solver["iterations"]} Newton, {solver["ls_iterations"]}
  line search (training: {trained["iterations"]}/{trained["ls_iterations"]} in mujoco_warp,
  where its solves converge; CPU MuJoCo needs the higher cap)

Inference: `mean` applies the raw policy mean; `expected` applies
E[clip(N(mean, std^2), lo, hi)] with the learned std. A muscle policy trained with
a large std (around 1 or more) needs `expected`, because clipping makes the noise
itself drive its muscles; one annealed to a small std runs on the raw mean, as robot
policies do.

Actuator command delays in training (deployment must add them itself):
{delay_lines}

Files: `actor.onnx` (input `observation` (1, {manifest["observation"]["size"]}),
output `mean` (1, {action["size"]})), `manifest.json` (the full contract and
checksums), `model.mjb`, `reference.npz`, `parity.npz`.

```
osl-deploy <this folder> --list-clips
```
"""


def export_bundle(
  checkpoint: str | Path,
  output: str | Path | None = None,
  library: str | Path | None = None,
  inference: Inference | None = None,
  overwrite: bool = False,
  solver_iterations: int | None = None,
  ls_solver_iterations: int | None = None,
) -> Path:
  """Export ``checkpoint`` as a bundle (see the module docstring); returns it.

  Args:
    checkpoint: ``model_<N>.pt`` of a gait-library run.
    output: The bundle folder (default ``<run>/deploy/<checkpoint stem>``).
    library: The clip folder, when the run's is not on this machine.
    inference: The manifest's default inference; None picks ``expected`` for MSK
      policies whose median muscle std exceeds 0.3 and ``mean`` for the others.
    overwrite: Replace an existing bundle (once the new one is complete).
    solver_iterations, ls_solver_iterations: The bundle model's solver caps (see
      :func:`build_components`); recorded in the manifest's ``deployment``.
  """
  checkpoint = Path(checkpoint).resolve()
  output = Path(output or checkpoint.parent / "deploy" / checkpoint.stem).resolve()
  if output.exists() and not overwrite:
    raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
  c = build_components(checkpoint, library, solver_iterations, ls_solver_iterations)
  parts, command = c.parts, c.command
  action_dim = sum(term.action_dim for term in parts.actions.values())
  actor = load_actor(checkpoint, action_dim)
  std = action_std(actor, action_dim)
  payload_iter = torch.load(checkpoint, map_location="meta", weights_only=False).get(
    "iter"
  )

  output.parent.mkdir(parents=True, exist_ok=True)
  tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
  umask = os.umask(0)
  os.umask(umask)
  tmp.chmod(0o777 & ~umask)  # mkdtemp's 0700 would hide the bundle from others.
  try:
    obs_dim = export_actor_onnx(actor, tmp / "actor.onnx")
    manifest: dict[str, Any] = policy_metadata(parts, std, obs_dim, inference)
    manifest["socket"] = c.spec.socket
    manifest["touch_sites"] = touch_sites_manifest(parts.model)
    if manifest["touch_sites"]["version"] != c.cfg.osl_touch_sites:
      raise RuntimeError(
        f"model.mjb carries the {manifest['touch_sites']['version']!r} touch sites, "
        f"but the run trained with {c.cfg.osl_touch_sites!r}."
      )
    manifest["run"] = {
      "task": c.spec.task_id,
      "model": c.spec.model,
      "socket": c.spec.socket,
      "reward": c.spec.reward,
      "variant": c.spec.variant,
      "experiment": c.spec.experiment,
      "run": c.spec.run,
      "checkpoint": checkpoint.name,
    }
    manifest["reference"] = {
      "clip_names": list(command.clip_names),
      "policy_dt": parts.step_dt,
      "lookahead_steps": command.cfg.lookahead_steps,
      "lookahead_stride": command.cfg.lookahead_stride,
      "blend_time_s": command.cfg.blend_time_s,
      "twist_metric_weights": list(command.cfg.twist_metric_weights),
    }
    manifest["deployment"] = {
      "solver": c.solver["deployment"],
      "training_solver": c.solver["training"],
      "note": (
        "Solver caps of model.mjb. Training ran mujoco_warp at training_solver, "
        "where its solves converge; CPU MuJoCo needs a higher cap to converge "
        "(it truncates the contact solve at 4 iterations under the elliptic cone at "
        "impratio 3)."
      ),
    }
    library_dir = Path(command.cfg.library_dir)
    manifest["provenance"] = _provenance(
      checkpoint, library_dir, int(payload_iter) if payload_iter is not None else None
    )

    mujoco.mj_saveModel(parts.model, str(tmp / "model.mjb"))
    arrays: dict[str, Any] = {
      key: command.data[key].cpu().numpy() for key in REFERENCE_FIELDS
    }
    arrays["num_frames"] = command.num_frames.cpu().numpy()
    arrays["twists"] = command.twists.cpu().numpy()
    arrays["strides"] = command.strides.cpu().numpy()
    np.savez_compressed(tmp / "reference.npz", **arrays)
    observations = normalizer_samples(actor, PARITY_SAMPLES)
    means, error = onnx_parity(tmp / "actor.onnx", actor, observations)
    rtol, atol = PARITY_TOLERANCE
    with torch.inference_mode():
      reference_means = actor.as_onnx(verbose=False)(torch.from_numpy(observations))  # type: ignore[operator]
    np.testing.assert_allclose(means, reference_means.numpy(), rtol=rtol, atol=atol)
    np.savez_compressed(
      tmp / "parity.npz",
      observation=observations[0],
      mean=means[0],
      observations=observations,
      means=means,
    )
    manifest["validation"] = {
      "onnx_max_abs_error": error,
      "parity_inputs": "the actor's observation mean, then mean + std * z (N(0, 1))",
      "physics_transfer_verified": False,
    }
    manifest["files"] = {name: sha256(tmp / name) for name in CHECKED_FILES}
    attach_metadata(
      tmp / "actor.onnx", {k: v for k, v in manifest.items() if k != "files"}
    )
    manifest["files"]["actor.onnx"] = sha256(tmp / "actor.onnx")
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
# Hugging Face.
##


def upload_command(
  bundle: Path, repo: str, path_in_repo: str, private: bool, message: str
) -> list[str]:
  """The ``hf upload`` call for a bundle."""
  return [
    "hf",
    "upload",
    repo,
    str(bundle),
    path_in_repo,
    "--repo-type",
    "model",
    "--private" if private else "--no-private",
    "--commit-message",
    message,
  ]


def upload_bundle(
  bundle: str | Path,
  repo: str,
  path_in_repo: str | None = None,
  private: bool = True,
) -> str:
  """Upload a bundle folder to a Hugging Face model repo; returns its URL.

  Uses ``huggingface_hub`` when importable, otherwise the ``hf`` command line tool
  on ``PATH`` (both read the token ``hf auth login`` stores). A missing repo is
  created, private unless ``private=False``.

  Raises:
    RuntimeError: If neither is available or the upload fails.
  """
  bundle = Path(bundle).resolve()
  manifest = json.loads((bundle / "manifest.json").read_text())
  run = manifest.get("run", {})
  path_in_repo = path_in_repo or f"{run.get('run', bundle.parent.name)}/{bundle.name}"
  message = f"Add deployment bundle {path_in_repo} ({run.get('task') or 'osl'})"
  url = f"https://huggingface.co/{repo}/tree/main/{path_in_repo}"
  try:
    from huggingface_hub import HfApi  # type: ignore[import-not-found]
  except ImportError:
    HfApi = None
  if HfApi is not None:
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
      folder_path=str(bundle),
      repo_id=repo,
      repo_type="model",
      path_in_repo=path_in_repo,
      commit_message=message,
    )
    return url
  hf = shutil.which("hf")
  command = upload_command(bundle, repo, path_in_repo, private, message)
  if hf is None:
    raise RuntimeError(
      "Neither huggingface_hub nor the hf tool is available. Upload with:\n  "
      + " ".join(command)
    )
  done = subprocess.run([hf, *command[1:]], check=False)
  if done.returncode != 0:
    raise RuntimeError(
      f"hf upload exited with {done.returncode}. Retry with:\n  " + " ".join(command)
    )
  return url


##
# Command line.
##


@dataclass
class ExportConfig:
  checkpoint: tyro.conf.Positional[Path]
  """``model_<N>.pt`` of a gait-library run (its run directory holds ``params/``)."""
  output: Path | None = None
  """Bundle folder; default ``<run>/deploy/<checkpoint stem>``."""
  library: Path | None = None
  """Clip folder, when the run's ``library_dir`` is not on this machine."""
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
  """The bundle model's Newton iteration cap; default the run's, at least
  ``DEPLOY_SOLVER_ITERATIONS`` (10)."""
  ls_solver_iterations: int | None = None
  """Its line-search cap; default the run's, at least ``DEPLOY_LS_ITERATIONS`` (20)."""


def main() -> None:
  import mjlab

  cfg = tyro.cli(ExportConfig, config=mjlab.TYRO_FLAGS)
  torch.set_num_threads(cfg.threads)
  inference = None if cfg.inference == "auto" else cfg.inference
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
  print(
    f"[INFO] Bundle: {bundle}\n"
    f"[INFO] {manifest['run']['task']}, default inference "
    f"{manifest['action']['default_inference']}, ONNX max abs error "
    f"{manifest['validation']['onnx_max_abs_error']:.3g}"
  )
  if cfg.hf_repo:
    try:
      url = upload_bundle(bundle, cfg.hf_repo, cfg.hf_path, private=not cfg.hf_public)
    except Exception as err:
      print(f"[WARN] Upload failed: {err}", file=sys.stderr)
      sys.exit(1)
    print(f"[INFO] Uploaded: {url}")


if __name__ == "__main__":
  main()
