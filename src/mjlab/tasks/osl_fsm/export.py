"""Export a stage-1.5 checkpoint as a deployment bundle for osl_deploy, on CPU.

Stage 1.5 trains stage 2's human actor beside the OSL's classical walking
controller (:mod:`mjlab.tasks.osl_fsm`). No OSL network was trained, so the bundle
is a stage-2 bundle without one, which osl_deploy's state machine completes
(``python -m osl_deploy.osl_fsm``): schema 2, kind ``osl_split_dual`` (the kind
osl_deploy composes controllers from), with

* ``human_actor.onnx`` and ``controllers.human``: the human actor and its section,
  exactly as stage 2 writes them (:func:`~mjlab.tasks.osl_split.export.
  human_controller`);
* ``controllers.osl``: stage 2's OSL section without ``onnx`` and ``io``. Its
  ``observation`` (the ``osl_split_sensors_v1`` contract: channels, sensors, tare,
  saturation, history, latency, reset), ``action`` (the motor block, std 0) and
  ``training`` sections stay, because the state machine reads its sensors and its
  motor map from them. The run's ``osl`` group is clean, so the latency section has
  no buses: the controller read its sensors exactly and at once. ``memory`` says the
  controller is a state machine, and ``fsm`` holds the controller as the run
  recorded it, the osl_deploy command line that reproduces it and whether its gains
  and thresholds are osl_deploy's own;
* ``model.mjb``: the compiled stage-1.5 scene with the OSL sensors, at the
  deployment solver caps, with the OSL foot touch sites the run trained with (the
  manifest's ``touch_sites`` declares them, read from this file);
* ``reference.npz``: the library as the training command preprocessed it;
* ``parity.npz``: the human actor's ONNX outputs on inputs drawn from its own
  statistics; stage 2's OSL observation conformance sequence
  (``osl_conformance_*``, :func:`~mjlab.tasks.osl_split.export.osl_conformance`),
  over :data:`CONFORMANCE_STEPS` steps of a looped clip; and the controller's own
  conformance (``osl_fsm_conformance_*``, :func:`fsm_conformance`): the states,
  torques and actions mjlab's controller takes on that sequence's rows, which
  osl_deploy's ``fsm_policy`` reproduces row by row;
* ``README.md``: a summary for people, the deploy command first (not checksummed).

There is no ``osl_actor.onnx``. osl_deploy's ``osl_fsm`` defaults to the sdk rules,
so ``--rules`` is part of the deploy command (``controllers.osl.fsm.
deploy_command``) whichever rules the run trained with::

  uv run --no-sync python -m mjlab.tasks.osl_fsm.export \\
    logs/rsl_rl/<stage-1.5 experiment>/<run>/model_<N>.pt \\
    [--library <clip folder>] [--hf-repo <user>/<repo>]

The bundle goes to ``<run>/deploy/model_<N>`` unless ``--output`` says otherwise.
The run's ``params/`` decide the model, socket and reward, and its checkpoint the
controller (``infos["osl_fsm"]``) and the OSL foot touch sites. The task is rebuilt
from this checkout with both; the run's env.yaml must match it (as stage 2's: the
touch sites, simulation, decimation, the human's observation, actions, actuator
delays, and here the controller settings too), and the controller rebuilt from
this checkout must be the one the run recorded (gains, thresholds, body weight,
tare, motor map).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket, TouchSites
from mjlab.tasks.osl_gait_library import export as gl
from mjlab.tasks.osl_gait_library.mdp.commands import se2_compose
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.rl.export import (
  Inference,
  action_metadata,
  attach_metadata,
  export_actor_onnx,
  normalizer_samples,
  onnx_parity,
  training_metadata,
)
from mjlab.tasks.osl_mimic.touch_sites import (
  run_touch_sites,
  set_touch_sites,
  touch_sites_manifest,
  touch_sites_readme_line,
)
from mjlab.tasks.osl_split import export as split
from mjlab.tasks.osl_split.rl.export import osl_policy_metadata
from mjlab.tasks.osl_split.rl.layout import OSL_ACTION_TERM, ActionLayout
from mjlab.tasks.osl_split.rl.ppo import OSL_OBS_GROUP
from mjlab.tasks.osl_split.rl.warmstart import lineage_touch_sites
from mjlab.utils.lab_api.math import quat_mul

from .config import fsm_task_id
from .config.env_cfgs import OslFsmEnvCfg, configure_fsm, osl_fsm_env_cfg
from .fsm import (
  JOINTS,
  RULES,
  STATES,
  BatchedFsm,
  impedance_settings,
  rule_set,
  threshold_settings,
)
from .mdp import READINGS, SIGNALS, OslFsmAction, OslFsmActionCfg
from .mdp.actions import _joint_of_target
from .rl.ppo import ALGORITHM
from .rl.runner import FSM_INFO_KEY, recorded_fsm_settings

if TYPE_CHECKING:
  from mjlab.envs.mdp.actions.actions import BaseAction
  from mjlab.managers.observation_manager import ObservationGroupCfg

HUMAN = split.HUMAN
OSL = split.OSL
ONNX_FILE = split.ONNX_FILES[HUMAN]
CHECKED_FILES = ("model.mjb", "reference.npz", ONNX_FILE, "parity.npz")
"""The checksummed files: no ``osl_actor.onnx``, since no OSL network was trained."""
ALGORITHM_CLASS = "mjlab.tasks.osl_fsm.rl.ppo:HumanOnlyPPO"
HUMAN_STATE_KEY = "human_actor_state_dict"
SOURCE_FOLDERS = (*split.SOURCE_FOLDERS, "tasks/osl_fsm")
STAGE = 1.5
OSL_CONTROLLER = "osl_fsm"
"""``run.osl_controller``: what drives the OSL, osl_deploy's state machine."""
FSM_FORMAT = "mjlab.osl_fsm/1"
"""``controllers.osl.fsm.format``; bump it on layout changes."""
FSM_SETTINGS = (
  "rules",
  "every",
  "knee_velocity",
  "swing_load_exit",
  "body_mass",
  "body_mass_kg",
  "body_weight_n",
  "initial_state",
  "tare_n",
  "nm_per_ctrl",
  "ctrl_limits",
  "impedance",
  "thresholds",
  "sensors",
  "update",
)
"""The recorded controller settings (:meth:`~.mdp.OslFsmAction.settings`) the
``fsm`` block carries; ``body_mass`` is the configured one (None: the model's), and
``initial_state`` training's."""
FSM_CONTROLLER = "fsm"
"""The action term's ``controller`` a stage-1.5 run trains with: the walking
controller drives the OSL (``"policy"`` would give it the policy's zero rows)."""
DEPLOY_MODULE = "osl_deploy.osl_fsm"
DEPLOY_INITIAL_STATE = "e_stance"
"""Where osl_deploy's state machine starts every episode."""
CONFORMANCE_STEPS = 250
"""Rows of the conformance sequences: 2.5 s, over two strides of the conformance
clip (1.18 s at 1 m/s; stage 2 plays 40 steps), so the controller passes through
its states several times. The clip loops past its last frame (:func:`looped_clip`)."""
CLIP_FIELDS = (
  "root_pos",
  "root_quat",
  "root_lin_vel",
  "root_ang_vel_w",
  "joint_pos",
  "joint_vel",
)
"""The command arrays the conformance states are built from
(:func:`~mjlab.tasks.osl_split.export.reference_states`)."""
NORMALIZATION = (
  "none: the state machine reads the raw values (no network, no normalizer)"
)


##
# The run.
##


@dataclass(frozen=True)
class FsmRunSpec:
  """What a stage-1.5 run trained: from its ``params/agent.yaml``."""

  model: Model
  socket: Socket
  reward: Reward
  experiment: str
  run: str

  @property
  def task_id(self) -> str:
    return fsm_task_id(self.model, self.socket, self.reward)


def fsm_run_spec(checkpoint: Path, agent: Mapping[str, Any]) -> FsmRunSpec:
  """The run's model, socket and reward, checked to be a stage-1.5 run whose human
  actor this bundle schema covers.

  Raises:
    ValueError: For anything else (a stage-2 or stage-1 run: the message names its
      exporter).
  """
  algorithm = agent.get("algorithm") or {}
  name = algorithm.get("class_name")
  if name != ALGORITHM_CLASS:
    if name == split.ALGORITHM_CLASS:
      hint = "Export a stage-2 run with python -m mjlab.tasks.osl_split.export."
    else:
      hint = "Export a stage-1 run with python -m mjlab.tasks.osl_gait_library.export."
    raise ValueError(
      f"{checkpoint.parent / 'params/agent.yaml'} is not a stage-1.5 run "
      f"(algorithm {name!r}). {hint}"
    )
  spec = gl.run_spec(checkpoint, dict(agent))
  if spec.variant != "unicycle":
    raise ValueError(f"Stage 1.5 runs on the unicycle library, not {spec.variant!r}.")
  if agent.get("clip_actions") is not None:
    raise ValueError("Wrapper action clipping is outside the bundle schema.")
  groups = agent.get("obs_groups") or {}
  if list(groups.get("actor") or ()) != ["actor"]:
    raise ValueError(
      "The human actor must read the 'actor' observation group alone "
      f"(obs_groups.actor = {groups.get('actor')})."
    )
  return FsmRunSpec(spec.model, spec.socket, spec.reward, spec.experiment, spec.run)


def load_payload(checkpoint: Path) -> dict[str, Any]:
  """The stage-1.5 checkpoint, memory-mapped (only the human actor and the
  controller settings are read).

  Raises:
    ValueError: If it is not a stage-1.5 checkpoint (a stage-2 or stage-1 one: the
      message names its exporter), or records no controller settings.
  """
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
  if not isinstance(payload, dict):
    raise ValueError(f"{checkpoint} is not a checkpoint dictionary.")
  kind = payload.get("algorithm")
  if kind != ALGORITHM:
    if "osl_actor_state_dict" in payload:
      hint = (
        " It is a stage-2 checkpoint: export it with python -m "
        "mjlab.tasks.osl_split.export."
      )
    elif "actor_state_dict" in payload:
      hint = (
        " It is a stage-1 checkpoint: export it with python -m "
        "mjlab.tasks.osl_gait_library.export."
      )
    else:
      hint = ""
    raise ValueError(
      f"{checkpoint} is not a stage-1.5 checkpoint (algorithm {kind!r}).{hint}"
    )
  if HUMAN_STATE_KEY not in payload:
    raise ValueError(f"{checkpoint} lacks the human actor ({HUMAN_STATE_KEY!r}).")
  if recorded_fsm_settings(payload) is None:
    raise ValueError(
      f"{checkpoint} records no OSL controller settings (infos[{FSM_INFO_KEY!r}]): "
      "not a checkpoint of a stage-1.5 run."
    )
  return payload


def stored_settings(checkpoint: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
  """The controller the checkpoint's env ran (``infos["osl_fsm"]``), checked to be
  complete and to be the walking controller.

  Raises:
    ValueError: For missing settings, or an OSL that followed the policy's rows.
  """
  settings = recorded_fsm_settings(payload) or {}
  missing = [key for key in (*FSM_SETTINGS, "controller") if key not in settings]
  if missing:
    raise ValueError(f"{checkpoint}'s OSL controller settings lack {missing}.")
  if settings["controller"] != FSM_CONTROLLER:
    raise ValueError(
      f"{checkpoint}'s OSL followed the policy's rows (controller "
      f"{settings['controller']!r}), not the walking controller: there is no OSL "
      "controller to deploy."
    )
  return settings


def check_saved_controller(
  saved: Mapping[str, Any], settings: Mapping[str, Any]
) -> None:
  """The run's env.yaml configures the controller its checkpoint recorded.

  Raises:
    ValueError: If the env.yaml has no controller or another one (a checkpoint
      moved there from another run).
  """
  fsm = saved.get("osl_fsm")
  if not isinstance(fsm, Mapping):
    raise ValueError("The run's params/env.yaml has no osl_fsm: not a stage-1.5 run.")
  try:
    rules = rule_set(str(fsm.get("rules")), fsm.get("swing_load_exit"))
  except ValueError as error:
    raise ValueError(f"The run's params/env.yaml osl_fsm: {error}") from error
  term = (saved.get("actions") or {}).get(OSL_ACTION_TERM) or {}
  configured = {
    "rules": rules.name,
    "swing_load_exit": rules.swing_load_exit,
    "body_mass": fsm.get("body_mass"),
    "initial_state": fsm.get("initial_state"),
    "controller": term.get("controller"),
  }
  differences = {
    key: (value, settings.get(key))
    for key, value in configured.items()
    if value != settings.get(key)
  }
  if differences:
    lines = "\n".join(
      f"  {key}: env.yaml {ours!r}, checkpoint {theirs!r}"
      for key, (ours, theirs) in differences.items()
    )
    raise ValueError(
      "The checkpoint records another OSL controller than its run's "
      f"params/env.yaml configures:\n{lines}\nWas the checkpoint moved into another "
      "run's directory?"
    )


def check_clean_group(group: ObservationGroupCfg) -> None:
  """The ``osl`` group reads its sensors exactly and at once, as the controller
  does: no bus lag, no randomization, no noise.

  Raises:
    ValueError: Naming each term that does otherwise.
  """
  issues: list[str] = []
  for name, term in group.terms.items():
    if term is None:
      continue
    if int(term.delay_min_lag) or int(term.delay_max_lag):
      issues.append(f"{name} lags {term.delay_min_lag}-{term.delay_max_lag} steps")
    if term.params.get("randomize", False):
      issues.append(f"{name} randomizes")
    if term.noise is not None:
      issues.append(f"{name} adds noise")
  if issues:
    raise ValueError(
      f"The run's osl group is not clean ({'; '.join(issues)}). The walking "
      "controller reads its sensors exactly and at once, and the bundle's "
      "observation contract has to say what it read."
    )


def fsm_config(
  spec: FsmRunSpec,
  saved: dict[str, Any],
  settings: Mapping[str, Any],
  *,
  touch_sites: TouchSites,
) -> OslFsmEnvCfg:
  """The run's stage-1.5 env config, rebuilt from this checkout in train mode with
  its controller ``settings`` and OSL foot ``touch_sites``, ready to compile on CPU.

  The touch sites, the human's observation, the actions, the simulation and the
  actuator delays must match the run's (:func:`~mjlab.tasks.osl_gait_library.
  export.check_actor_config`), and so must its env.yaml's controller
  (:func:`check_saved_controller`). The ``osl`` group takes the run's term settings
  (:func:`~mjlab.tasks.osl_split.export.apply_osl_group`) and must be clean
  (:func:`check_clean_group`).

  The ``osl_motors`` term is then switched to ``controller="policy"``: the CPU
  conformance sets the previous OSL command itself (:meth:`~mjlab.tasks.osl_split.
  export.CpuOslEnv.set_state`), which the controller would ignore to read the
  scene's sensors, which no simulation fills here. The term's scale, clip and
  targets are the same either way.

  Raises:
    ValueError: On drift, or an ``osl`` group outside the schema or not clean.
  """
  cfg = osl_fsm_env_cfg(spec.model, spec.socket, spec.reward, "train")
  configure_fsm(cfg, settings)
  set_touch_sites(cfg, touch_sites)
  gl.check_actor_config(cfg, saved)
  check_saved_controller(saved, settings)
  spec_fn = str(saved["scene"]["entities"]["robot"].get("spec_fn"))
  if spec_fn != split.OSL_SPEC_FN:
    raise ValueError(
      f"The run's robot spec_fn is {spec_fn!r}, not {split.OSL_SPEC_FN!r}."
    )
  group = cfg.observations[OSL_OBS_GROUP]
  split.apply_osl_group(group, saved["observations"][OSL_OBS_GROUP])
  check_clean_group(group)
  term = cfg.actions[OSL_ACTION_TERM]
  if not isinstance(term, OslFsmActionCfg):
    raise TypeError(f"The rebuilt {OSL_ACTION_TERM!r} is a {type(term).__name__}.")
  cfg.actions = {**cfg.actions, OSL_ACTION_TERM: replace(term, controller="policy")}
  return cfg


def check_rebuilt_controller(term: BaseAction, settings: Mapping[str, Any]) -> None:
  """The controller rebuilt from this checkout and the bundle's model is the one
  the run recorded, setting for setting (:data:`FSM_SETTINGS`).

  Raises:
    TypeError: If ``term`` is not the controller's action term.
    ValueError: Naming the settings that differ.
  """
  if not isinstance(term, OslFsmAction):
    raise TypeError(f"The rebuilt {OSL_ACTION_TERM!r} is a {type(term).__name__}.")
  rebuilt = term.settings()
  plain = split._plain
  differ = [key for key in FSM_SETTINGS if plain(rebuilt[key]) != plain(settings[key])]
  if differ:
    raise ValueError(
      "The OSL controller rebuilt from this checkout differs from the one the run "
      f"recorded in {differ}: its gains, thresholds or model changed since the run "
      "trained, so the bundle could not describe the controller the human learned "
      "beside."
    )


@dataclass(frozen=True)
class FsmRun:
  """A stage-1.5 run rebuilt on CPU: what its bundle is written from."""

  checkpoint: Path
  spec: FsmRunSpec
  agent: dict[str, Any]
  payload: Mapping[str, Any]
  settings: dict[str, Any]
  """The controller the run recorded (``infos["osl_fsm"]``)."""
  stage: dict[str, Any]
  """The checkpoint's ``infos["split"]``: the stage start, lineage and curriculum."""
  lineage: dict[str, Any] | None
  """The warm start's lineage, plain types."""
  touch_sites: TouchSites
  task: SimpleNamespace
  """What stage 2's section builders read: ``cfg``, ``model``, ``joint_names``,
  ``command``, ``actions`` (the OSL term under ``controller="policy"``),
  ``layout`` and ``step_dt``."""
  solver: dict[str, Any]
  """The compiled model's solver caps and the run's (``deployment``/``training``)."""


def build_run(
  checkpoint: str | Path,
  library: str | Path | None = None,
  solver_iterations: int | None = None,
  ls_solver_iterations: int | None = None,
) -> FsmRun:
  """Rebuild a stage-1.5 run's task on CPU, without a Warp simulation.

  Args:
    checkpoint: ``model_<N>.pt`` of a stage-1.5 run (its run directory holds
      ``params/``).
    library: The clip folder, when the run's is not on this machine.
    solver_iterations, ls_solver_iterations: The compiled model's solver caps;
      default the run's, at least 10 and 20.

  Raises:
    ValueError: If the run or checkpoint is not a stage-1.5 one, the warm start's
      lineage disagrees (model, socket, touch sites, action terms), or the task or
      controller rebuilt from this checkout differs from the run's.
  """
  checkpoint = Path(checkpoint).resolve()
  params = checkpoint.parent / "params"
  agent = gl.read_config(params / "agent.yaml")
  saved = gl.read_config(params / "env.yaml")
  spec = fsm_run_spec(checkpoint, agent)
  payload = load_payload(checkpoint)
  settings = stored_settings(checkpoint, payload)
  stage = dict((payload.get("infos") or {}).get("split") or {})
  lineage = split._plain(stage.get("lineage"))
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

  cfg = fsm_config(spec, saved, settings, touch_sites=touch.value)
  env, command, actions, solver = gl.compile_task(
    cfg, saved, library, solver_iterations, ls_solver_iterations
  )
  check_rebuilt_controller(actions[OSL_ACTION_TERM], settings)
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
  task = SimpleNamespace(
    cfg=cfg,
    model=env.sim.mj_model,
    joint_names=list(env.scene["robot"].joint_names),
    command=command,
    actions=actions,
    layout=layout,
    step_dt=float(env.step_dt),
  )
  return FsmRun(
    checkpoint=checkpoint,
    spec=spec,
    agent=agent,
    payload=payload,
    settings=settings,
    stage=stage,
    lineage=lineage,
    touch_sites=touch.value,
    task=task,
    solver=solver,
  )


##
# The OSL controller.
##


def osl_deploy_differences(settings: Mapping[str, Any]) -> list[str]:
  """The recorded settings in which the controller differs from osl_deploy's
  hard-coded one: its impedance table, thresholds and the rules' own (compared with
  :mod:`~mjlab.tasks.osl_fsm.fsm`'s, which were ported from osl_deploy's
  ``osl_fsm``)."""
  rules = RULES.get(str(settings["rules"]))
  expected = {
    "impedance": impedance_settings(),
    "thresholds": threshold_settings(),
    "every": None if rules is None else rules.every,
    "knee_velocity": None if rules is None else rules.knee_velocity,
  }
  return [
    key for key, value in expected.items() if split._plain(settings[key]) != value
  ]


def deploy_options(
  settings: Mapping[str, Any], model: mujoco.MjModel
) -> dict[str, Any]:
  """The options of osl_deploy's state machine (its ``fsm_policy`` keywords, the
  flags of :func:`deploy_command`) that reproduce the run's controller on the
  bundle's ``model``.

  The rules always, since osl_deploy defaults to the sdk rules; the sdk rules' swing
  exit always, explicitly; the body mass when the run set one, or when the model's
  own mass (osl_deploy's default) is not the run's.
  """
  rules = rule_set(settings["rules"], settings["swing_load_exit"])
  options: dict[str, Any] = {"rules": rules.name}
  if rules.name == "sdk":
    options["swing_load_exit"] = rules.swing_load_exit
  mass = float(settings["body_mass_kg"])
  if settings["body_mass"] is not None or float(model.body_mass.sum()) != mass:
    options["body_mass"] = mass
  return options


def deploy_command(options: Mapping[str, Any]) -> str:
  """The osl_deploy command line that runs the bundle (``BUNDLE``) beside the
  controller of :func:`deploy_options`. osl_deploy's ``osl_fsm`` defaults the OSL's
  sensor lag and motor delay to zero, as training ran them."""
  words = ["python", "-m", DEPLOY_MODULE, "BUNDLE", "--rules", str(options["rules"])]
  if "swing_load_exit" in options:
    words.append(
      "--swing-load-exit" if options["swing_load_exit"] else "--no-swing-load-exit"
    )
  if "body_mass" in options:
    words += ["--body-mass", repr(float(options["body_mass"]))]
  return " ".join(words)


def fsm_block(settings: Mapping[str, Any], model: mujoco.MjModel) -> dict[str, Any]:
  """``controllers.osl.fsm``: the controller as the run recorded it, the channels it
  reads, the osl_deploy command line and options that reproduce it, and whether its
  gains and thresholds are osl_deploy's own (osl_deploy ignores the block today)."""
  options = deploy_options(settings, model)
  return {
    "format": FSM_FORMAT,
    **{key: split._plain(settings[key]) for key in FSM_SETTINGS},
    "signals": dict(SIGNALS),
    "deploy_command": deploy_command(options),
    "deploy_options": options,
    "matches_osl_deploy_constants": not osl_deploy_differences(settings),
  }


def fsm_controller(
  c: SimpleNamespace, settings: Mapping[str, Any], lineage: Any
) -> dict[str, Any]:
  """The OSL controller's section: stage 2's (:func:`~mjlab.tasks.osl_split.
  export.osl_controller`) without ``onnx`` and ``io``, the ``osl_split_sensors_v1``
  observation built by mjlab's own observation manager over the bundle's model,
  with no normalization; a std of 0 in the motor block; the clean group's
  randomization (none) under ``training``; the motors' torque map; the state
  machine as ``memory``; and the :func:`fsm_block`."""
  actions = {OSL_ACTION_TERM: c.actions[OSL_ACTION_TERM]}
  parts = split._parts(c, actions, actor_group=False)
  std = np.zeros(c.layout.dim(OSL))
  env = split.CpuOslEnv(c.cfg, c.model, c.actions, c.step_dt)
  mujoco.mj_forward(c.model, env.data)
  manager = split.observation_manager(env, c.cfg.observations[OSL_OBS_GROUP])
  env.observation_manager = manager
  dim = int(manager.group_obs_dim[OSL_OBS_GROUP][0])
  alg = SimpleNamespace(
    osl_actor=SimpleNamespace(obs_groups=[OSL_OBS_GROUP], obs_dim=dim),
    osl_std=torch.as_tensor(std),
    layout=c.layout,
  )
  device = osl_policy_metadata(env, alg, lineage)  # type: ignore[arg-type]
  obs = device["obs"]
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
  training = training_metadata(parts)
  training["observation_randomization"] = device["randomization"]
  return {
    "observation": {
      "schema": split.OSL_OBS_SCHEMA,
      "group": OSL_OBS_GROUP,
      "size": obs["dim"],
      "terms": [t["name"] for t in obs["terms"]],
      "term_sizes": {t["name"]: t["dim"] for t in obs["terms"]},
      "layout": obs["layout"],
      "index": obs["index"],
      "normalization": NORMALIZATION,
      "term_layout": obs["terms"],
      "latency": {
        "unit": "control steps",
        "buses": buses,
        "default": "validation_steps",
        "note": split.LATENCY_NOTE,
      },
      "previous_command": split.PREVIOUS_COMMAND,
      "reset": device["reset"],
      "sensors": device["sensors"],
      "format": device["format"],
    },
    "action": action_metadata(parts, std, "mean", complete=False),
    "training": training,
    "device": device["action"],
    "memory": {
      "kind": "state_machine",
      "states": list(STATES),
      "reset": (
        "every episode starts in early stance (osl_deploy's convention; the run "
        f"trained with initial_state {settings['initial_state']!r})"
      ),
    },
    "fsm": fsm_block(settings, c.model),
  }


##
# Conformance.
##


def conformance_clip(command: Any) -> int:
  """The clip stage 2's conformance plays: the one nearest its turning walk
  (:data:`~mjlab.tasks.osl_split.export.CONFORMANCE_TWIST`)."""
  twists = command.twists.cpu().double().numpy()
  target = np.asarray(split.CONFORMANCE_TWIST)
  return int(np.argmin(np.linalg.norm(twists - target, axis=1)))


def looped_clip(command: Any, clip: int, steps: int) -> SimpleNamespace:
  """``clip`` of the library ``command`` played for ``steps`` frames, as the
  one-clip command :func:`~mjlab.tasks.osl_split.export.osl_conformance` reads.

  Past its last frame the clip starts over, each loop placed where the previous one
  ended by the clip's stride transform, as the training command unrolls it: the
  root's pose and velocities turn and shift with the loop, the joints repeat. The
  first loop is the clip's own frames.
  """
  frames = int(command.num_frames[clip])
  stride = command.strides[clip].detach().cpu().double()
  anchor = torch.zeros(3, dtype=torch.float64)
  loops: dict[str, list[torch.Tensor]] = {key: [] for key in CLIP_FIELDS}
  for start in range(0, steps, frames):
    count = min(frames, steps - start)
    loop = {
      key: command.data[key][clip, :count].detach().cpu().double()
      for key in CLIP_FIELDS
    }
    if start:
      anchor = se2_compose(anchor, stride)
      loop = _placed(loop, anchor)
    for key in CLIP_FIELDS:
      loops[key].append(loop[key])
  return SimpleNamespace(
    data={key: torch.cat(loops[key])[None] for key in CLIP_FIELDS},
    num_frames=torch.tensor([steps]),
    twists=command.twists[clip : clip + 1].detach().cpu(),
  )


def _placed(
  loop: dict[str, torch.Tensor], anchor: torch.Tensor
) -> dict[str, torch.Tensor]:
  """A loop of clip frames moved by the planar pose ``anchor`` (x, y, yaw)."""
  cos, sin = torch.cos(anchor[2]), torch.sin(anchor[2])

  def turn(v: torch.Tensor) -> torch.Tensor:
    x, y = v[:, 0], v[:, 1]
    return torch.stack((cos * x - sin * y, sin * x + cos * y, v[:, 2]), dim=-1)

  count = loop["root_pos"].shape[0]
  half, zero = anchor[2] / 2, torch.zeros_like(anchor[2])
  yaw = torch.stack((torch.cos(half), zero, zero, torch.sin(half)))
  shift = torch.stack((anchor[0], anchor[1], zero))
  return {
    "root_pos": turn(loop["root_pos"]) + shift,
    "root_quat": quat_mul(yaw.expand(count, 4), loop["root_quat"]),
    "root_lin_vel": turn(loop["root_lin_vel"]),
    "root_ang_vel_w": turn(loop["root_ang_vel_w"]),
    "joint_pos": loop["joint_pos"],
    "joint_vel": loop["joint_vel"],
  }


def newest_channels(observation: Mapping[str, Any]) -> dict[str, int]:
  """``{channel: index of its newest frame}`` over a sensor observation's terms
  (its manifest section's ``term_layout``), as osl_deploy's state machine places
  its signals."""
  index: dict[str, int] = {}
  for term in observation["term_layout"]:
    width, frames = int(term["frame_dim"]), int(term["history_length"])
    newest = int(term["offset"]) + (frames - 1) * width
    index.update({name: newest + c for c, name in enumerate(term["channels"])})
  return index


def fsm_conformance(
  observations: np.ndarray, section: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, np.ndarray]:
  """The walking controller over a sequence of zero-lag ``osl`` observations, from
  early stance, with the action term's own arithmetic.

  ``section`` is the OSL controller's manifest section and ``settings`` the run's
  controller. At each row the controller reads the newest frame of the
  :data:`~.mdp.SIGNALS` channels (the load is the leg's compression,
  ``-osl_ankle_load_force_y``: the frame is tared already), takes at most one
  transition and computes the new state's torque in float32, as
  :class:`~.mdp.OslFsmAction` does in training, then the command through the
  nominal motor map: ``ctrl = clip(torque / nm_per_ctrl, ctrl_limits)`` and
  ``action = (ctrl - offset) / scale``, in the controller's output order.

  Returns:
    ``state`` (S,) the state code after each update (an index into
    :data:`~.fsm.STATES`), ``torque`` (S, 2) the knee and ankle torque in N m
    before the motors' limits, ``action`` (S, A) the command.
  """
  index = newest_channels(section["observation"])
  columns = [index[SIGNALS[reading]] for reading in READINGS]
  rules = rule_set(settings["rules"], settings["swing_load_exit"])
  fsm = BatchedFsm(rules, float(settings["body_weight_n"]), 1)
  limits = settings["ctrl_limits"]

  def per_joint(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)

  nm_per_ctrl = per_joint([settings["nm_per_ctrl"][joint] for joint in JOINTS])
  low = per_joint([limits[joint][0] for joint in JOINTS])
  high = per_joint([limits[joint][1] for joint in JOINTS])
  action = section["action"]
  targets = [target for block in action["blocks"] for target in block["targets"]]
  order = _joint_of_target(targets)
  scale = per_joint(action["scale"])
  offset = per_joint(action["offset"])
  rows = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
  states: list[int] = []
  torques: list[torch.Tensor] = []
  commands: list[torch.Tensor] = []
  for row in rows:
    knee, knee_velocity, ankle, ankle_velocity, force = row[columns].reshape(-1, 1)
    state = fsm.update(-force, knee, knee_velocity, ankle)
    q = torch.stack((knee, ankle), dim=1)
    qd = torch.stack((knee_velocity, ankle_velocity), dim=1)
    torque = fsm.torque(q, qd)
    ctrl = torch.clamp(torque / nm_per_ctrl, low, high)
    states.append(int(state[0]))
    torques.append(torque[0])
    commands.append(((ctrl[:, order] - offset) / scale)[0])
  return {
    "state": np.asarray(states, dtype=np.int64),
    "torque": torch.stack(torques).numpy(),
    "action": torch.stack(commands).numpy(),
  }


##
# The bundle.
##


def _readme(manifest: dict[str, Any]) -> str:
  run, human, osl = (
    manifest["run"],
    manifest["controllers"][HUMAN],
    manifest["controllers"][OSL],
  )
  fsm = osl["fsm"]
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

  command = fsm["deploy_command"].replace("BUNDLE", "<this folder>")
  swing_exit = ""
  if fsm["rules"] == "sdk":
    swing_exit = f", load-based swing exit {'on' if fsm['swing_load_exit'] else 'off'}"
  mass = "the model's mass" if fsm["body_mass"] is None else "a set mass"
  own = "" if fsm["matches_osl_deploy_constants"] else " NOT"
  signals = ", ".join(f"`{channel}`" for channel in fsm["signals"].values())
  stage1 = lineage.get("stage1_checkpoint")
  return f"""# {run["task"]}: {run["run"]} / {run["checkpoint"]}

Stage-1.5 deployment bundle (schema {manifest["schema_version"]}, `{manifest["kind"]}`)
exported by mjlab (`mjlab.tasks.osl_fsm.export`) for osl_deploy (plain CPU MuJoCo
{manifest["mujoco_version"]}): a human actor trained beside the OSL's classical
walking controller. There is no OSL network: osl_deploy's state machine drives the
OSL. Run the bundle with

```
{command}
```

That is the controller the human learned beside: {fsm["rules"]} rules{swing_exit},
body weight {fsm["body_weight_n"]:.1f} N ({mass}), its sensors read at once and its
torque applied without delay (osl_fsm's defaults). osl_deploy's `osl_fsm` defaults
to `--rules sdk`, so the `--rules` flag is required: without it the human meets
another controller than the one it learned beside. Plain `osl-deploy` stops on this
bundle, which has no `osl_actor.onnx`.

- Model: {run["model"]}, socket: {run["socket"]}, reward: {run["reward"]}, library:
  {run["variant"]} ({len(manifest["reference"]["clip_names"])} clips)
{touch_sites_readme_line(manifest["touch_sites"])}
- Checkpoint: `{run["checkpoint"]}` (iteration {run["iteration"]}, stage-1.5
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

## OSL controller (the walking controller, no network)

- Four states ({", ".join(osl["memory"]["states"])}), each a spring-damper per
  joint; gains and thresholds are{own} osl_deploy's own (`controllers.osl.fsm`)
- Once per control step it reads the newest frame of {signals} (the load is the
  tared axial force, compression positive), takes at most one transition and
  commands the new state's torque through the motor block
  {", ".join(f"{b['name']} ({b['size']})" for b in osl["action"]["blocks"])} at the
  motors' nominal gains
- Training started each episode with initial state `{fsm["initial_state"]}`;
  osl_deploy starts every episode in early stance
- Actuator command delay in training: {delays(osl)}

`parity.npz` holds the OSL observation conformance sequence (`osl_conformance_*`:
CPU MuJoCo states and the observation mjlab's own terms build from them) and the
controller's (`osl_fsm_conformance_*`: its state, torque and action on each row of
`osl_conformance_obs_lag0`, from early stance; `validation.osl_fsm_conformance` says
how osl_deploy reproduces them).

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
  """Export ``checkpoint`` as a stage-1.5 bundle (module docstring); returns it.

  Args:
    checkpoint: ``model_<N>.pt`` of a stage-1.5 run.
    output: The bundle folder (default ``<run>/deploy/<checkpoint stem>``).
    library: The clip folder, when the run's is not on this machine.
    human_inference: The human's default inference; None picks ``expected`` for a
      median muscle std above 0.3 and ``mean`` otherwise.
    overwrite: Replace an existing bundle (once the new one is complete).
    solver_iterations, ls_solver_iterations: The bundle model's solver caps;
      default the run's, at least 10 and 20.
  """
  checkpoint = Path(checkpoint).resolve()
  output = Path(output or checkpoint.parent / "deploy" / checkpoint.stem).resolve()
  if output.exists() and not overwrite:
    raise FileExistsError(f"{output} exists; pass --overwrite True to replace it.")
  run = build_run(checkpoint, library, solver_iterations, ls_solver_iterations)
  spec, c, settings = run.spec, run.task, run.settings
  actor = gl.build_actor(
    run.agent["actor"], run.payload[HUMAN_STATE_KEY], "actor", c.layout.dim(HUMAN)
  )

  output.parent.mkdir(parents=True, exist_ok=True)
  tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
  umask = os.umask(0)
  os.umask(umask)
  tmp.chmod(0o777 & ~umask)  # mkdtemp's 0700 would hide the bundle from others.
  try:
    obs_dim = export_actor_onnx(actor, tmp / ONNX_FILE)
    sections = {
      HUMAN: split.human_controller(c, actor, obs_dim, human_inference),
      OSL: fsm_controller(c, settings, run.lineage),
    }
    split.check_coverage(c.model, sections)
    iteration = run.payload.get("iter")
    manifest: dict[str, Any] = {
      "schema_version": split.SCHEMA_VERSION,
      "kind": split.BUNDLE_KIND,
      "mujoco_version": mujoco.__version__,
      "decimation": int(c.cfg.decimation),
      "policy_dt": c.step_dt,
      "root_joint": "robot/root",
      "joint_names": [f"robot/{j}" for j in c.joint_names],
      "socket": spec.socket,
      "touch_sites": touch_sites_manifest(c.model),
      "run": {
        "task": spec.task_id,
        "stage": STAGE,
        "osl_controller": OSL_CONTROLLER,
        "model": spec.model,
        "socket": spec.socket,
        "reward": spec.reward,
        "variant": "unicycle",
        "experiment": spec.experiment,
        "run": spec.run,
        "checkpoint": checkpoint.name,
        "iteration": int(iteration) if iteration is not None else None,
        "stage_iteration": run.payload.get("stage_iteration"),
        "stage_start": run.stage.get("stage_start"),
      },
      "reference": {
        "clip_names": list(c.command.clip_names),
        "policy_dt": c.step_dt,
        "lookahead_steps": c.command.cfg.lookahead_steps,
        "lookahead_stride": c.command.cfg.lookahead_stride,
        "blend_time_s": c.command.cfg.blend_time_s,
        "twist_metric_weights": list(c.command.cfg.twist_metric_weights),
        "used_by": [HUMAN],
      },
      "deployment": {
        "solver": run.solver["deployment"],
        "training_solver": run.solver["training"],
        "note": (
          "Solver caps of model.mjb. Training ran mujoco_warp at training_solver, "
          "where its solves converge; CPU MuJoCo needs a higher cap to converge "
          "(it truncates the contact solve at 4 iterations under the elliptic cone "
          "at impratio 3)."
        ),
      },
      "composition": {
        "controllers": list(split.CONTROLLERS),
        "env_action_terms": c.layout.to_list(),
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
        "osl": (
          "The walking controller (controllers.osl.fsm), no network: osl_deploy's "
          "state machine, python -m osl_deploy.osl_fsm."
        ),
      },
      "controllers": sections,
      "lineage": run.lineage,
    }
    if manifest["touch_sites"]["version"] != run.touch_sites:
      raise RuntimeError(
        f"model.mjb carries the {manifest['touch_sites']['version']!r} touch sites, "
        f"but the run trained with {run.touch_sites!r}."
      )
    if not sections[OSL]["fsm"]["matches_osl_deploy_constants"]:
      raise RuntimeError(  # check_rebuilt_controller compared them already.
        "The recorded controller differs from osl_deploy's in "
        f"{osl_deploy_differences(settings)}."
      )
    manifest["provenance"] = gl._provenance(
      checkpoint,
      Path(c.command.cfg.library_dir),
      int(iteration) if iteration is not None else None,
      SOURCE_FOLDERS,
    )
    mujoco.mj_saveModel(c.model, str(tmp / "model.mjb"))
    arrays: dict[str, Any] = {
      key: c.command.data[key].cpu().numpy() for key in gl.REFERENCE_FIELDS
    }
    arrays["num_frames"] = c.command.num_frames.cpu().numpy()
    arrays["twists"] = c.command.twists.cpu().numpy()
    arrays["strides"] = c.command.strides.cpu().numpy()
    np.savez_compressed(tmp / "reference.npz", **arrays)

    rtol, atol = gl.PARITY_TOLERANCE
    observations = normalizer_samples(actor, gl.PARITY_SAMPLES)
    means, error = onnx_parity(tmp / ONNX_FILE, actor, observations)
    module = cast(Any, actor).as_onnx(verbose=False)
    with torch.inference_mode():
      reference = module(torch.from_numpy(observations))
    np.testing.assert_allclose(means, reference.numpy(), rtol=rtol, atol=atol)
    parity: dict[str, Any] = {
      f"{HUMAN}_observations": observations,
      f"{HUMAN}_means": means,
    }
    clip = conformance_clip(c.command)
    conformance = split.osl_conformance(
      c.cfg,
      c.model,
      c.actions,
      c.step_dt,
      looped_clip(c.command, clip, CONFORMANCE_STEPS),
      c.joint_names,
      steps=CONFORMANCE_STEPS,
    )
    conformance["clip"] = np.asarray(clip)
    parity.update({f"osl_conformance_{k}": v for k, v in conformance.items()})
    controller = fsm_conformance(conformance["obs_lag0"], sections[OSL], settings)
    parity.update({f"osl_fsm_conformance_{k}": v for k, v in controller.items()})
    np.savez_compressed(tmp / "parity.npz", **parity)
    options = sections[OSL]["fsm"]["deploy_options"]
    keywords = ", ".join(f"{key}={value!r}" for key, value in options.items())
    manifest["validation"] = {
      "human_onnx_max_abs_error": error,
      "parity_inputs": (
        "human_observations: the human actor's observation mean, then mean + std "
        "* z (N(0, 1)); human_means: their ONNX outputs"
      ),
      "osl_conformance": {
        "keys": sorted(f"osl_conformance_{k}" for k in conformance),
        "steps": int(conformance["qpos"].shape[0]),
        "lags": conformance["lags"].tolist(),
        "clip": clip,
        "clip_frames": int(c.command.num_frames[clip]),
        "procedure": (
          "The states play the clip in a loop, each loop placed by the clip's "
          "stride transform. For each step t: mj_resetData; set qpos, qvel, act "
          "and ctrl from osl_conformance_*[t]; mj_forward; the previous OSL "
          "command is osl_conformance_last_action[t] (0 at t = 0, a reset). Build "
          "the osl frame, push it through the latency and history buffers with a "
          "reset at t = 0, and compare with osl_conformance_obs_lag<L>[t] (float32 "
          "terms: about 1e-5 relative). osl_conformance_frames[t] is the "
          "undelayed frame."
        ),
      },
      "osl_fsm_conformance": {
        "keys": sorted(f"osl_fsm_conformance_{k}" for k in controller),
        "steps": int(controller["state"].shape[0]),
        "input": "osl_conformance_obs_lag0",
        "rules": settings["rules"],
        "swing_load_exit": settings["swing_load_exit"],
        "body_weight_n": settings["body_weight_n"],
        "initial_state": DEPLOY_INITIAL_STATE,
        "states": list(STATES),
        "procedure": (
          "Reset the controller to e_stance, then one update per row of "
          "osl_conformance_obs_lag0: read the newest frame of fsm.signals (load = "
          "-osl_ankle_load_force_y: the frame is tared already), take at most one "
          "transition, compute the new state's torque "
          "(osl_fsm_conformance_torque, N m, knee then ankle, before the motors' "
          "limits) and the action commanding it (osl_fsm_conformance_action, in "
          "controllers.osl.action's output order: ctrl = clip(torque / "
          "nm_per_ctrl, ctrl_limits), action = (ctrl - offset) / scale). "
          "osl_fsm_conformance_state is the state after each update (an index "
          "into states)."
        ),
        "osl_deploy": (
          f"policy = osl_fsm.fsm_policy(bundle, bundle.controller('osl'), "
          f"{keywords}); policy.reset(); then policy(row) for each row: "
          "policy.machine.state is states[osl_fsm_conformance_state[t]] and the "
          "action it returns osl_fsm_conformance_action[t], to float32 precision "
          "(mjlab computes the torque in float32, osl_deploy in float64)."
        ),
      },
      "physics_transfer_verified": False,
    }
    manifest["files"] = {name: gl.sha256(tmp / name) for name in CHECKED_FILES}
    header = {k: manifest[k] for k in ("schema_version", "kind", "run")}
    attach_metadata(
      tmp / ONNX_FILE, split._plain({**header, "controller": HUMAN, **sections[HUMAN]})
    )
    manifest["files"][ONNX_FILE] = gl.sha256(tmp / ONNX_FILE)
    (tmp / "manifest.json").write_text(
      json.dumps(split._plain(manifest), indent=2) + "\n"
    )
    (tmp / "README.md").write_text(_readme(split._plain(manifest)))
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
class FsmExportConfig:
  checkpoint: tyro.conf.Positional[Path]
  """``model_<N>.pt`` of a stage-1.5 run (its run directory holds ``params/``)."""
  output: Path | None = None
  """Bundle folder; default ``<run>/deploy/<checkpoint stem>``."""
  library: Path | None = None
  """Clip folder, when the run's ``library_dir`` is not on this machine."""
  human_inference: Literal["auto", "mean", "expected"] = "auto"
  """The human's default inference: ``auto`` picks ``expected`` for a median
  muscle std above 0.3 and ``mean`` otherwise. The OSL runs the walking
  controller."""
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
  cfg = tyro.cli(FsmExportConfig, config=mjlab.TYRO_FLAGS)
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
  human, fsm = manifest["controllers"][HUMAN], manifest["controllers"][OSL]["fsm"]
  print(
    f"[INFO] Bundle: {bundle}\n"
    f"[INFO] {manifest['run']['task']}: human {human['action']['size']} outputs "
    f"({human['action']['default_inference']}; ONNX max abs error "
    f"{manifest['validation']['human_onnx_max_abs_error']:.3g}), OSL the walking "
    f"controller ({fsm['rules']} rules)\n"
    f"[INFO] Deploy: {fsm['deploy_command'].replace('BUNDLE', str(bundle))}"
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
