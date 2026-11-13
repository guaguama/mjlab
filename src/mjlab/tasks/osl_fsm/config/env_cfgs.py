"""The stage-1.5 environment: stage 2's, with the OSL under the walking controller.

Everything not listed here is ``osl_split_env_cfg(model, socket, reward, mode,
dr_cfg)``: stage 1's unicycle task (simulation, library command, terminations,
twist reward, physics randomization with the human's 20-40 ms delay and the pushes),
the OSL sensors, the human's ``actor`` observation, the split reward ``mimic``, whose
human channel the human trains on, and the leg-contact sensor. The changes:

* ``osl_motors`` becomes :class:`~..mdp.OslFsmActionCfg` with the stage-1 term's
  fields: the classical controller drives the OSL from the prosthesis's own sensors,
  and the policy's two OSL rows are ignored;
* the OSL motors' actuator delay is zero (``OslDrCfg(osl_delay_s=(0, 0))`` unless
  ``dr_cfg`` says otherwise): osl_deploy runs the controller at zero delay, and its
  stock gains go unstable with a single 10 ms step of it (the late-stance ankle);
* the ``osl`` group is clean and immediate in every mode (:func:`clean_osl_group`).
  No network reads it, but it is the observation contract osl_deploy's controller
  reads from a bundle, which the exporter writes, and its newest frame is exactly
  the controller's reading;
* one metric per controller diagnostic (:data:`~..mdp.FSM_DIAGNOSTICS`).

The controller's settings are the config's ``osl_fsm`` (``--env.osl-fsm.rules``,
``--env.osl-fsm.swing-load-exit``, ``--env.osl-fsm.body-mass``,
``--env.osl-fsm.initial-state``), copied into the action term whenever the config is
constructed, which tyro does after applying the command line. Code that changes
``osl_fsm`` on a built config calls :func:`sync_osl_fsm` (or :func:`configure_fsm`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, get_args

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.osl_fsm.fsm import RuleName, rule_set
from mjlab.tasks.osl_fsm.mdp import (
  FSM_DIAGNOSTICS,
  InitialState,
  OslFsmAction,
  OslFsmActionCfg,
  fsm_component,
)
from mjlab.tasks.osl_gait_library.config.dr import OslDrCfg
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward
from mjlab.tasks.osl_mimic.touch_sites import OslEnvCfg
from mjlab.tasks.osl_split.config.env_cfgs import osl_split_env_cfg
from mjlab.tasks.osl_split.config.sensors import ANKLE_FORCE
from mjlab.tasks.osl_split.mdp import (
  OslObsCfg,
  load_cell_sensor_names,
  osl_observation_group,
  osl_validation_group,
)
from mjlab.tasks.osl_split.rl.layout import OSL_ACTION_TERM

OSL_OBS_GROUP = "osl"
FSM_FLAGS: dict[str, str] = {
  "rules": "--env.osl-fsm.rules",
  "swing_load_exit": "--env.osl-fsm.swing-load-exit",
  "body_mass": "--env.osl-fsm.body-mass",
  "initial_state": "--env.osl-fsm.initial-state",
}
"""The controller settings a checkpoint records and a run must agree on (keys of
:meth:`~..mdp.OslFsmAction.settings`), each with the flag that sets it."""


@dataclass
class OslFsmCfg:
  """The OSL walking controller's settings (``--env.osl-fsm.*``)."""

  rules: RuleName = "myosuite"
  """``"myosuite"``: MyoSuite's MyoOSLController, where a state leaves as soon as any
  of its conditions holds and early swing ends below a knee velocity of 3 deg/s.
  ``"sdk"``: the opensourceleg SDK example's, where early stance and early swing
  need both of their conditions and early swing compares with 3 rad/s."""
  swing_load_exit: bool | None = None
  """The sdk rules only: early swing returns to early stance once the leg carries
  0.4 body weights again. None is their default, on; False is the example
  unchanged. The myosuite rules refuse True."""
  body_mass: float | None = None
  """kg, scaling the load thresholds. None: the robot's nominal mass (82.289 kg on
  the musculoskeletal model), osl_deploy's default."""
  initial_state: InitialState = "pose"
  """``"pose"``: on the first control step of an episode, the state is guessed from
  the start pose, since training starts at random clip frames; ``"e_stance"``:
  every episode starts in early stance, as in osl_deploy."""


@dataclass(kw_only=True)
class OslFsmEnvCfg(OslEnvCfg):
  """A stage-1.5 env config: its ``osl_motors`` term runs the controller that
  ``osl_fsm`` configures (and its robot builds the touch sites ``osl_touch_sites``
  names, as every OSL task's)."""

  osl_fsm: OslFsmCfg = field(default_factory=OslFsmCfg)

  def __post_init__(self) -> None:
    super().__post_init__()
    sync_osl_fsm(self)


def check_osl_fsm(fsm: OslFsmCfg) -> None:
  """Refuse settings the controller cannot run, naming the flags.

  Raises:
    ValueError: For unknown rules, the swing exit asked of the myosuite rules, a
      body mass that is not positive and finite, or an unknown initial state.
  """
  try:
    rule_set(fsm.rules, fsm.swing_load_exit)
  except ValueError as error:
    raise ValueError(
      f"{error} ({FSM_FLAGS['rules']} {fsm.rules}, {FSM_FLAGS['swing_load_exit']} "
      f"{fsm.swing_load_exit})"
    ) from error
  mass = fsm.body_mass
  if mass is not None and not (math.isfinite(mass) and mass > 0.0):
    raise ValueError(f"{FSM_FLAGS['body_mass']} must be positive and finite: {mass}")
  if fsm.initial_state not in get_args(InitialState):
    raise ValueError(
      f"Unknown {FSM_FLAGS['initial_state']} {fsm.initial_state!r}; choose from "
      f"{list(get_args(InitialState))}"
    )


def sync_osl_fsm(cfg: OslFsmEnvCfg) -> None:
  """Give the ``osl_motors`` term the config's ``osl_fsm`` settings.

  The config gets a new actions dict rather than an edited term: the field-copying
  constructions share the dict between configs.

  Raises:
    ValueError: For settings :func:`check_osl_fsm` refuses, or an ``osl`` group
      without the ankle load cell, which the controller reads (osl_deploy's refuses
      a bundle whose observation lacks its axial force).
    TypeError: If ``osl_motors`` is not an :class:`~..mdp.OslFsmActionCfg`.
  """
  fsm = cfg.osl_fsm
  check_osl_fsm(fsm)
  term = cfg.actions.get(OSL_ACTION_TERM)
  if not isinstance(term, OslFsmActionCfg):
    raise TypeError(
      f"A stage-1.5 env config needs an OslFsmActionCfg {OSL_ACTION_TERM!r} action "
      f"term (see osl_fsm_env_cfg); got {type(term).__name__}."
    )
  group = cfg.observations.get(OSL_OBS_GROUP)
  loadcell = None if group is None else group.terms.get("loadcell")
  if loadcell is not None and ANKLE_FORCE not in load_cell_sensor_names(
    loadcell.params
  ):
    raise ValueError(
      "The osl group's loadcell term leaves out the ankle load cell, whose axial "
      "force the OSL controller reads: keep "
      "--env.observations.osl.terms.loadcell.params.ankle True."
    )
  synced = replace(
    term,
    rules=fsm.rules,
    swing_load_exit=fsm.swing_load_exit,
    body_mass=fsm.body_mass,
    initial_state=fsm.initial_state,
  )
  cfg.actions = {**cfg.actions, OSL_ACTION_TERM: synced}


def clean_osl_group() -> ObservationGroupCfg:
  """The ``osl`` group read exactly and immediately: stage 2's group, 25 frames of
  every term, with no perception randomization and both buses at zero lag. Its
  newest frame is the reading the controller takes in the next ``process_actions``
  (the load cell tared alike), which osl_deploy's controller takes from the same
  channels."""
  cfg = OslObsCfg(delay_steps=(0, 0), validation_delay_steps=0)
  return osl_validation_group(osl_observation_group("train", cfg), 0)


def osl_fsm_env_cfg(
  model: Model = "msk",
  socket: Socket = "compliant",
  reward: Reward = "clf",
  mode: Mode = "train",
  dr_cfg: OslDrCfg | None = None,
  fsm: OslFsmCfg | None = None,
) -> OslFsmEnvCfg:
  """Create a stage-1.5 environment over the unicycle library.

  Args:
    model: ``"robot"`` or ``"msk"``, as stage 1.
    socket: ``"rigid"`` or ``"compliant"``, as stage 1.
    reward: ``"mimic"`` or ``"clf"``, as stage 2.
    mode: ``"train"``, ``"validation"`` or ``"play"``, as stage 2. The ``osl``
      group is clean in every mode.
    dr_cfg: Stage 1's physics randomization ranges (default: ``OslDrCfg`` with the
      OSL motors' delay at zero).
    fsm: The controller's settings (default ``OslFsmCfg()``; copied).
  """
  if dr_cfg is None:
    dr_cfg = OslDrCfg(osl_delay_s=(0.0, 0.0))
  fsm = OslFsmCfg() if fsm is None else replace(fsm)
  cfg = osl_split_env_cfg(model, socket, reward, mode, dr_cfg=dr_cfg)
  stage1 = cfg.actions[OSL_ACTION_TERM]
  if not isinstance(stage1, JointEffortActionCfg):
    raise TypeError(f"Stage 2's {OSL_ACTION_TERM!r} is a {type(stage1).__name__}.")
  # In place: the term keeps its slot in the action order.
  cfg.actions[OSL_ACTION_TERM] = OslFsmActionCfg(
    **{f.name: getattr(stage1, f.name) for f in fields(stage1)},
    rules=fsm.rules,
    swing_load_exit=fsm.swing_load_exit,
    body_mass=fsm.body_mass,
    initial_state=fsm.initial_state,
  )
  cfg.observations[OSL_OBS_GROUP] = clean_osl_group()
  for name in FSM_DIAGNOSTICS:
    cfg.metrics[name] = MetricsTermCfg(
      func=fsm_component, params={"name": name}, reduce="mean"
    )
  return OslFsmEnvCfg(
    **{f.name: getattr(cfg, f.name) for f in fields(cfg)}, osl_fsm=fsm
  )


def fsm_settings(env: Any) -> dict[str, Any]:
  """The controller settings a built stage-1.5 env (wrapped or not) resolved:
  :meth:`~..mdp.OslFsmAction.settings`, which the runner stores in checkpoints.

  Raises:
    TypeError: If the env's ``osl_motors`` term is not the controller's.
  """
  unwrapped = getattr(env, "unwrapped", env)
  term = unwrapped.action_manager.get_term(OSL_ACTION_TERM)
  if not isinstance(term, OslFsmAction):
    raise TypeError(
      f"The env's {OSL_ACTION_TERM!r} term is a {type(term).__name__}, not the OSL "
      "controller's: not a stage-1.5 env."
    )
  return term.settings()


def configure_fsm(
  cfg: ManagerBasedRlEnvCfg, settings: Mapping[str, Any] | OslFsmCfg
) -> OslFsmCfg:
  """Set a stage-1.5 env config's controller, action term included, in place: from
  an :class:`OslFsmCfg` or a checkpoint's stored settings (:func:`fsm_settings`;
  their :data:`FSM_FLAGS` keys are read). Returns the ``osl_fsm`` it set.

  Raises:
    TypeError: If ``cfg`` is not an :class:`OslFsmEnvCfg`.
    ValueError: For stored settings missing a key, or settings
      :func:`sync_osl_fsm` refuses.
  """
  if not isinstance(cfg, OslFsmEnvCfg):
    raise TypeError(f"Not a stage-1.5 env config: {type(cfg).__name__}.")
  if isinstance(settings, OslFsmCfg):
    fsm = replace(settings)
  else:
    missing = [key for key in FSM_FLAGS if key not in settings]
    if missing:
      raise ValueError(f"The stored OSL controller settings lack {missing}.")
    fsm = OslFsmCfg(**{key: settings[key] for key in FSM_FLAGS})
  check_osl_fsm(fsm)  # Before the assignment: a refusal leaves cfg as it was.
  cfg.osl_fsm = fsm
  sync_osl_fsm(cfg)
  return fsm
