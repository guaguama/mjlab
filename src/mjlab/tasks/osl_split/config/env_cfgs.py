"""The stage-2 split environment: stage 1's unicycle task plus the OSL's own sensors.

Everything not listed here is ``osl_gait_library_env_cfg(model, socket, reward, mode,
"unicycle")``: the simulation, the library command, the terminations, the twist
reward, the stage-1 physics randomization (actuator delays included) and the human's
observation, unchanged. The changes:

* the robot entity gains the OSL sensors (``config/sensors.py``), added on top of
  stage 1's foot-contact edit;
* the stage-1 ``critic`` group is dropped: it was a copy of ``actor``, which the human
  actor and both critics read;
* an ``osl`` group holds the OSL actor's observation, with perception randomization
  in training (:func:`~..mdp.osl_observation_group`). Its default sensor set is the
  real leg's (encoders, IMU, ankle load cell, heel and toe contact: 500 inputs);
  :func:`sync_osl_group` keeps it consistent whenever the config is constructed,
  which tyro does after applying the command line;
* the imitation reward becomes :class:`~..mdp.split_mimic_reward` under the same name,
  ``mimic``, so the stage-1 diagnostics keep their keys: it returns the same team
  reward, exposes the two agents' reward channels, and adds their diagnostics;
* the scene keeps stage 1's contact sensor between the OSL and the biological left
  leg (:func:`~.sensors.leg_contact_sensor_cfg`; added here to a stage-1 config
  without one), which the split reward reports as leg-contact diagnostics and, with
  ``regularizers.leg_contact`` above 0, charges to the channels
  ``regularizers.leg_contact_owner`` names: a sensor changes no physics.

The stage-1 perception switches (``OslDrCfg.encoder_bias`` and ``obs_noise``) edit
the ``actor`` group, which is now the human's observation, so they are refused.

One opt-in addition, off by default and then absent: the gait-symmetry penalty
(``--env.gait-symmetry.weight``, :class:`GaitSymmetryCfg`), an additive reward term
that both agents earn (:func:`sync_gait_symmetry`).

:func:`apply_split_stage` makes these changes to any stage-1 OSL config, so other
stage-2 tasks (stairs) chain it after their own stage-1 builder, and give their
config class :class:`OslSplitEnvCfg`'s ``__post_init__`` (or call
:func:`sync_osl_group` and :func:`sync_gait_symmetry`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.osl_gait_library.config.dr import OslDrCfg
from mjlab.tasks.osl_gait_library.config.env_cfgs import (
  apply_terrain,
  osl_gait_library_env_cfg,
)
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward
from mjlab.tasks.osl_mimic.leg_contact import add_leg_contact_sensor
from mjlab.tasks.osl_mimic.mdp import mimic_component
from mjlab.tasks.osl_mimic.touch_sites import CLI_FLAG as TOUCH_SITES_FLAG
from mjlab.tasks.osl_mimic.touch_sites import ENV_FIELD as TOUCH_SITES_FIELD
from mjlab.tasks.osl_mimic.touch_sites import OslEnvCfg
from mjlab.tasks.osl_split import mdp

from .sensors import with_osl_sensors

REWARD_NAME = "mimic"
"""The imitation reward's term name, kept from stage 1."""
PERCEPTION_DR_EVENT = "osl_encoder_bias"
"""Stage 1's encoder-bias event (``OslDrCfg.encoder_bias``)."""
PERCEPTION_DR_REFUSAL = (
  "OslDrCfg.encoder_bias and obs_noise corrupt the 'actor' group, which in stage 2 "
  "is the human's observation. The OSL's perception randomization is OslObsCfg's "
  "(osl_obs)."
)
HEEL_FLAG = "--env.observations.osl.terms.contact.params.heel"
GAIT_SYMMETRY_FIELD = "gait_symmetry"
"""The env config field of :class:`GaitSymmetryCfg`."""
GAIT_SYMMETRY_FLAG = "--env.gait-symmetry.weight"
MOTION_COMMAND = "motion"


@dataclass
class GaitSymmetryCfg:
  """The gait-symmetry penalty (``--env.gait-symmetry.weight``,
  ``--env.gait-symmetry.sigma``): each step, both agents earn ``-weight x (1 -
  exp(-s / sigma^2))``, where s is the asymmetry the policy adds between the OSL
  side and the intact side half a stride earlier (:mod:`~..mdp.symmetry`). It is
  bounded by the weight, so the step's reward stays positive at the tracking
  rewards' 0.5-0.8 and the twist term's 0.2.

  With ``weight`` 0, the default, there is no such term: the reward and metric
  terms are stage 2's alone."""

  weight: float = 0.0
  """The penalty's bound per step, ``w`` (>= 0); 0 leaves the term out."""
  sigma: float = mdp.SYMMETRY_SIGMA
  """Radians: the asymmetry scale. A step whose RMS asymmetry ``sqrt(s)`` is sigma
  pays 63% of the bound."""


@dataclass(kw_only=True)
class OslSplitEnvCfg(OslEnvCfg):
  """A stage-2 env config: its robot builds the touch sites ``osl_touch_sites``
  names (as every OSL task's), its ``osl`` group follows the sensor switches and
  the touch sites (:func:`sync_osl_group`), and its rewards carry the gait-symmetry
  penalty that ``gait_symmetry`` turns on (:func:`sync_gait_symmetry`)."""

  gait_symmetry: GaitSymmetryCfg = field(default_factory=GaitSymmetryCfg)

  def __post_init__(self) -> None:
    # tyro builds the config after applying the command line, so this sees the
    # final switches, touch sites and penalty.
    super().__post_init__()
    sync_osl_group(self)
    sync_gait_symmetry(self)


def sync_osl_group(cfg: Any) -> None:
  """Make a stage-2 config's ``osl`` group consistent, in place: the drop rule (a
  sensor term whose switches are all off is left out), then the checks.

  The config gets a new observations dict when a term is dropped, rather than an
  edited group: the field-copying constructions share it between configs. It reads
  only ``cfg.observations["osl"]`` and ``cfg.osl_touch_sites``.

  Raises:
    ValueError: For a config without an ``osl`` group, an unknown term, a group
      left without a sensor term, or heel contact under the ``original`` touch
      sites, whose ``r_foot`` zone never fires (naming both flags).
  """
  group = cfg.observations.get(mdp.OSL_OBS_GROUP)
  if group is None:
    raise ValueError(f"A stage-2 env config needs an {mdp.OSL_OBS_GROUP!r} group.")
  synced = mdp.osl_sensor_group(group)
  contact = synced.terms.get("contact")
  touch_sites = getattr(cfg, TOUCH_SITES_FIELD, None)
  if (
    contact is not None
    and mdp.osl_switch(contact.params, "heel")
    and touch_sites == "original"
  ):
    raise ValueError(
      "The OSL actor's heel contact reads the r_foot touch sensor, whose zone "
      f"never fires under {TOUCH_SITES_FLAG} original (the OSL heel lands behind "
      "it): the input would be dead, and the running normalizer would blow up its "
      f"first real reading. Pass {TOUCH_SITES_FLAG} feet_extended (or "
      f"heel_extended), or turn the heel contact off with {HEEL_FLAG} False."
    )
  if synced is not group:
    cfg.observations = {**cfg.observations, mdp.OSL_OBS_GROUP: synced}


def sync_gait_symmetry(cfg: Any) -> None:
  """Give a stage-2 config the gait-symmetry penalty its ``gait_symmetry`` asks for,
  in place: at weight 0 none (a config without the field counts as 0), otherwise
  the reward term :data:`~..mdp.SYMMETRY_METRIC` (:class:`~..mdp.gait_symmetry` at
  the negative weight, which the runner adds to both agents' rewards, as the twist
  term) and one metric per :data:`~..mdp.SYMMETRY_DIAGNOSTICS`.

  The config gets new rewards and metrics dicts when they change, rather than
  edited ones: the field-copying constructions share them between configs.

  Raises:
    ValueError: For a negative or non-finite weight, a sigma that is not positive
      and finite, or a weight on a task whose motion has no stride period (the
      stairs' single clip), naming the flag.
  """
  settings = getattr(cfg, GAIT_SYMMETRY_FIELD, None) or GaitSymmetryCfg()
  weight, sigma = float(settings.weight), float(settings.sigma)
  if not (math.isfinite(weight) and weight >= 0.0):
    raise ValueError(f"{GAIT_SYMMETRY_FLAG} must be >= 0 and finite: {weight}")
  if not (math.isfinite(sigma) and sigma > 0.0):
    raise ValueError(f"--env.gait-symmetry.sigma must be > 0 and finite: {sigma}")
  term = mdp.SYMMETRY_METRIC
  diagnostics = set(mdp.SYMMETRY_DIAGNOSTICS)
  rewards = {name: r for name, r in cfg.rewards.items() if name != term}
  metrics = {name: m for name, m in cfg.metrics.items() if name not in diagnostics}
  if weight == 0.0:
    if len(rewards) != len(cfg.rewards) or len(metrics) != len(cfg.metrics):
      cfg.rewards, cfg.metrics = rewards, metrics
    return
  motion = cfg.commands.get(MOTION_COMMAND)
  if not isinstance(motion, GaitLibraryCommandCfg):
    raise ValueError(
      f"{GAIT_SYMMETRY_FLAG} {weight:g}: the gait-symmetry penalty compares the OSL "
      "side with the intact side half a stride earlier, and this task's motion "
      f"({type(motion).__name__}) has no stride period: one clip that does not "
      f"loop. Leave {GAIT_SYMMETRY_FLAG} at 0 here."
    )
  rewards[term] = RewardTermCfg(
    func=mdp.gait_symmetry,
    weight=-weight,
    params={"command_name": MOTION_COMMAND, "sigma": sigma},
  )
  for name in mdp.SYMMETRY_DIAGNOSTICS:
    metrics[name] = MetricsTermCfg(
      func=mimic_component, params={"name": name, "reward_name": term}
    )
  cfg.rewards, cfg.metrics = rewards, metrics


def add_gait_symmetry_metric(cfg: Any) -> bool:
  """Give an evaluation config the gait-symmetry metrics term
  (:data:`~..mdp.SYMMETRY_METRIC`, :class:`~..mdp.gait_symmetry` at the config's
  sigma), whose per-step asymmetry the runner's validation pass accumulates, penalty
  on or off. Returns whether it did: only a motion with a stride period has one
  (the gait library's, not the stairs' clip). In place, with a new metrics dict.
  """
  if not isinstance(cfg.commands.get(MOTION_COMMAND), GaitLibraryCommandCfg):
    return False
  settings = getattr(cfg, GAIT_SYMMETRY_FIELD, None) or GaitSymmetryCfg()
  metric = MetricsTermCfg(
    func=mdp.gait_symmetry,
    params={"command_name": MOTION_COMMAND, "sigma": float(settings.sigma)},
  )
  cfg.metrics = {**cfg.metrics, mdp.SYMMETRY_METRIC: metric}
  return True


def check_no_perception_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Refuse a stage-1 config that carries stage 1's perception randomization
  (``OslDrCfg.encoder_bias``'s event, ``obs_noise``'s noise on the ``actor`` group):
  in stage 2 the ``actor`` group is the human's observation.

  Raises:
    ValueError: If it does.
  """
  actor = cfg.observations.get("actor")
  noisy = actor is not None and any(
    term is not None and term.noise is not None for term in actor.terms.values()
  )
  if PERCEPTION_DR_EVENT in cfg.events or noisy:
    raise ValueError(PERCEPTION_DR_REFUSAL)


def apply_split_stage(
  cfg: ManagerBasedRlEnvCfg,
  reward: Reward,
  mode: Mode,
  osl_obs: mdp.OslObsCfg | None = None,
) -> None:
  """Turn a stage-1 OSL config into its stage-2 split counterpart, in place (the
  module docstring's changes): the OSL sensors, the leg-contact sensor (unless
  stage 1's config has it), no ``critic`` group, the ``osl`` group, and the split
  reward with its metrics.

  The ``osl`` group is :func:`~..mdp.osl_observation_group` of ``mode`` and
  ``osl_obs``; :func:`sync_osl_group` checks it against the touch sites once the
  config is an :class:`OslSplitEnvCfg` (or when called).

  Args:
    cfg: A stage-1 OSL config: a robot entity with a ``spec_fn``, ``actor`` and
      ``critic`` groups and a ``mimic`` reward term.
    reward: ``"mimic"`` or ``"clf"``: the type ``cfg``'s ``mimic`` term computes.
    mode: ``"train"``, ``"validation"`` or ``"play"``.
    osl_obs: The OSL observation (default ``OslObsCfg()``).

  Raises:
    ValueError: If ``cfg`` carries stage 1's perception randomization.
  """
  check_no_perception_dr(cfg)
  entity = cfg.scene.entities["robot"]
  cfg.scene.entities["robot"] = replace(
    entity, spec_fn=with_osl_sensors(entity.spec_fn)
  )
  add_leg_contact_sensor(cfg)

  del cfg.observations["critic"]
  cfg.observations[mdp.OSL_OBS_GROUP] = mdp.osl_observation_group(mode, osl_obs)

  stage1 = cfg.rewards[REWARD_NAME]
  cfg.rewards[REWARD_NAME] = replace(
    stage1,
    func=mdp.split_mimic_reward,
    weight=1.0,
    params={**stage1.params, "channels": mdp.SplitRewardCfg()},
  )
  clf = reward == "clf"
  for name in dict.fromkeys((*mdp.SPLIT_DIAGNOSTICS, *mdp.SPLIT_CLF_DIAGNOSTICS)):
    if clf or name not in mdp.SPLIT_CLF_DIAGNOSTICS:
      cfg.metrics[name] = MetricsTermCfg(func=mimic_component, params={"name": name})
  # Per episode: the share of steps with a leg contact, and the hardest strike.
  for name, reduce in zip(mdp.SPLIT_CONTACT_DIAGNOSTICS, ("mean", "max"), strict=True):
    cfg.metrics[name] = MetricsTermCfg(
      func=mimic_component, params={"name": name}, reduce=reduce
    )


def osl_split_env_cfg(
  model: Model = "msk",
  socket: Socket = "compliant",
  reward: Reward = "clf",
  mode: Mode = "train",
  dr_cfg: OslDrCfg | None = None,
  osl_obs: mdp.OslObsCfg | None = None,
) -> OslSplitEnvCfg:
  """Create a stage-2 split environment over the unicycle library.

  Args:
    model: ``"robot"`` or ``"msk"``, as stage 1.
    socket: ``"rigid"`` or ``"compliant"``, as stage 1.
    reward: ``"mimic"`` or ``"clf"``. It may differ from stage 1's (the reward
      switch); the model and the socket may not.
    mode: ``"train"``, ``"validation"`` or ``"play"``, as stage 1. The OSL
      observation is corrupted in training only.
    dr_cfg: Stage 1's physics randomization ranges (default ``OslDrCfg()``).
    osl_obs: The OSL observation and its perception randomization (default
      ``OslObsCfg()``, the real leg's sensors).

  Raises:
    ValueError: If ``dr_cfg`` turns on a stage-1 perception switch, or for an
      ``osl`` group :func:`sync_osl_group` refuses.
  """
  if dr_cfg is not None and (dr_cfg.encoder_bias or dr_cfg.obs_noise):
    raise ValueError(PERCEPTION_DR_REFUSAL)
  cfg = osl_gait_library_env_cfg(
    model, socket, reward, mode, variant="unicycle", dr_cfg=dr_cfg
  )
  apply_split_stage(cfg, reward, mode, osl_obs)
  return OslSplitEnvCfg(**{f.name: getattr(cfg, f.name) for f in fields(cfg)})


def osl_split_terrain_env_cfg(
  model: Model = "msk",
  socket: Socket = "compliant",
  reward: Reward = "clf",
  mode: Mode = "train",
  dr_cfg: OslDrCfg | None = None,
  osl_obs: mdp.OslObsCfg | None = None,
) -> OslSplitEnvCfg:
  """The stage-2 split environment on stage 1's generated terrain: the terrain
  task's grid, level curriculum, start height and ``nan_term`` in training and play
  (:func:`~mjlab.tasks.osl_gait_library.config.env_cfgs.apply_terrain`); validation
  stays flat. Everything else is :func:`osl_split_env_cfg`'s."""
  cfg = osl_split_env_cfg(model, socket, reward, mode, dr_cfg, osl_obs)
  if mode != "validation":
    apply_terrain(cfg, mode)
  return cfg
