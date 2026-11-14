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
* the scene gains a contact sensor between the OSL and the biological left leg
  (:func:`~.sensors.leg_contact_sensor_cfg`), which the split reward reports as
  leg-crossing diagnostics (no penalty): a sensor changes no physics.

The stage-1 perception switches (``OslDrCfg.encoder_bias`` and ``obs_noise``) edit
the ``actor`` group, which is now the human's observation, so they are refused.

:func:`apply_split_stage` makes these changes to any stage-1 OSL config, so other
stage-2 tasks (stairs) chain it after their own stage-1 builder, and give their
config class :class:`OslSplitEnvCfg`'s ``__post_init__`` (or call
:func:`sync_osl_group`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.osl_gait_library.config.dr import OslDrCfg
from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward
from mjlab.tasks.osl_mimic.mdp import mimic_component
from mjlab.tasks.osl_mimic.touch_sites import CLI_FLAG as TOUCH_SITES_FLAG
from mjlab.tasks.osl_mimic.touch_sites import ENV_FIELD as TOUCH_SITES_FIELD
from mjlab.tasks.osl_mimic.touch_sites import OslEnvCfg
from mjlab.tasks.osl_split import mdp

from .sensors import leg_contact_sensor_cfg, with_osl_sensors

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


@dataclass(kw_only=True)
class OslSplitEnvCfg(OslEnvCfg):
  """A stage-2 env config: its robot builds the touch sites ``osl_touch_sites``
  names (as every OSL task's), and its ``osl`` group follows the sensor switches
  and the touch sites (:func:`sync_osl_group`)."""

  def __post_init__(self) -> None:
    # tyro builds the config after applying the command line, so this sees the
    # final switches and touch sites.
    super().__post_init__()
    sync_osl_group(self)


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
  module docstring's changes): the OSL sensors, the leg-contact sensor, no
  ``critic`` group, the ``osl`` group, and the split reward with its metrics.

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
  cfg.scene.sensors = (*cfg.scene.sensors, leg_contact_sensor_cfg(cfg.decimation))

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
