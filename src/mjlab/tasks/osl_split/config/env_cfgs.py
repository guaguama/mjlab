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
  in training (:func:`~..mdp.osl_observation_group`);
* the imitation reward becomes :class:`~..mdp.split_mimic_reward` under the same name,
  ``mimic``, so the stage-1 diagnostics keep their keys: it returns the same team
  reward, exposes the two agents' reward channels, and adds their diagnostics;
* the scene gains a contact sensor between the OSL and the biological left leg
  (:func:`~.sensors.leg_contact_sensor_cfg`), which the split reward reports as
  leg-crossing diagnostics (no penalty): a sensor changes no physics.

The stage-1 perception switches (``OslDrCfg.encoder_bias`` and ``obs_noise``) edit
the ``actor`` group, which is now the human's observation, so they are refused.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.osl_gait_library.config.dr import OslDrCfg
from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward
from mjlab.tasks.osl_mimic.mdp import mimic_component
from mjlab.tasks.osl_mimic.touch_sites import OslEnvCfg
from mjlab.tasks.osl_split import mdp

from .sensors import leg_contact_sensor_cfg, with_osl_sensors

REWARD_NAME = "mimic"
"""The imitation reward's term name, kept from stage 1."""


def osl_split_env_cfg(
  model: Model = "msk",
  socket: Socket = "compliant",
  reward: Reward = "clf",
  mode: Mode = "train",
  dr_cfg: OslDrCfg | None = None,
  osl_obs: mdp.OslObsCfg | None = None,
) -> OslEnvCfg:
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
      ``OslObsCfg()``).

  Raises:
    ValueError: If ``dr_cfg`` turns on a stage-1 perception switch.
  """
  if dr_cfg is not None and (dr_cfg.encoder_bias or dr_cfg.obs_noise):
    raise ValueError(
      "OslDrCfg.encoder_bias and obs_noise corrupt the 'actor' group, which in "
      "stage 2 is the human's observation. The OSL's perception randomization is "
      "OslObsCfg's (osl_obs)."
    )
  cfg = osl_gait_library_env_cfg(
    model, socket, reward, mode, variant="unicycle", dr_cfg=dr_cfg
  )

  entity = cfg.scene.entities["robot"]
  cfg.scene.entities["robot"] = replace(
    entity, spec_fn=with_osl_sensors(entity.spec_fn)
  )
  cfg.scene.sensors = (*cfg.scene.sensors, leg_contact_sensor_cfg(cfg.decimation))

  del cfg.observations["critic"]
  cfg.observations["osl"] = mdp.osl_observation_group(mode, osl_obs)

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
  return cfg
