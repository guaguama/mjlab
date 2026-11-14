"""The stairs stage-2 environment: stairs stage 1's, through unicycle stage 2's split.

Everything is ``osl_stairs_env_cfg(model, socket, reward, mode, dr_cfg)``: the clip
and its terrain boxes at their world coordinates, osl_mimic's simulation,
terminations (the clip's end among them, as a time-out, in a 300 s episode) and
1.0 m root-deviation limit, the gait-library physics randomization, and the blind
stage-1 observation, which becomes the human's ``actor`` group. Then exactly unicycle
stage 2's changes, by the same function
(:func:`~mjlab.tasks.osl_split.config.env_cfgs.apply_split_stage`): the OSL sensors,
the leg-contact sensor, no ``critic`` group, the ``osl`` group from the same builder
(so the OSL actor reads unicycle stage 2's sensor set, with its switches and
defaults), and the split reward with its metrics. Stairs has no twist term, so each
agent trains on its own reward channel alone.

One default differs from stairs stage 1's: the musculoskeletal model's CLF penalty
weight ``pen_w`` is unicycle stage 2's (0, not osl_mimic's 0.25). The warm start
refuses a stage-1 run trained with another (``--env.rewards.mimic.params.clf.pen-w``
sets it).

:class:`OslSplitStairsEnvCfg` is both packages' config class at once: whenever it is
constructed, which tyro does after applying the command line, it builds the touch
sites, makes the ``osl`` group consistent with the sensor switches and touch sites
(:func:`~mjlab.tasks.osl_split.config.env_cfgs.sync_osl_group`) and points the
terrain at the motion's clip (:func:`~mjlab.tasks.osl_stairs.config.env_cfgs.
sync_terrain`). Code that edits a built config calls those two.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_gait_library.config.dr import OslDrCfg
from mjlab.tasks.osl_gait_library.config.env_cfgs import UNICYCLE_MSK_CLF_PEN_W
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward
from mjlab.tasks.osl_split import mdp
from mjlab.tasks.osl_split.config.env_cfgs import (
  PERCEPTION_DR_REFUSAL,
  REWARD_NAME,
  OslSplitEnvCfg,
  apply_split_stage,
)
from mjlab.tasks.osl_stairs.config.env_cfgs import OslStairsEnvCfg, osl_stairs_env_cfg


@dataclass(kw_only=True)
class OslSplitStairsEnvCfg(OslStairsEnvCfg, OslSplitEnvCfg):
  """A stairs stage-2 env config: its robot builds the touch sites
  ``osl_touch_sites`` names, its ``osl`` group follows the sensor switches and the
  touch sites, and its terrain follows its motion command (in that order)."""


def osl_split_stairs_env_cfg(
  model: Model = "msk",
  socket: Socket = "compliant",
  reward: Reward = "clf",
  mode: Mode = "train",
  dr_cfg: OslDrCfg | None = None,
  osl_obs: mdp.OslObsCfg | None = None,
) -> OslSplitStairsEnvCfg:
  """Create a stairs stage-2 environment.

  Args:
    model: ``"robot"`` or ``"msk"``, as stairs stage 1.
    socket: ``"rigid"`` or ``"compliant"``, as stairs stage 1.
    reward: ``"mimic"`` or ``"clf"``. It may differ from stage 1's (the reward
      switch); the model and the socket may not.
    mode: ``"train"``, ``"validation"`` or ``"play"``, as stairs stage 1. The OSL
      observation is corrupted in training only.
    dr_cfg: Stage 1's physics randomization ranges (default ``OslDrCfg()``).
    osl_obs: The OSL observation and its perception randomization (default
      ``OslObsCfg()``, unicycle stage 2's default: the real leg's sensors).

  Raises:
    ValueError: If ``dr_cfg`` turns on a stage-1 perception switch, or for an
      ``osl`` group :func:`~mjlab.tasks.osl_split.config.env_cfgs.sync_osl_group`
      refuses.
  """
  if dr_cfg is not None and (dr_cfg.encoder_bias or dr_cfg.obs_noise):
    raise ValueError(PERCEPTION_DR_REFUSAL)
  cfg = osl_stairs_env_cfg(model, socket, reward, mode, dr_cfg)
  if model == "msk":
    clf = cfg.rewards[REWARD_NAME].params["clf"]
    if clf is not None:
      clf.pen_w = UNICYCLE_MSK_CLF_PEN_W
  apply_split_stage(cfg, reward, mode, osl_obs)
  return OslSplitStairsEnvCfg(**{f.name: getattr(cfg, f.name) for f in fields(cfg)})
