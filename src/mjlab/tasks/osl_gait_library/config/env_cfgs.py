"""The gait-library environment: a musclemimic imitation task over a library.

Everything not listed here is ``osl_mimic_env_cfg(model, socket, reward, mode)``:
2 ms Euler steps at 100 Hz control, musclemimic's observation, the MimicReward or
CLF reward, the training terminations and no randomization. The changes:

* the reference comes from :class:`~..mdp.GaitLibraryCommand` (sesteban's G1
  unicycle command);
* the commanded twist and the stride clock join the observation (actor and critic);
* a stride-averaged twist reward (``mdp.stride_twist``) joins the imitation reward;
* clips loop, so the trajectory-end termination is gone and episodes last 10 s, as
  in sesteban's G1 tasks.

Two variants:

* ``"walk_fwd"``: the forward-walking library plus idle, sesteban's continuous
  sampler over its speed span, no randomization (round 1, unchanged);
* ``"unicycle"``: the full unicycle library (forward and backward arcs, pivots,
  idle), with commands drawn at random from the library's clips in sesteban's G1
  unicycle mode split, episodes starting at a random heading, and stage-1 domain
  randomization (``config/dr.py``). Its MSK tasks train with the stage-1 recipe: the
  ``UNICYCLE_MSK_*`` reward and termination settings below and the PPO schedule in
  ``config/rl_cfg.py``.
"""

from __future__ import annotations

import math
import os
from typing import Literal

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.osl_gait_library import mdp
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward, osl_mimic_env_cfg
from mjlab.tasks.osl_mimic.mdp import mimic_component

from .dr import OslDrCfg, apply_osl_dr

Variant = Literal["walk_fwd", "unicycle"]

LIBRARY_ENV_VAR = "MJLAB_OSL_GAIT_LIBRARY"
"""Default forward library folder when ``--env.commands.motion.library-dir`` is not
given."""
UNICYCLE_LIBRARY_ENV_VAR = "MJLAB_OSL_UNICYCLE_LIBRARY"
"""Default unicycle library folder, likewise."""

EPISODE_LENGTH_S = 10.0
REL_STATIC_ENVS = 0.15
"""sesteban's G1 unicycle stand fraction."""

TWIST_STD = 0.1
"""About two 0.05 m/s grid steps per axis."""
TWIST_WEIGHT = 0.06
"""sesteban's twist term holds 0.3 of about 5.3 positive weight (~6%); the CLF
reward's positive part peaks at 1, so the same share is 0.06. MimicReward's peaks
at 1.31 (rvel counts twice), so under it the same weight is a ~4.6% share; it is
kept equal so the two variants differ only in the imitation reward."""

TWIST_DIAGNOSTICS = (
  "twist_reward",
  "twist_err_vx",
  "twist_err_vy",
  "twist_err_wz",
  "twist_vx_cmd",
  "twist_vx_robot",
)

# ===== The unicycle library (clipper unicycle_v001, library.json "grid") ========
# Commands are drawn from the library itself (``sampling="library"``): a mode in
# sesteban's G1 unicycle split, then one of that mode's clips uniformly at random,
# commanded at its exact label. Per clip that is idle 15%, each of the 50 pivots
# 0.6%, each of the 147 backward arcs 0.15%, each of the 581 forward arcs 0.057%.
UNICYCLE_REL_TURN_ENVS = 0.3
"""Pivots, sesteban's walk and jog unicycle fraction."""
UNICYCLE_REL_BACK_ENVS = 0.4
"""Of the arcs, the backward fraction (sesteban's)."""
# The ranges below are the library's grid. They drive only the continuous sampler
# (``sampling="continuous"``), kept as the alternative to library draws; the load
# check verifies every twist they span snaps within 0.1 of a clip.
UNICYCLE_FWD_ARC_REGIONS: tuple[mdp.TwistBox, ...] = (
  # The 0.05-step band, vx 0.5-1.5 x |wz| <= 0.5 (441 clips).
  ((0.5, 1.5), (0.0, 0.5), 0.75),
  # The 0.1-step tight band, vx 0.6-1.2 x |wz| 0.6-1.5 (140 clips), sampled from
  # |wz| 0.5 so that the gap between the two bands is covered too.
  ((0.6, 1.2), (0.5, 1.5), 0.25),
)
UNICYCLE_BCK_ARC_REGIONS: tuple[mdp.TwistBox, ...] = (((-0.8, -0.5), (0.0, 0.5), 1.0),)
UNICYCLE_WZ_RANGE = (0.5, 1.7)
"""Pivot yaw-rate magnitudes."""
UNICYCLE_TWIST_METRIC_WEIGHTS = mdp.span_normalized_weights((-0.8, 1.5), (-1.7, 1.7))
"""The nearest-clip metric over the library's whole (vx, wz) span."""
UNICYCLE_START_YAW_RANGE = math.pi
"""Episodes start at a uniformly random heading. The observation is world-frame, and
a policy trained only on the clips' native headings walked straight and on arcs only
there: deployed, it fell once an arc or a re-base turned it 130-220 deg away (the
pivots, which sweep every heading in training, held)."""

# ===== The unicycle MSK recipe (tuned on -MSK-CLF-Compliant, 2026-11) =============
UNICYCLE_MSK_ROOT_DEVIATION = 2.0
"""The root's drift limit in m (osl_mimic's 1.0). The room lets a drifting policy
recover instead of being reset; it raised the raw mean's stress survival from ~44%
to ~55% within 400 iterations, with validation unchanged."""
UNICYCLE_MSK_TWIST_WEIGHT = 0.2
"""The twist term's weight (``TWIST_WEIGHT`` elsewhere). Curved walks that fail turn
too slowly and fall inward; 0.2 cut forward and tight-arc failures and the yaw-rate
error, with stress survival unchanged."""
UNICYCLE_MSK_CLF_PEN_W = 0.0
"""The CLF decrease-condition penalty (0.25 elsewhere). It saturated at a constant
-0.22 that, with the reward clipped at 0, zeroed the reward of every state far from
the reference; without it training led on every metric from iteration 50."""


def osl_gait_library_env_cfg(
  model: Model = "robot",
  socket: Socket = "rigid",
  reward: Reward = "mimic",
  mode: Mode = "train",
  variant: Variant = "walk_fwd",
  dr_cfg: OslDrCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create a gait-library environment.

  Args:
    model: ``"robot"`` (27 position servos + 2 OSL motors) or ``"msk"`` (339
      muscles + 2 OSL motors, fingerless).
    socket: ``"rigid"`` or ``"compliant"``. Library clips are rigid; the compliant
      socket's joints are held at zero in the reference.
    reward: ``"mimic"`` (MimicReward) or ``"clf"`` (CLFReward).
    mode: ``"train"`` samples twists and random phases; ``"validation"`` plays every
      clip from phase 0 at its own twist (one env per clip, set by the runner) and
      ends episodes only on the 0.5 m site check; ``"play"`` samples like training,
      with endless episodes.
    variant: ``"walk_fwd"`` (forward library, no randomization) or ``"unicycle"``
      (full library, randomization).
    dr_cfg: Randomization ranges of the unicycle variant (default ``OslDrCfg()``).
  """
  cfg = osl_mimic_env_cfg(model, socket, reward, mode)
  validation = mode == "validation"
  env_var = UNICYCLE_LIBRARY_ENV_VAR if variant == "unicycle" else LIBRARY_ENV_VAR
  motion_cfg = mdp.GaitLibraryCommandCfg(
    library_dir=os.environ.get(env_var, ""),
    fixed_clips=validation,
    resampling_time_range=(1.0e9, 1.0e9) if validation else (3.0, 8.0),
    rel_static_envs=REL_STATIC_ENVS,
    debug_vis=mode == "play",  # The reference ghost.
  )
  if variant == "unicycle":
    motion_cfg.sampling = "library"
    motion_cfg.rel_turn_envs = UNICYCLE_REL_TURN_ENVS
    motion_cfg.rel_back_envs = UNICYCLE_REL_BACK_ENVS
    motion_cfg.fwd_arc_regions = UNICYCLE_FWD_ARC_REGIONS
    motion_cfg.bck_arc_regions = UNICYCLE_BCK_ARC_REGIONS
    motion_cfg.wz_range = UNICYCLE_WZ_RANGE
    motion_cfg.twist_metric_weights = UNICYCLE_TWIST_METRIC_WEIGHTS
    motion_cfg.start_yaw_range = 0.0 if validation else UNICYCLE_START_YAW_RANGE
  cfg.commands = {"motion": motion_cfg}

  motion = {"command_name": "motion"}
  for group in cfg.observations.values():
    group.terms["twist"] = ObservationTermCfg(func=mdp.twist_command, params=motion)
    group.terms["phase"] = ObservationTermCfg(func=mdp.stride_phase, params=motion)

  del cfg.terminations["trajectory_end"]

  cfg.rewards["twist"] = RewardTermCfg(
    func=mdp.stride_twist, weight=TWIST_WEIGHT, params={**motion, "std": TWIST_STD}
  )
  for name in TWIST_DIAGNOSTICS:
    cfg.metrics[name] = MetricsTermCfg(
      func=mimic_component, params={"name": name, "reward_name": "twist"}
    )

  if variant == "unicycle" and model == "msk":
    cfg.rewards["twist"].weight = UNICYCLE_MSK_TWIST_WEIGHT
    clf = cfg.rewards["mimic"].params["clf"]
    if clf is not None:
      clf.pen_w = UNICYCLE_MSK_CLF_PEN_W
    if "root_deviation" in cfg.terminations:  # Training mode only.
      root = cfg.terminations["root_deviation"]
      root.params = {**root.params, "threshold": UNICYCLE_MSK_ROOT_DEVIATION}

  if variant == "unicycle":
    apply_osl_dr(cfg, model, socket, mode, dr_cfg)

  cfg.episode_length_s = EPISODE_LENGTH_S if mode != "play" else 1.0e9
  return cfg
