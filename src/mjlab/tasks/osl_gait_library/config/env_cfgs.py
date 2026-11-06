"""The gait-library environment: a musclemimic imitation task over a library.

Everything not listed here is ``osl_mimic_env_cfg(model, socket, reward, mode)``:
2 ms Euler steps at 100 Hz control, musclemimic's observation, the MimicReward or
CLF reward, the training terminations and no randomization. The variants are the
musculoskeletal model with CLF and the robot model with MimicReward. The changes:

* the reference comes from :class:`~..mdp.GaitLibraryCommand` (sesteban's G1
  unicycle command, here over a forward-walking library plus idle);
* the commanded twist and the stride clock join the observation (actor and critic);
* a stride-averaged twist reward (``mdp.stride_twist``) joins the imitation reward;
* clips loop, so the trajectory-end termination is gone and episodes last 10 s, as
  in sesteban's G1 tasks.
"""

from __future__ import annotations

import os

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.osl_gait_library import mdp
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward, osl_mimic_env_cfg
from mjlab.tasks.osl_mimic.mdp import mimic_component

LIBRARY_ENV_VAR = "MJLAB_OSL_GAIT_LIBRARY"
"""Default library folder when ``--env.commands.motion.library-dir`` is not given."""

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


def osl_gait_library_env_cfg(
  model: Model = "robot",
  socket: Socket = "rigid",
  reward: Reward = "mimic",
  mode: Mode = "train",
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
  """
  cfg = osl_mimic_env_cfg(model, socket, reward, mode)
  validation = mode == "validation"
  cfg.commands = {
    "motion": mdp.GaitLibraryCommandCfg(
      library_dir=os.environ.get(LIBRARY_ENV_VAR, ""),
      fixed_clips=validation,
      resampling_time_range=(1.0e9, 1.0e9) if validation else (3.0, 8.0),
      rel_static_envs=REL_STATIC_ENVS,
    )
  }

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

  cfg.episode_length_s = EPISODE_LENGTH_S if mode != "play" else 1.0e9
  return cfg
