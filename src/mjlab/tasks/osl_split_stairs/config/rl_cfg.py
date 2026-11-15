"""Runner settings: unicycle stage 2's, with stairs stage 1's episode handling.

Everything is ``osl_split_runner_cfg(model, socket, reward, experiment_name)``: the
OSL actor's network (256 x 3, LayerNorm, swish, initial std 0.2 with a 0.05 floor),
the learning rates and their schedule, the human's freeze, the critics' warm-up, the
blend, the reward channels and their anneal, the run length and the save interval.
The changes, each stairs stage 1's own value (``osl_stairs_runner_cfg``):

* no bootstrapping on time-outs: the clip's end is a time-out, and the critics
  copied from stage 1 learnt its value without it;
* validation of ``num_envs`` envs (100) from frame 0 for one clip length
  (``num_steps`` 0), so ``ValidationTeacher/`` compares with the stage-1
  checkpoint's own ``Validation/``;

plus the variant ``"stairs"`` (the unicycle exporters refuse it; ``export`` here
writes the stairs bundle), videos of one climb, and the stairs tags.
"""

from __future__ import annotations

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.config.rl_cfg import MuscleMimicPpoCfg
from mjlab.tasks.osl_split.config.rl_cfg import SplitRunnerCfg, osl_split_runner_cfg
from mjlab.tasks.osl_stairs.config.rl_cfg import osl_stairs_runner_cfg

VARIANT = "stairs"
"""``validation.variant``: not a gait library."""
VIDEO_STEPS = 800
"""Control steps per video: one climb (the stairs clips last about 7.3 s at 100 Hz)."""


def osl_split_stairs_runner_cfg(
  model: Model, socket: Socket, reward: Reward, experiment_name: str
) -> SplitRunnerCfg:
  cfg = osl_split_runner_cfg(model, socket, reward, experiment_name)
  stage1 = osl_stairs_runner_cfg(model, socket, reward, experiment_name)
  assert isinstance(stage1.algorithm, MuscleMimicPpoCfg)
  cfg.algorithm.bootstrap_time_outs = stage1.algorithm.bootstrap_time_outs
  cfg.validation.num_envs = stage1.validation.num_envs
  cfg.validation.num_steps = stage1.validation.num_steps
  cfg.validation.variant = VARIANT
  cfg.video.num_steps = VIDEO_STEPS
  cfg.wandb_tags = ("osl_fullbody", model, reward, "stairs", "split")
  return cfg
