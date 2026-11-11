"""Runner settings: the matching osl_mimic task's (musclemimic's PPO, validation from
frame 0 over one clip length, a video per checkpoint)."""

from __future__ import annotations

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.config.rl_cfg import OslMimicRunnerCfg, osl_mimic_runner_cfg


def osl_stairs_runner_cfg(
  model: Model, socket: Socket, reward: Reward, experiment_name: str
) -> OslMimicRunnerCfg:
  cfg = osl_mimic_runner_cfg(model, socket, reward, experiment_name)
  cfg.wandb_tags = ("osl_fullbody", model, reward, "stairs")
  return cfg
