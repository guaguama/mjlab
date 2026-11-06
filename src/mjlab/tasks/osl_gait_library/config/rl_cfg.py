"""Runner settings: the matching osl_mimic task's, plus time-out bootstrapping."""

from __future__ import annotations

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.config.rl_cfg import (
  MuscleMimicPpoCfg,
  OslMimicRunnerCfg,
  OslMimicValidationCfg,
  osl_mimic_runner_cfg,
)

from .env_cfgs import EPISODE_LENGTH_S


def osl_gait_library_runner_cfg(
  model: Model, socket: Socket, reward: Reward, experiment_name: str
) -> OslMimicRunnerCfg:
  cfg = osl_mimic_runner_cfg(model, socket, reward, experiment_name)
  assert isinstance(cfg.algorithm, MuscleMimicPpoCfg)
  # Library episodes are cut off by the 10 s limit, not finished, so the critic
  # bootstraps there (rsl_rl's default; musclemimic's clips simply end).
  cfg.algorithm.bootstrap_time_outs = True
  cfg.validation = OslMimicValidationCfg(
    interval=cfg.validation.interval,
    num_envs=0,  # One env per library clip.
    num_steps=round(EPISODE_LENGTH_S * 100),
    model=model,
    socket=socket,
    reward=reward,
  )
  cfg.wandb_tags = ("osl_fullbody", model, reward, "gait_library")
  return cfg
