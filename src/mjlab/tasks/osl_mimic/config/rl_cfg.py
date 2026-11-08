"""musclemimic's PPO settings (``fullbody/conf_osl_fullbody*.yaml``) for rsl_rl."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .env_cfgs import NUM_ENVS, Reward

NUM_STEPS = 80
TOTAL_TIMESTEPS = 1_000_000_000
NUM_UPDATES = (
  TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS
)  # 6103, as musclemimic computes it.
NUM_VALIDATIONS = 100
HIDDEN_DIMS = (1024,) * 12 + (2048,) * 3 + (1024,)
INIT_STD = {"robot": 0.2, "msk": 3.0}


@dataclass
class MuscleMimicModelCfg(RslRlModelCfg):
  """:class:`~mjlab.tasks.osl_mimic.rl.models.MuscleMimicMLPModel`."""

  class_name: str = "mjlab.tasks.osl_mimic.rl.models:MuscleMimicMLPModel"
  hidden_dims: tuple[int, ...] = HIDDEN_DIMS
  activation: str = "swish"
  """SiLU, musclemimic's ``silu``; rsl_rl names it ``swish``."""
  obs_normalization: bool = True
  layer_norm: bool = True
  layer_norm_eps: float = 1e-5
  hidden_gain: float = math.sqrt(2.0)
  output_gain: float = 0.01


@dataclass
class MuscleMimicPpoCfg(RslRlPpoAlgorithmCfg):
  """:class:`~mjlab.tasks.osl_mimic.rl.ppo.MuscleMimicPPO`."""

  class_name: str = "mjlab.tasks.osl_mimic.rl.ppo:MuscleMimicPPO"
  num_learning_epochs: int = 1
  num_mini_batches: int = 128
  learning_rate: float = 4e-4
  schedule: Literal["adaptive", "fixed"] = "fixed"
  """rsl_rl's KL-adaptive schedule is off; ``lr_schedule`` sets the rate."""
  gamma: float = 0.99
  lam: float = 0.95
  entropy_coef: float = 0.0
  value_loss_coef: float = 0.25
  """musclemimic's ``vf_coef`` 0.5 times the 0.5 inside its value loss."""
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  max_grad_norm: float = 1.0
  normalize_advantage_per_mini_batch: bool = True
  muon_beta: float = 0.95
  muon_nesterov: bool = True
  muon_ns_steps: int = 5
  lr_schedule: Literal["linear", "constant"] = "linear"
  lr_schedule_updates: int = NUM_UPDATES
  """Updates over which the rate decays linearly to 0 (keep = max_iterations)."""
  reward_normalization: bool = True
  actor_bias_init: Literal["qpos0", "none"] = "none"
  bootstrap_time_outs: bool = False
  """musclemimic never bootstraps: its episodes end where the clip ends. Tasks whose
  episodes are cut off by a time limit (looping libraries) turn this on."""


@dataclass
class OslMimicValidationCfg:
  """In-training validation; see :mod:`mjlab.tasks.osl_mimic.rl.runner`."""

  interval: int = NUM_UPDATES // NUM_VALIDATIONS
  """Iterations between validations (61). 0 disables validation."""
  num_envs: int = 100
  num_steps: int = 0
  """Steps per validation; 0 runs one clip length."""
  model: str = "robot"
  socket: str = "rigid"
  reward: str = "mimic"


@dataclass
class OslMimicVideoCfg:
  """A video of the deterministic policy at every checkpoint, as musclemimic records
  at its validations; see :meth:`~mjlab.tasks.osl_mimic.rl.runner.OslMimicOnPolicyRunner.record_video`."""

  enabled: bool = True
  num_steps: int = 3000
  """Control steps per video: musclemimic's OSL ``video_length`` (30 s at 100 Hz)."""
  width: int = 640
  height: int = 480
  seed: int = 0
  """Seeds the video env's draws (commands, resets), so every checkpoint's video
  plays the same command sequence. Training's random state is left untouched."""


@dataclass
class OslMimicRunnerCfg(RslRlOnPolicyRunnerCfg):
  validation: OslMimicValidationCfg = field(default_factory=OslMimicValidationCfg)
  video: OslMimicVideoCfg = field(default_factory=OslMimicVideoCfg)


def osl_mimic_runner_cfg(
  model: Model, socket: Socket, reward: Reward, experiment_name: str
) -> OslMimicRunnerCfg:
  init_std = INIT_STD[model]
  return OslMimicRunnerCfg(
    actor=MuscleMimicModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": init_std,
        "std_type": "log",
      },
    ),
    critic=MuscleMimicModelCfg(),
    algorithm=MuscleMimicPpoCfg(
      # conf_osl_fullbody_robot.yaml opts the servos into it; the muscles never.
      actor_bias_init="qpos0" if model == "robot" else "none",
    ),
    validation=OslMimicValidationCfg(model=model, socket=socket, reward=reward),
    experiment_name=experiment_name,
    num_steps_per_env=NUM_STEPS,
    max_iterations=NUM_UPDATES,
    # About 0.4 GB per checkpoint (two 24M-parameter networks plus optimizer
    # state); musclemimic keeps 10.
    save_interval=610,
    wandb_tags=("osl_fullbody", model, reward),
  )
