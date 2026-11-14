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
  lr_schedule_offset: int = 0
  """Updates before the linear decay starts counting. A resumed run keeps its update
  count, so set this to the checkpoint's count for a fresh schedule."""
  lr_final_ratio: float = 0.0
  """The linear decay ends at this fraction of ``learning_rate`` (musclemimic: 0)."""
  reward_normalization: bool = True
  actor_bias_init: Literal["qpos0", "none"] = "none"
  bootstrap_time_outs: bool = False
  """musclemimic never bootstraps: its episodes end where the clip ends. Tasks whose
  episodes are cut off by a time limit (looping libraries) turn this on."""
  std_cap_start: float = 0.0
  """Upper bound on every action std through update ``std_cap_begin_update``; 0
  disables the cap. The std stays learnable below it."""
  std_cap_end: float = 0.0
  """The bound from update ``std_cap_end_update`` on, geometric in between; 0 keeps
  ``std_cap_start``."""
  std_cap_begin_update: int = 0
  std_cap_end_update: int = 0
  """Update counts (the algorithm's, which a resumed run keeps) bracketing the cap's
  decay."""
  std_floor: float = 0.0
  """Lower bound on every action std; 0 disables it."""
  init_std_terms: dict[str, float] = field(default_factory=dict)
  """Initial std of named action terms, overriding the actor's ``init_std`` for their
  outputs; a resumed run keeps its checkpoint's std. On the command line, a Python
  dict: ``--agent.algorithm.init-std-terms "{'osl_motors': 0.3}"``."""
  muon_ns_dtype: Literal["float32", "bfloat16"] = "float32"
  """Newton-Schulz precision: optax's float32, or the reference Muon's bfloat16."""


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
  inference: tuple[str, ...] = ("mean",)
  """Deterministic actions validated, each under its own prefix: ``"mean"`` (the raw
  mean, musclemimic's, ``Validation/``), ``"expected"`` (the noise-averaged action
  ``E[clip(mu + sigma eps)]``, ``ValidationExpected/``) and ``"sampled"`` (the
  stochastic policy, ``ValidationSampled/``). On the command line, a Python tuple:
  ``--agent.validation.inference "('mean','expected')"``."""
  stress_interval: int = 0
  """Iterations between stress validations; 0 disables them. A stress validation
  plays the training distribution (random clips and switches, physical draws,
  pushes, random actuator delays) with deterministic actions, ending episodes only
  on validation's 0.5 m site check, the deployment harness's failure criterion. It
  reports each env's first episode (``Stress/``, ``StressExpected/``, ...)."""
  stress_num_envs: int = 512
  stress_num_steps: int = 1000
  """Steps per stress episode (10 s at 100 Hz); the stress env's episode length
  follows it, so ``--agent.validation.stress-num-steps 2000`` alone gives 20 s."""
  stress_inference: tuple[str, ...] = ("expected",)
  """Deterministic actions stress-validated (as ``inference``)."""
  seed: int = 0
  """Seeds torch's generators for the validation env's build and for every
  validation pass alike, so checkpoints meet the same draws; training's random
  state is left untouched. It cannot make passes bitwise identical on the GPU:
  mujoco_warp's GPU kernels are not deterministic, so the same weights can still
  score slightly differently from pass to pass there."""
  stress_seed: int = 0
  """Seeds every stress validation alike (commands, pushes, delays), so checkpoints
  meet the same disturbances. Training's random state is left untouched."""


@dataclass
class OslMimicVideoCfg:
  """A video of the deterministic policy at every checkpoint, as musclemimic records
  at its validations; see :meth:`~mjlab.tasks.osl_mimic.rl.runner.OslMimicOnPolicyRunner.record_video`."""

  enabled: bool = True
  num_steps: int = 3000
  """Control steps per video: musclemimic's OSL ``video_length`` (30 s at 100 Hz)."""
  width: int = 640
  height: int = 480
  inference: str = "auto"
  """``"mean"``, ``"expected"`` (see ``OslMimicValidationCfg.inference``) or
  ``"auto"``: ``"expected"`` when the policy drives muscles whose median std exceeds
  0.3, else ``"mean"``."""
  seed: int = 0
  """Seeds the video env's draws (commands, resets), so every checkpoint's video
  plays the same command sequence. Training's random state is left untouched."""


@dataclass
class OslMimicRunnerCfg(RslRlOnPolicyRunnerCfg):
  validation: OslMimicValidationCfg = field(default_factory=OslMimicValidationCfg)
  video: OslMimicVideoCfg = field(default_factory=OslMimicVideoCfg)
  env_stages: tuple[tuple[int, str, float], ...] = ()
  """Training-env settings staged by PPO update. Each ``(until_update, path, value)``
  runs ``value`` before update ``until_update`` and the env config's own value from
  then on; where two stages of one path overlap, the earlier-ending one holds.
  ``path`` is ``rewards.<term>.weight`` or ``terminations.<term>.<param>``. The
  update count is the algorithm's, which checkpoints carry, so a resumed run starts
  in the right stage. On the command line, e.g.
  ``--agent.env-stages "((1500,'terminations.root_deviation.threshold',1.0),)"``."""


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
    validation=OslMimicValidationCfg(
      model=model,
      socket=socket,
      reward=reward,
      # The MSK policy's raw mean fails while its muscle std is large; the
      # noise-averaged action shows what it has learned meanwhile.
      inference=("mean", "expected") if model == "msk" else ("mean",),
    ),
    experiment_name=experiment_name,
    num_steps_per_env=NUM_STEPS,
    max_iterations=NUM_UPDATES,
    # About 0.4 GB per checkpoint (two 24M-parameter networks plus optimizer
    # state); musclemimic keeps 10.
    save_interval=610,
    wandb_tags=("osl_fullbody", model, reward),
  )
