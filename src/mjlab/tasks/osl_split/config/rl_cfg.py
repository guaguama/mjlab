"""Runner settings for the stage-2 split tasks.

Everything is the stage-1 unicycle runner's (:func:`osl_gait_library_runner_cfg`),
plus what stage 2 adds:

* ``warm_start``: the stage-1 run (or checkpoint, or converted warm-start file) the
  human actor and both critics start from;
* ``osl_actor`` / ``osl_critic``: the fresh OSL actor and the OSL critic;
* ``algorithm`` (:class:`SplitPpoCfg`): per-module learning rates, per-agent PPO
  constants, the human's std policy and the carried return statistics;
* ``freeze``: the curriculum (critic-only warm-up, then the OSL actor, then the human
  actor unfreezes);
* ``blend``: an optional ramp from the unified policy's OSL head to the OSL actor
  during the freeze (off by default);
* ``channels``: the two reward channels' coordinate-group weights over stage 2.

Iteration counts here are *stage-local*: iteration 0 is stage 2's first update,
whatever global iteration number it is logged under (stage 2 continues stage 1's
iteration axis).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.osl_gait_library.config.rl_cfg import (
  GaitLibraryValidationCfg,
  osl_gait_library_runner_cfg,
)
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.config.rl_cfg import (
  INIT_STD,
  MuscleMimicModelCfg,
  MuscleMimicPpoCfg,
  OslMimicVideoCfg,
)

STAGE2_ITERATIONS = 1000
"""Default stage-2 length: short first runs, to see the freeze and the unfreeze."""
SAVE_INTERVAL = 250
"""Stage-local iterations between checkpoints (about 0.7 GB each: the human actor and
two 24M-parameter critics with their optimizer states)."""
OSL_HIDDEN_DIMS = (256, 256, 256)
OSL_INIT_STD = 0.2
"""In normalized action units: 0.2 x 2.88 x 49.4/58.4 = about 28/34 N m of knee/ankle
torque noise. The unified policy's OSL std had fallen to 0.07/0.05 by stage 1's
iteration 610; the MSK default of 3.0 would be near bang-bang torque."""
OSL_MIN_STD = 0.1
"""The OSL actor's std floor (rsl_rl's ``std_range``): with no entropy bonus the
unified policy's OSL std collapsed within ~600 iterations."""

OBS_GROUPS: dict[str, tuple[str, ...]] = {
  "actor": ("actor",),
  "critic": ("actor",),
  "osl_actor": ("osl",),
  "osl_critic": ("actor",),
}
"""rsl_rl observation sets: the human actor and both critics read the stage-1
``actor`` group (the stage-1 ``critic`` group was an identical copy and is dropped);
the OSL actor reads the ``osl`` group."""


@dataclass
class OslActorModelCfg(MuscleMimicModelCfg):
  """The fresh OSL actor: musclemimic's MLP family (LayerNorm, SiLU, its own
  observation normalizer), small."""

  hidden_dims: tuple[int, ...] = OSL_HIDDEN_DIMS
  distribution_cfg: dict | None = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": OSL_INIT_STD,
      "std_type": "log",
      "std_range": (OSL_MIN_STD, 1.0e6),
    }
  )


@dataclass
class SplitPpoCfg(MuscleMimicPpoCfg):
  """:class:`~mjlab.tasks.osl_split.rl.ppo.TwoAgentPPO`.

  Inherited and still used: ``gamma``, ``lam``, ``num_learning_epochs``,
  ``num_mini_batches``, ``clip_param`` (the human's), ``value_loss_coef``,
  ``entropy_coef`` (the human's), ``use_clipped_value_loss``, ``max_grad_norm`` (per
  module), ``normalize_advantage_per_mini_batch``, the ``muon_*`` constants,
  ``reward_normalization`` and ``bootstrap_time_outs``. Inherited and ignored:
  ``learning_rate``, ``lr_schedule``, ``lr_schedule_updates``, ``schedule``,
  ``desired_kl``, ``optimizer`` and ``actor_bias_init`` (the learning rates below
  replace the first three; the human's output bias comes from the warm start).
  """

  class_name: str = "mjlab.tasks.osl_split.rl.ppo:TwoAgentPPO"
  bootstrap_time_outs: bool = True
  actor_bias_init: Literal["qpos0", "none"] = "none"

  critic_lr: float = 4e-4
  """Both critics: linear from this to 0 over the stage, from stage iteration 0."""
  osl_actor_lr: float = 4e-4
  """The OSL actor: this rate on the same linear decay, once it starts learning."""
  human_actor_peak_lr: float = 1e-4
  """The human actor after the unfreeze: a linear ramp from 0 to this, then the
  linear decay to 0 at the end of the stage. Muon's step size is set by the rate,
  not the gradient, so this is the only brake on how far the warm human moves."""
  human_actor_ramp_iterations: int = 50
  schedule_iterations: int = 0
  """Iterations over which the rates decay to 0; 0 uses the run's
  ``max_iterations`` (kept in the checkpoint, so a resumed run keeps its schedule)."""

  osl_clip_param: float = 0.2
  osl_entropy_coef: float = 0.0
  osl_value_loss_coef: float = 0.25

  human_learn_std: bool = False
  """False keeps the human's std at its stage-1 value for the whole stage. The MSK
  human's mean muscle commands rely on that noise (clipped noise is co-contraction),
  so its std must not be lowered without compensating the mean."""
  return_count_iterations: float = 3.0
  """A copied critic keeps the stage-1 return statistics it was trained on, with
  their count reset to this many iterations' samples so they can follow the stage-2
  reward channels (a stage-1 count of ~1e9 would freeze them)."""
  osl_critic_obs: Literal["full", "full+osl"] = "full"
  """``"full+osl"`` appends the OSL actor's observation to the OSL critic's input
  (zero-initialized weights, its own normalizer)."""


@dataclass
class WarmStartCfg:
  path: str = ""
  """A stage-1 run directory, a stage-1 checkpoint (``model_N.pt``), or a warm-start
  file written by ``python -m mjlab.tasks.osl_split.scripts.split``. Required to
  train; ``play`` loads stage-2 checkpoints instead."""
  checkpoint: str = ""
  """Checkpoint file inside a run directory; empty picks the highest-numbered
  ``model_N.pt``."""


@dataclass
class SplitFreezeCfg:
  human_actor_iterations: int = 200
  """Stage-local iterations with the human actor frozen (weights, std and observation
  statistics). Set from the first curves."""
  critic_warmup_iterations: int = 10
  """Stage-local iterations with only the critics learning: the OSL actor acts with
  its initial policy until then."""
  reward_switch_critic_warmup_iterations: int = 50
  """The critic-only warm-up when stage 2's reward type differs from stage 1's (the
  critics then start fresh)."""


@dataclass
class SplitBlendCfg:
  enabled: bool = False
  """Drive the OSL with ``(1 - beta) * teacher + beta * osl_actor``, where the teacher
  is the unified policy's OSL head on the frozen human trunk. ``beta`` ramps from 0
  to 1 starting when the OSL actor starts learning; it must reach 1 by the human
  unfreeze, after which the trunk (and so the teacher) changes."""
  ramp_iterations: int = 100


@dataclass
class ChannelWeightsCfg:
  """Per-coordinate weights of one reward channel, by coordinate group."""

  human: float = 1.0
  """Human-only coordinates: the root and every non-OSL mimic site (and, under
  MimicReward, every non-OSL joint)."""
  osl_sites: float = 1.0
  """The three OSL mimic sites (knee, ankle and foot assemblies). Shared: the hip,
  the socket and the knee move them together."""
  osl_joints: float = 0.0
  """The OSL knee and ankle joint errors: the only coordinates the OSL fully controls
  and senses. Absent from stage 1's CLF (weight 0 there); part of MimicReward's joint
  kernels (weight 1 there)."""


def _stage1_weights(reward: Reward) -> ChannelWeightsCfg:
  return ChannelWeightsCfg(osl_joints=0.0 if reward == "clf" else 1.0)


@dataclass
class SplitChannelsCfg:
  """The two reward channels' group weights over the stage.

  Each channel holds its start weights until ``anneal_start_iteration`` and then
  moves linearly to its end weights over ``anneal_iterations``. The start weights
  are the stage-1 equivalents, so both channels begin as the stage-1 reward. Only
  relative weights within a channel matter; a group's share of a channel is
  ``weight x coordinates / sum`` (logged as ``Channels/<channel>/share_<group>``).
  """

  human_start: ChannelWeightsCfg = field(default_factory=ChannelWeightsCfg)
  human_end: ChannelWeightsCfg = field(default_factory=ChannelWeightsCfg)
  osl_start: ChannelWeightsCfg = field(default_factory=ChannelWeightsCfg)
  osl_end: ChannelWeightsCfg = field(
    default_factory=lambda: ChannelWeightsCfg(human=0.5, osl_sites=1.0, osl_joints=10.0)
  )
  """Under the CLF the 84 human coordinates at 0.5, 18 OSL-site coordinates at 1 and
  2 OSL joint coordinates at 10 give shares of about 52/23/25%."""
  anneal_start_iteration: int = -1
  """-1: at the human unfreeze (``freeze.human_actor_iterations``)."""
  anneal_iterations: int = 200
  additive_term_weights: dict[str, tuple[float, float]] = field(default_factory=dict)
  """``{term: (human, osl)}`` multipliers on the env's additive reward terms (every
  term but ``mimic``, e.g. ``twist``); a term not listed counts fully in both."""


@dataclass
class SplitValidationCfg(GaitLibraryValidationCfg):
  expected: bool = True
  """Also validate with the noise-averaged human action ``E[clip(mu + sigma eps)]``
  (``ValidationExpected/``): the raw mean action of the stage-1 protocol
  (``Validation/``) ends every MSK episode early."""
  zero_shot: bool = True
  """Validate the split policies once before stage 2's first update."""


@dataclass
class SplitRunnerCfg(RslRlOnPolicyRunnerCfg):
  # Narrower types than the base fields (pyright's
  # reportIncompatibleVariableOverride), so tyro and the runner see the extras.
  actor: MuscleMimicModelCfg = field(  # pyright: ignore
    default_factory=MuscleMimicModelCfg
  )
  """The human actor: stage 1's architecture (the warm start must match it)."""
  critic: MuscleMimicModelCfg = field(  # pyright: ignore
    default_factory=MuscleMimicModelCfg
  )
  """The human critic: stage 1's critic architecture."""
  osl_actor: OslActorModelCfg = field(default_factory=OslActorModelCfg)
  osl_critic: MuscleMimicModelCfg = field(default_factory=MuscleMimicModelCfg)
  """The OSL critic: a copy of the stage-1 critic, so stage 1's architecture."""
  algorithm: SplitPpoCfg = field(default_factory=SplitPpoCfg)  # pyright: ignore
  validation: SplitValidationCfg = field(default_factory=SplitValidationCfg)
  video: OslMimicVideoCfg = field(default_factory=OslMimicVideoCfg)
  warm_start: WarmStartCfg = field(default_factory=WarmStartCfg)
  freeze: SplitFreezeCfg = field(default_factory=SplitFreezeCfg)
  blend: SplitBlendCfg = field(default_factory=SplitBlendCfg)
  channels: SplitChannelsCfg = field(default_factory=SplitChannelsCfg)
  export_osl_onnx: bool = True
  """Export the OSL actor (ONNX plus a metadata JSON) next to every checkpoint."""


def osl_split_runner_cfg(
  model: Model, socket: Socket, reward: Reward, experiment_name: str
) -> SplitRunnerCfg:
  base = osl_gait_library_runner_cfg(
    model, socket, reward, experiment_name, variant="unicycle"
  )
  shared = {
    f.name: getattr(base, f.name)
    for f in fields(base)
    if f.name not in {"algorithm", "validation"}
  }
  cfg = SplitRunnerCfg(**shared)
  assert isinstance(base.algorithm, MuscleMimicPpoCfg)
  inherited = {
    f.name: getattr(base.algorithm, f.name)
    for f in fields(MuscleMimicPpoCfg)
    if f.name not in {"class_name", "actor_bias_init", "bootstrap_time_outs"}
  }
  cfg.algorithm = SplitPpoCfg(**inherited)
  cfg.actor = MuscleMimicModelCfg(
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": INIT_STD[model],  # Replaced by the warm start's.
      "std_type": "log",
    }
  )
  cfg.critic = MuscleMimicModelCfg()
  cfg.osl_actor = OslActorModelCfg()
  cfg.osl_critic = MuscleMimicModelCfg()
  cfg.obs_groups = dict(OBS_GROUPS)
  cfg.max_iterations = STAGE2_ITERATIONS
  cfg.save_interval = SAVE_INTERVAL
  cfg.validation = SplitValidationCfg(
    interval=base.validation.interval,
    num_envs=base.validation.num_envs,
    num_steps=base.validation.num_steps,
    model=model,
    socket=socket,
    reward=reward,
    variant="unicycle",
  )
  start = _stage1_weights(reward)
  cfg.channels = SplitChannelsCfg(
    human_start=start,
    human_end=_stage1_weights(reward),
    osl_start=_stage1_weights(reward),
  )
  cfg.wandb_tags = ("osl_fullbody", model, reward, "unicycle", "split")
  return cfg
