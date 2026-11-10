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
* ``blend``: a ramp from the unified policy's OSL head (the teacher) to the OSL actor
  during the freeze (on by default);
* ``channels``: the two reward channels' coordinate-group weights over stage 2;
* ``validation``: the evaluation families, the zero-shot teacher check and the stress
  validation;
* ``resume_curriculum``: whether a resumed run keeps its checkpoint's curriculum or
  takes the command line's.

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
from mjlab.tasks.osl_split.rl.ppo import STAGE1_ONLY_KNOBS, check_stage1_knobs

STAGE2_ITERATIONS = 1000
"""Default stage-2 length: short first runs, to see the freeze and the unfreeze."""
SAVE_INTERVAL = 250
"""Stage-local iterations between checkpoints (about 0.7 GB each: the human actor and
two 24M-parameter critics with their optimizer states)."""
STRESS_INTERVAL = 120
"""Stage-local iterations between stress validations (stage 1's)."""
OSL_HIDDEN_DIMS = (256, 256, 256)
OSL_INIT_STD = 0.2
"""In normalized action units: 0.2 x 2.88 x 49.4/58.4 = about 28/34 N m of knee/ankle
torque noise while the fresh actor explores. The unified policy's OSL rows end stage
1 at 0.04-0.07, which the learned std falls towards."""
OSL_MIN_STD = 0.05
"""The OSL actor's std floor (rsl_rl's ``std_range``; there is no upper bound and no
std cap). With no entropy bonus stage 1's OSL std fell to 0.04-0.07 on its own; the
floor keeps a fresh, partially observed actor exploring, at about 7/8 N m of
knee/ankle noise, near the 6-9 N m the frozen human trained against."""
INHERITED_PPO_FIELDS = (
  "num_learning_epochs",
  "num_mini_batches",
  "gamma",
  "lam",
  "entropy_coef",
  "value_loss_coef",
  "use_clipped_value_loss",
  "clip_param",
  "max_grad_norm",
  "normalize_advantage_per_mini_batch",
  "muon_beta",
  "muon_nesterov",
  "muon_ns_steps",
  "muon_ns_dtype",
  "reward_normalization",
)
"""The stage-1 recipe's PPO settings stage 2 keeps. Every other algorithm field is
stage 2's own (:class:`SplitPpoCfg`)."""

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
  module), ``normalize_advantage_per_mini_batch``, the ``muon_*`` constants
  (``muon_ns_dtype`` included), ``reward_normalization`` and
  ``bootstrap_time_outs``; ``lr_final_ratio`` is the floor of stage 2's rate
  schedule.

  Stage-1 knobs stage 2 has no use for keep neutral values here, and any other value
  is refused when the config is built (on the command line, before the env is) and
  again by ``construct_algorithm``, naming what replaces it: ``learning_rate`` and
  the ``lr_schedule*`` settings (the per-module rates below), the ``std_cap_*``
  settings, ``std_floor`` and ``init_std_terms`` (the human's std stays the warm
  start's; the OSL actor's lives in ``osl_actor.distribution_cfg``),
  ``actor_bias_init`` (the warm start sets the human's output bias) and rsl_rl's
  ``schedule``, ``desired_kl``, ``optimizer`` and ``share_cnn_encoders``.

  Movement per iteration scales with ``num_mini_batches`` times the rate (one Muon
  step per minibatch, of a length the rate sets), so compare rates at a fixed
  ``num_mini_batches``.
  """

  class_name: str = "mjlab.tasks.osl_split.rl.ppo:TwoAgentPPO"
  bootstrap_time_outs: bool = True
  actor_bias_init: Literal["qpos0", "none"] = "none"
  """Refused in stage 2 (keep "none"): the warm start sets the human's output
  bias."""
  learning_rate: float = 0.0
  """Refused in stage 2 (keep 0): the per-module rates below replace it."""
  schedule: Literal["adaptive", "fixed"] = "fixed"
  """Refused in stage 2 (keep "fixed"): stage 2 has no KL-adaptive rate."""
  desired_kl: float = 0.01
  """Refused in stage 2 (keep 0.01): stage 2 has no KL-adaptive rate."""
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """Refused in stage 2 (keep "adam"): one MuonAdam per module (``muon_*``)."""
  share_cnn_encoders: bool = False
  """Refused in stage 2 (keep False): stage 2's networks are MLPs."""
  lr_schedule: Literal["linear", "constant"] = "linear"
  """Refused in stage 2 (keep "linear"): ``schedule_iterations``,
  ``lr_final_ratio`` and ``human_actor_lr_final_ratio`` set the stage-local
  schedule."""
  lr_schedule_updates: int = 0
  """Refused in stage 2 (keep 0): ``schedule_iterations`` sets the schedule's
  length."""
  lr_schedule_offset: int = 0
  """Refused in stage 2 (keep 0): ``--agent.resume-curriculum config`` gives a
  resumed run the command line's schedule."""
  std_cap_start: float = 0.0
  """Refused in stage 2 (keep 0), as are ``std_cap_end``, ``std_cap_begin_update``
  and ``std_cap_end_update``: the human's std stays the warm start's, and the OSL
  actor's range is ``osl_actor.distribution_cfg``."""
  std_cap_end: float = 0.0
  """Refused in stage 2 (keep 0): see ``std_cap_start``."""
  std_cap_begin_update: int = 0
  """Refused in stage 2 (keep 0): see ``std_cap_start``."""
  std_cap_end_update: int = 0
  """Refused in stage 2 (keep 0): see ``std_cap_start``."""
  std_floor: float = 0.0
  """Refused in stage 2 (keep 0): the OSL actor's floor is
  ``osl_actor.distribution_cfg``'s ``std_range``."""
  init_std_terms: dict[str, float] = field(default_factory=dict)
  """Refused in stage 2 (keep empty): the OSL actor's initial std is
  ``osl_actor.distribution_cfg``'s ``init_std``; the human's is the warm start's."""
  lr_final_ratio: float = 0.1
  """The critics' and the OSL actor's rates decay linearly to this fraction of their
  peak at the end of the schedule (``schedule_iterations``) and hold it beyond, so a
  run extended past it keeps learning."""
  muon_ns_dtype: Literal["float32", "bfloat16"] = "bfloat16"
  """Newton-Schulz precision of all four optimizers, the stage-1 recipe's: bfloat16
  for the musculoskeletal tasks (learning 7.2 -> 5.6 s per iteration there), float32
  for the robot ones."""

  human_critic_lr: float = 1e-4
  """The human critic's peak rate. Its target, the human channel, starts as the
  stage-1 reward it was trained on; stage 1's critics ended at 3e-5 to 5e-5. It
  decays with the stage but never below ``human_actor_peak_lr``."""
  osl_critic_lr: float = 3e-4
  """The OSL critic's peak rate: it follows a fresh OSL actor and the channel
  anneal."""
  osl_actor_lr: float = 4e-4
  """The OSL actor: this rate on the stage's decay, once it starts learning."""
  human_actor_peak_lr: float = 3e-5
  """The human actor after the unfreeze: a linear ramp from 0 to this, then
  ``human_actor_lr_final_ratio``. The rate stage 1's low-std (0.07) policies
  converged at: Muon's step size is set by the rate, not the gradient, so PPO's clip
  cannot limit it, and the clip fraction climbs with rate / std (stage 1 stayed
  below about 6e-4; 3e-5 / 0.07 is 4.3e-4). A clip fraction above about 0.6 means
  the human is moving too far."""
  human_actor_ramp_iterations: int = 50
  human_actor_lr_final_ratio: float = 1.0
  """Below 1 the human's rate decays linearly from its peak to this fraction of it
  at the end of the schedule; 1 holds the peak (stage 1's final phase trained at a
  constant rate)."""
  schedule_iterations: int = 0
  """The schedule's length ``S``, in stage-local iterations; 0 uses the run's
  ``max_iterations``. Kept in the checkpoint, so a resumed run keeps it unless it
  takes the config's curriculum (``resume_curriculum="config"``), where a non-zero
  value extends or shortens it."""

  osl_clip_param: float = 0.2
  osl_entropy_coef: float = 0.0
  osl_value_loss_coef: float = 0.25

  human_learn_std: bool = False
  """False keeps the human's std at its warm-start value for the whole stage (the
  user's decision): a std the human did not train with changes what its mean
  commands do. Stage-1 recipe runs end at 0.07, where the raw mean is the gait;
  lower a large std in stage 1, not here."""
  return_count_iterations: float = 3.0
  """A copied critic keeps the stage-1 return statistics it was trained on, with
  their count reset to this many iterations' samples so they can follow the stage-2
  reward channels (a stage-1 count of ~2e9 would freeze them)."""
  osl_critic_obs: Literal["full", "full+osl"] = "full"
  """``"full+osl"`` appends the OSL actor's observation to the OSL critic's input
  (zero-initialized weights, its own normalizer)."""
  osl_critic_init: Literal["stage1", "fresh"] = "stage1"
  """``"stage1"`` copies the stage-1 critic into the OSL critic. ``"fresh"`` starts
  it from random weights on the stage-1 observation statistics, so it may be built
  smaller (``--agent.osl-critic.hidden-dims``), a memory fallback. A fresh critic
  gets the longer critic-only warm-up
  (``freeze.reward_switch_critic_warmup_iterations``) to fit before the OSL actor
  learns from it. A resumed run must repeat both flags."""

  def __post_init__(self) -> None:
    # On the command line this fails before the env is built; construct_algorithm
    # checks the runner's dict again.
    check_stage1_knobs({name: getattr(self, name) for name in STAGE1_ONLY_KNOBS})


@dataclass
class WarmStartCfg:
  path: str = ""
  """A stage-1 run directory, a stage-1 checkpoint (``model_N.pt``), or a warm-start
  file written by ``python -m mjlab.tasks.osl_split.scripts.split``. Required to
  train; ``play`` loads stage-2 checkpoints instead."""
  checkpoint: str = ""
  """Checkpoint file inside a run directory; empty picks the highest-numbered
  ``model_N.pt``."""
  allow_pen_w_mismatch: bool = False
  """Start even though stage 1 trained with another CLF penalty weight (``pen_w``)
  than stage 2's: the copied critics and return statistics then value another
  reward. Refused otherwise."""


@dataclass
class SplitFreezeCfg:
  human_actor_iterations: int = 200
  """Stage-local iterations with the human actor frozen (weights, std and observation
  statistics). Set from the first curves."""
  critic_warmup_iterations: int = 10
  """Stage-local iterations with only the critics learning: the OSL actor acts with
  its initial policy (or the blend) until then. The critics' rates ramp up over
  it."""
  reward_switch_critic_warmup_iterations: int = 50
  """The critic-only warm-up when a critic starts fresh: when stage 2's reward type
  differs from stage 1's (both critics) or under
  ``algorithm.osl_critic_init="fresh"`` (the OSL critic)."""


@dataclass
class SplitBlendCfg:
  enabled: bool = True
  """Drive the OSL with ``(1 - beta) * teacher + beta * osl_actor``, where the teacher
  is the unified policy's OSL head on the frozen human trunk. ``beta`` ramps from 0
  to 1 starting when the OSL actor starts learning; it must reach 1 by the human
  unfreeze, after which the trunk (and so the teacher) changes. Without it the frozen
  human starts on a limp prosthesis (the fresh actor commands about 0.6 N m). PPO
  learns from the OSL actor's own sample, and nothing pulls its mean towards the
  teacher, so the handover at ``beta = 1`` can still be abrupt."""
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
  anneal_start_iteration: int = 10
  """A fixed stage-local iteration: 10 is the default critic warm-up's end, so the
  OSL actor gets its joint coordinates while the human is frozen. It does not follow
  the warm-up: under a longer one (a reward switch, a fresh OSL critic) the anneal
  starts while the OSL actor still waits. -1: at the human unfreeze
  (``freeze.human_actor_iterations``)."""
  anneal_iterations: int = 150
  """Done (at iteration 160) before the human unfreezes at 200, so the anneal, the
  unfreeze and the human's normalizer starting to move do not land together."""
  additive_term_weights: dict[str, tuple[float, float]] = field(default_factory=dict)
  """``{term: (human, osl)}`` multipliers on the env's additive reward terms (every
  term but ``mimic``, e.g. ``twist``); a term not listed counts fully in both."""


@dataclass
class SplitValidationCfg(GaitLibraryValidationCfg):
  """Stage 2's in-training evaluations (:mod:`mjlab.tasks.osl_split.rl.runner`).

  Each evaluation plays one or more *families*, each under its own prefix:

  * ``"mean"``: the human's raw mean and the OSL actor's mean (``Validation/``,
    ``Stress/``), the deployment metric;
  * ``"expected"``: the human's noise-averaged action ``E[clip(mu + sigma eps)]``
    and the OSL actor's mean (``ValidationExpected/``, ``StressExpected/``); off by
    default (at the recipe's human std of 0.07 it is nearly ``"mean"``);
  * ``"osl_sampled"``: the human's raw mean and the OSL actor's sampled action,
    seeded (``ValidationOslSampled/``, ``StressOslSampled/``). Its gap to
    ``"mean"`` shows an OSL actor that relies on its own action noise, which no
    policy statistic shows (the OSL torque is rarely clipped, so
    ``Policy/noise_gap/osl_motors`` stays near 0); stress validation plays it.

  The zero-shot evaluation, before the first update, plays the fresh OSL actor
  (``Validation/`` there is the floor it starts from) and the *teacher*: the
  unified stage-1 policy's OSL head on the human trunk, which with the human is the
  stage-1 policy itself, raw mean (``ValidationTeacher/``, ``StressTeacher/``) and
  noise-averaged on every row as stage 1 averages it (``ValidationTeacherExpected/``).
  They should match the stage-1 checkpoint's own ``Validation/`` and
  ``ValidationExpected/``, which shows that the stage-2 env reproduces stage 1, and
  their gap is the warm start's own noise reliance.
  """

  inference: tuple[str, ...] = ("mean",)
  """Families every ``interval`` stage-local iterations (see the class docstring).
  On the command line, a Python tuple: ``--agent.validation.inference
  "('mean','osl_sampled')"``."""
  zero_shot_inference: tuple[str, ...] = ("mean",)
  """Families at the zero-shot validation; the teacher's come on top
  (``teacher``). The OSL actor is fresh there, so any family of it fails at the
  floor."""
  teacher: bool = True
  """Play the teacher families at the zero-shot validation (``ValidationTeacher/``,
  ``ValidationTeacherExpected/``) and stress validation (``StressTeacher/``)."""
  zero_shot: bool = True
  """Validate (and stress-validate) the split policies once before stage 2's first
  update."""
  seed: int = 0
  """Seeds the ``osl_sampled`` validation family, so every validation draws the same
  OSL action noise. Training's random state is left untouched."""
  stress_interval: int = STRESS_INTERVAL
  """Stage-local iterations between stress validations; 0 disables them (and the
  zero-shot one). Stress validation plays a copy of the training config, OSL sensor
  noise and every randomization included, with 512 envs (``stress_num_envs``),
  ending episodes only on validation's 0.5 m site check, and reports each env's first
  episode: stage 1's best predictor of deployment, and the only one for the OSL actor
  until osl_deploy runs a split pair. Its physical draws follow ``stress_seed``, not
  the run's seed."""
  stress_inference: tuple[str, ...] = ("mean", "osl_sampled")
  """Families stress-validated every ``stress_interval`` iterations (see the class
  docstring): the deployment proxy and the OSL actor's noise reliance."""
  zero_shot_stress_inference: tuple[str, ...] = ()
  """Families at the zero-shot stress validation; the teacher's comes on top
  (``teacher``). The OSL actor is fresh there, so none by default."""


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
  """The OSL critic: a copy of the stage-1 critic, so stage 1's architecture (any,
  with ``algorithm.osl_critic_init="fresh"``)."""
  algorithm: SplitPpoCfg = field(default_factory=SplitPpoCfg)  # pyright: ignore
  validation: SplitValidationCfg = field(default_factory=SplitValidationCfg)
  video: OslMimicVideoCfg = field(default_factory=OslMimicVideoCfg)
  warm_start: WarmStartCfg = field(default_factory=WarmStartCfg)
  freeze: SplitFreezeCfg = field(default_factory=SplitFreezeCfg)
  blend: SplitBlendCfg = field(default_factory=SplitBlendCfg)
  channels: SplitChannelsCfg = field(default_factory=SplitChannelsCfg)
  resume_curriculum: Literal["checkpoint", "config"] = "checkpoint"
  """On ``--agent.resume``: ``"checkpoint"`` continues on the curriculum the
  checkpoint trained with (schedule, freeze, blend, channels, additive weights),
  whatever the command line says, so a resume cannot freeze an unfrozen human again;
  ``"config"`` takes the command line's, to retune or extend a run (set
  ``--agent.algorithm.schedule-iterations`` to lengthen the schedule; 0 keeps the
  checkpoint's). Either way the differences are printed."""
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
    # Stage 2 starts where stage 1 ended, on the env config's final values, so it
    # takes none of stage 1's env stages.
    if f.name not in {"algorithm", "validation", "env_stages"}
  }
  cfg = SplitRunnerCfg(**shared)
  assert isinstance(base.algorithm, MuscleMimicPpoCfg)
  cfg.algorithm = SplitPpoCfg(
    **{name: getattr(base.algorithm, name) for name in INHERITED_PPO_FIELDS}
  )
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
