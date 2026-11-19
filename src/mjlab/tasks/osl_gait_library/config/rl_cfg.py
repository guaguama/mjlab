"""Runner settings: the matching osl_mimic task's, plus time-out bootstrapping, and
for the unicycle tasks the stage-1 recipe of their model: MSK's
(:func:`apply_unicycle_msk_recipe`) and the robot's
(:func:`apply_unicycle_robot_recipe`)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.config.rl_cfg import (
  HIDDEN_DIMS,
  NUM_STEPS,
  MuscleMimicPpoCfg,
  OslMimicRunnerCfg,
  OslMimicValidationCfg,
  osl_mimic_runner_cfg,
)

from .env_cfgs import VALIDATION_EPISODE_LENGTH_S, Variant

# ===== The unicycle MSK recipe (tuned on -MSK-CLF-Compliant, 2026-11) =============
# musclemimic's muscle std of 3.0 with no bound let the 2026-11-06 run settle at a
# std of 2.1 whose gait needed the action noise: its raw mean fell within seconds
# and only the noise-averaged action walked. The recipe starts lower, anneals the
# std below 0.1 so the raw mean is the gait, and lowers the rate with it, since PPO's
# clip fraction climbs as the std falls.
UNICYCLE_MSK_INIT_STD = 1.0
"""The muscles' initial action std; musclemimic's 3.0 matched it in training and
lagged on the raw mean."""
UNICYCLE_MSK_INIT_STD_TERMS = {"osl_motors": 0.3}
"""The OSL motors' initial std, on their [-1, 1] targets."""
UNICYCLE_MSK_STD_CAP = (1.0, 0.07)
"""The std cap at its first and last update, geometric in between."""
UNICYCLE_MSK_STD_CAP_UPDATES = (300, 4000)
"""The updates where the cap starts falling and where it reaches its end value."""
UNICYCLE_MSK_LR_UPDATES = 4000
"""The 4e-4 rate decays linearly over this many updates, then holds."""
UNICYCLE_MSK_LR_FINAL_RATIO = 0.1
UNICYCLE_MSK_ITERATIONS = 4500
UNICYCLE_MSK_VALIDATION_INTERVAL = 60
UNICYCLE_MSK_STRESS_INTERVAL = 120
"""Stress validation (the training distribution, raw mean and noise-averaged
action, 512 envs) predicts deployment best; validation runs the nominal plant."""
UNICYCLE_MSK_ENV_STAGES = (
  (1500, "terminations.root_deviation.threshold", 1.0),
  (3000, "rewards.twist.weight", 0.06),
)
"""The env config holds the recipe's final drift limit (2 m) and twist weight (0.2);
training reaches them in stages, as the tuned run did. From the first update, 2 m let
the policy trail its reference (forward speed error 0.17 vs 0.13 m/s at update 1400)
and fall more often; switched on at 1500 it taught recovery. The 0.2 twist weight
paid off once the std was small (tight-arc failures 0.05 vs 0.19 at update 3800)."""

# ===== The unicycle robot recipe (round 7, 2026-11; not yet tuned on the robot) ====
# The 27 position servos and 2 OSL motors learn in fewer samples than the muscles
# (the legacy 8192-env robot run, W&B h60d0llm, validated cleanly at ~0.45-0.5x the
# MSK teacher's samples), so the robot trains 2000 iterations at 8192 envs (1.3e9
# samples) in two phases: library sampling to update 1500, then continuous
# tight-arc sampling at the held rate (a resume with the sampling flags, as the MSK
# teacher's last phase). Every count here is set for 8192 envs; none derives from
# osl_mimic's ``NUM_UPDATES`` (4096 envs). Under Muon a step's length is set by the
# rate, so per-iteration movement is the rate times ``num_mini_batches``, whatever
# the batch size.
UNICYCLE_ROBOT_ITERATIONS = 2000
"""The run's last checkpoint is model_1999: phase A trains the first
``UNICYCLE_ROBOT_PHASE_A_ITERATIONS`` (``--agent.max-iterations 1500``), phase B
resumes from model_1499 for the remaining 501 (an rsl_rl resume counts its
iterations from the checkpoint's)."""
UNICYCLE_ROBOT_PHASE_A_ITERATIONS = 1500
"""Phase A's length: where the rate reaches its floor and the std cap its end."""
UNICYCLE_ROBOT_LR = 4e-4
UNICYCLE_ROBOT_LR_UPDATES = 1500
"""The rate decays linearly to ``UNICYCLE_ROBOT_LR_FINAL_RATIO`` of itself (4e-5)
over this many updates, then holds through phase B (musclemimic's decay to 0 left
the legacy run learning almost nothing in its last ~90 iterations)."""
UNICYCLE_ROBOT_LR_FINAL_RATIO = 0.1
UNICYCLE_ROBOT_INIT_STD = 0.2
"""Every row's initial std (servos and OSL motors alike): osl_mimic's robot value."""
UNICYCLE_ROBOT_STD_CAP = (0.2, 0.07)
"""The std cap at its first and last update, geometric in between. It binds early on
the arms' high-std rows and makes the teacher end at the std stage 2 assumes: stage 2
holds the human's std fixed, and its human rate (3e-5) was set for 0.07."""
UNICYCLE_ROBOT_STD_CAP_UPDATES = (200, 1500)
UNICYCLE_ROBOT_STD_FLOOR = 0.03
"""A floor under every row (at most the cap's end): uncapped, a robot run's leg rows
fell to 0.021, and stage 2 freezes whatever the teacher ends with."""
UNICYCLE_ROBOT_NUM_MINI_BATCHES = 128
"""5120-sample minibatches at 8192 envs x 80 steps. Stage 2 inherits it, and its
rates were set for 128 Muon steps per iteration: keep it."""
UNICYCLE_ROBOT_SAVE_INTERVAL = 250
UNICYCLE_ROBOT_VALIDATION_INTERVAL = 125
"""Lean evaluation: a robot validation pass costs about 2 min, so validating every
30 iterations (osl_mimic's) spent ~17-20% of the wall time."""
UNICYCLE_ROBOT_STRESS_INTERVAL = 250
UNICYCLE_ROBOT_INFERENCE = ("mean",)
"""Validation and stress validation play the raw mean: the robot deploys it (the
robot's ``auto`` inference is the mean)."""
UNICYCLE_ROBOT_ENV_STAGES = (
  (400, "terminations.root_deviation.threshold", 1.0),
  (1200, "rewards.twist.weight", 0.06),
)
"""The robot recipe's stages, after the CLF tasks' ramps (0 until update 500, half
until 1000), never instead of them: the env config holds the final drift limit
(2 m) and twist weight (0.2), reached as the MSK teacher reached them. Offset from
the ramps' steps, so no two changes land on one update (each would be blurred in
the other's ``Stages/`` curves)."""


@dataclass
class GaitLibraryValidationCfg(OslMimicValidationCfg):
  variant: str = "walk_fwd"
  """The task variant whose validation env the runner builds."""


@dataclass
class GaitLibraryRunnerCfg(OslMimicRunnerCfg):
  # A narrower type than the base field's (pyright's
  # reportIncompatibleVariableOverride), so tyro and the runner see ``variant``.
  validation: GaitLibraryValidationCfg = field(  # pyright: ignore
    default_factory=GaitLibraryValidationCfg
  )


def osl_gait_library_runner_cfg(
  model: Model,
  socket: Socket,
  reward: Reward,
  experiment_name: str,
  variant: Variant = "walk_fwd",
  robot_recipe: bool = True,
) -> GaitLibraryRunnerCfg:
  """The runner settings of a gait-library task.

  Args:
    model, socket, reward, experiment_name: As ``osl_mimic_runner_cfg``.
    variant: ``"walk_fwd"`` or ``"unicycle"``; the unicycle tasks get their model's
      stage-1 recipe.
    robot_recipe: False keeps the robot's unicycle settings as they were before its
      recipe (osl_mimic's), for stage-2 tasks whose stage 1 is not the robot recipe
      (``osl_split_runner_cfg(..., robot_recipe=False)``). The MSK recipe applies
      either way.
  """
  base = osl_mimic_runner_cfg(model, socket, reward, experiment_name)
  cfg = GaitLibraryRunnerCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
  assert isinstance(cfg.algorithm, MuscleMimicPpoCfg)
  # Library episodes are cut off by the 10 s limit, not finished, so the critic
  # bootstraps there (rsl_rl's default; musclemimic's clips simply end).
  cfg.algorithm.bootstrap_time_outs = True
  cfg.validation = GaitLibraryValidationCfg(
    interval=base.validation.interval,
    num_envs=0,  # One env per library clip.
    # One validation episode (20 s at 100 Hz); stages 1.5 and 2 inherit it.
    num_steps=round(VALIDATION_EPISODE_LENGTH_S * 100),
    model=model,
    socket=socket,
    reward=reward,
    inference=base.validation.inference,
    variant=variant,
  )
  library_tag = "unicycle" if variant == "unicycle" else "gait_library"
  cfg.wandb_tags = ("osl_fullbody", model, reward, library_tag)
  if variant == "unicycle" and model == "msk":
    apply_unicycle_msk_recipe(cfg)
  elif variant == "unicycle" and model == "robot" and robot_recipe:
    apply_unicycle_robot_recipe(cfg)
  return cfg


def apply_unicycle_msk_recipe(cfg: GaitLibraryRunnerCfg) -> None:
  """Set the stage-1 recipe's PPO schedule and validation cadence, and add its env
  stages to the config's own (the CLF tasks' reward ramps)."""
  alg = cfg.algorithm
  assert isinstance(alg, MuscleMimicPpoCfg)
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = UNICYCLE_MSK_INIT_STD
  alg.init_std_terms = dict(UNICYCLE_MSK_INIT_STD_TERMS)
  alg.std_cap_start, alg.std_cap_end = UNICYCLE_MSK_STD_CAP
  alg.std_cap_begin_update, alg.std_cap_end_update = UNICYCLE_MSK_STD_CAP_UPDATES
  alg.lr_schedule_updates = UNICYCLE_MSK_LR_UPDATES
  alg.lr_final_ratio = UNICYCLE_MSK_LR_FINAL_RATIO
  # Learning 7.2 -> 5.6 s per iteration, with no measurable change in training.
  alg.muon_ns_dtype = "bfloat16"
  cfg.max_iterations = UNICYCLE_MSK_ITERATIONS
  cfg.save_interval = 500
  cfg.validation.interval = UNICYCLE_MSK_VALIDATION_INTERVAL
  cfg.validation.stress_interval = UNICYCLE_MSK_STRESS_INTERVAL
  cfg.validation.stress_inference = ("mean", "expected")
  cfg.env_stages = (*cfg.env_stages, *UNICYCLE_MSK_ENV_STAGES)


def apply_unicycle_robot_recipe(cfg: GaitLibraryRunnerCfg) -> None:
  """Set the robot recipe's PPO schedule, networks and evaluation cadence, every
  value explicitly, and add its env stages to the config's own (the CLF tasks'
  reward ramps; none under MimicReward)."""
  alg = cfg.algorithm
  assert isinstance(alg, MuscleMimicPpoCfg)
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = UNICYCLE_ROBOT_INIT_STD
  for model_cfg in (cfg.actor, cfg.critic):
    model_cfg.hidden_dims = HIDDEN_DIMS
  alg.init_std_terms = {}
  alg.std_cap_start, alg.std_cap_end = UNICYCLE_ROBOT_STD_CAP
  alg.std_cap_begin_update, alg.std_cap_end_update = UNICYCLE_ROBOT_STD_CAP_UPDATES
  alg.std_floor = UNICYCLE_ROBOT_STD_FLOOR
  alg.actor_bias_init = "qpos0"
  alg.learning_rate = UNICYCLE_ROBOT_LR
  alg.lr_schedule = "linear"
  alg.lr_schedule_updates = UNICYCLE_ROBOT_LR_UPDATES
  alg.lr_schedule_offset = 0
  alg.lr_final_ratio = UNICYCLE_ROBOT_LR_FINAL_RATIO
  alg.muon_ns_dtype = "bfloat16"
  alg.num_mini_batches = UNICYCLE_ROBOT_NUM_MINI_BATCHES
  cfg.num_steps_per_env = NUM_STEPS
  cfg.max_iterations = UNICYCLE_ROBOT_ITERATIONS
  cfg.save_interval = UNICYCLE_ROBOT_SAVE_INTERVAL
  cfg.validation.interval = UNICYCLE_ROBOT_VALIDATION_INTERVAL
  cfg.validation.inference = UNICYCLE_ROBOT_INFERENCE
  cfg.validation.stress_interval = UNICYCLE_ROBOT_STRESS_INTERVAL
  cfg.validation.stress_inference = UNICYCLE_ROBOT_INFERENCE
  cfg.video.enabled = True
  cfg.env_stages = (*cfg.env_stages, *UNICYCLE_ROBOT_ENV_STAGES)
