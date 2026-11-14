"""Runner settings: the matching osl_mimic task's, plus time-out bootstrapping, and
for the unicycle MSK tasks the stage-1 recipe (:func:`apply_unicycle_msk_recipe`)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.config.rl_cfg import (
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
) -> GaitLibraryRunnerCfg:
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
  return cfg


def apply_unicycle_msk_recipe(cfg: GaitLibraryRunnerCfg) -> None:
  """Set the stage-1 recipe's PPO schedule, env stages and validation cadence."""
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
  cfg.env_stages = UNICYCLE_MSK_ENV_STAGES
