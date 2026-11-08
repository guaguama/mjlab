"""Evaluate a gait-library checkpoint on the GPU with the runner's own validations.

::

  uv run --no-sync python -m mjlab.tasks.osl_gait_library.evaluate <model_N.pt> \\
    [--inference "('mean','expected')"] [--stress-envs 512] [--library DIR] \\
    [--json-out results.json]

(mjlab's command line takes collections as quoted Python literals.)

The run's ``params/agent.yaml`` names its model, socket, reward and variant; the
task is rebuilt from code (as the exporter does), the checkpoint is loaded into the
task's runner, and the runner's passes run once per inference mode:

* validation (``Validation*/``): every library clip in its own env from phase 0 at
  its own twist, on the nominal plant with mid-range actuator delays, for one
  episode, ending early only on the 0.5 m site check;
* stress validation (``Stress*/``): each env's first episode of the training
  distribution (random clips and switches, physical draws, pushes, random delays),
  with the same failure check.

Results print as one JSON object; nothing is logged to TensorBoard or W&B.

``--legacy-contacts`` evaluates a checkpoint trained before 2026-11-07 under the
contact physics it trained with: the pyramidal cone, impratio 1, the OSL foot shell
colliding and no pelvis-socket exclusion.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

from .export import config_mismatches, read_config, run_spec


@dataclass
class EvaluateConfig:
  checkpoint: tyro.conf.Positional[Path]
  """``model_<N>.pt`` of a gait-library run (its run directory holds ``params/``)."""
  inference: tuple[str, ...] = ("mean", "expected")
  """Deterministic actions to evaluate: ``mean``, ``expected`` and/or ``sampled``."""
  validation: bool = True
  """Run the per-clip validation."""
  validation_steps: int = 1000
  stress_envs: int = 512
  """Envs of the stress validation; 0 skips it."""
  stress_steps: int = 1000
  stress_seed: int = 0
  library: Path | None = None
  """Clip folder; default the run's own ``library_dir`` (or the task's env var)."""
  legacy_contacts: bool = False
  """Pre-2026-11-07 contact physics (see the module docstring)."""
  solver_iterations: int | None = None
  """Override the solver's iterations (and ``ls_solver_iterations`` its line search)
  in every evaluation env, e.g. to probe how far a policy depends on truncated
  contact solves."""
  ls_solver_iterations: int | None = None
  json_out: Path | None = None
  device: str = "cuda:0"


LEGACY_FOOT_GEOMS = (
  r"^(foot_col[134]_l|bofoot_col[12]_l|OSL_KA_L1_osl_foot_col[123]"
  r"|OSL_KA_L1_osl_foot_assembly_geom_1)$"
)
"""The foot geoms of runs before 2026-11-07: the OSL foot shell mesh included."""


def use_legacy_contacts(env_cfg: Any) -> None:
  """Contacts of runs before 2026-11-07, for every env built from now on: the OSL
  foot shell collides (with the feet's contact priority and friction draws), the
  pelvis-socket pair is not excluded, and the cone is pyramidal at impratio 1. The
  evaluation envs inherit the solver settings from ``env_cfg``."""
  from mjlab.asset_zoo.robots.osl_fullbody import osl_constants
  from mjlab.tasks.osl_gait_library.config import dr

  osl_constants._fix_osl_contacts = lambda spec: None  # type: ignore[assignment]
  # Read by the priority spec edit at build time.
  dr.FOOT_GEOMS = LEGACY_FOOT_GEOMS  # type: ignore[assignment]
  env_cfg.sim.mujoco.cone = "pyramidal"
  env_cfg.sim.mujoco.impratio = 1.0
  friction = env_cfg.events.get("foot_friction")
  if friction is not None:
    friction.params["asset_cfg"].geom_names = (LEGACY_FOOT_GEOMS,)


def evaluate(cfg: EvaluateConfig) -> dict[str, float]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.osl_gait_library.config.env_cfgs import (
    LIBRARY_ENV_VAR,
    UNICYCLE_LIBRARY_ENV_VAR,
  )
  from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
  from mjlab.tasks.osl_gait_library.rl.runner import GaitLibraryOnPolicyRunner
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  checkpoint = cfg.checkpoint.resolve()
  params = checkpoint.parent / "params"
  agent = read_config(params / "agent.yaml")
  spec = run_spec(checkpoint, agent)
  task_id = spec.task_id
  if task_id is None:
    raise ValueError(f"No registered task matches the run: {spec}")

  library = cfg.library
  if library is None:
    saved = read_config(params / "env.yaml")
    saved_dir = saved.get("commands", {}).get("motion", {}).get("library_dir", "")
    env_var = (
      UNICYCLE_LIBRARY_ENV_VAR if spec.variant == "unicycle" else LIBRARY_ENV_VAR
    )
    library = Path(saved_dir) if saved_dir else Path(os.environ.get(env_var, ""))
  if not library.is_dir():
    raise FileNotFoundError(f"Library folder {library} not found; pass --library.")
  if spec.variant == "unicycle":
    os.environ[UNICYCLE_LIBRARY_ENV_VAR] = str(library)
  else:
    os.environ[LIBRARY_ENV_VAR] = str(library)

  configure_torch_backends()
  # The runner's evaluation envs take their simulation settings from this training
  # config (and stress validation plays a copy of it), so every override goes here.
  env_cfg = load_env_cfg(task_id)
  if cfg.legacy_contacts:
    use_legacy_contacts(env_cfg)
  if cfg.solver_iterations is not None:
    env_cfg.sim.mujoco.iterations = cfg.solver_iterations
  if cfg.ls_solver_iterations is not None:
    env_cfg.sim.mujoco.ls_iterations = cfg.ls_solver_iterations
  motion = env_cfg.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = str(library)
  for issue in config_mismatches(env_cfg, read_config(params / "env.yaml")):
    print(f"[WARN] Differs from the run's env.yaml: {issue}", file=sys.stderr)
  env_cfg.scene.num_envs = 2  # The runner needs a training env; the evals own theirs.

  agent_cfg = asdict(load_rl_cfg(task_id))
  agent_cfg["logger"] = "tensorboard"
  agent_cfg["video"] = {**agent_cfg.get("video", {}), "enabled": False}
  # The checkpoint's own std, untouched by any training-time bound.
  agent_cfg["algorithm"].update(
    std_cap_start=0.0, std_cap_end=0.0, std_floor=0.0, init_std_terms={}
  )
  validation = agent_cfg["validation"]
  validation.update(
    inference=list(cfg.inference),
    num_steps=cfg.validation_steps,
    stress_num_envs=cfg.stress_envs,
    stress_num_steps=cfg.stress_steps,
    stress_inference=list(cfg.inference),
    stress_seed=cfg.stress_seed,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  vec = RslRlVecEnvWrapper(env)
  runner = GaitLibraryOnPolicyRunner(vec, agent_cfg, log_dir=None, device=cfg.device)
  runner.load(str(checkpoint), map_location=cfg.device)
  iteration = int(runner.current_learning_iteration)

  results: dict[str, float] = {"iteration": float(iteration)}
  try:
    if cfg.validation:
      results.update(runner.validate(iteration))
    if cfg.stress_envs > 0:
      results.update(runner.stress_validate(iteration))
  finally:
    runner.close()
    vec.close()
  return results


def main() -> None:
  import mjlab

  cfg = tyro.cli(EvaluateConfig, config=mjlab.TYRO_FLAGS)
  torch.set_num_threads(2)
  results = evaluate(cfg)
  text = json.dumps(results, indent=1, sort_keys=True)
  if cfg.json_out is not None:
    cfg.json_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.json_out.write_text(text)
  print(text)
  sys.stdout.flush()


if __name__ == "__main__":
  main()
