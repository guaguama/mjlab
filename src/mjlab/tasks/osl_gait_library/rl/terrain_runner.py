"""The terrain tasks' runners: the gait-library (stage 1) runner on generated terrain.

:class:`TerrainRunnerMixin` adds to a gait-library runner (stage 1's, and stage 2's
in :mod:`mjlab.tasks.osl_split.rl.terrain_runner`):

* **checkpoints** record every env's terrain level and type, with the terrain
  spec's digest (``infos["terrain"]``, :func:`~..terrain.terrain_state`), and a
  training resume restores them as it starts training (then restarts every
  episode on its restored tile); a checkpoint without the record (a flat-ground run
  forked onto terrain) leaves the envs at their random start levels, and says so;
* **stress validation** is frozen: a copy of the training config on the stress
  ground (:data:`~..terrain.STRESS_TERRAIN_SEED`), without the curriculum, its envs
  laid out uniformly over every type and level
  (:func:`~..terrain.uniform_levels`), and reports survival per band of levels,
  type and both (``Stress/survival_rate/<type>_<band>``, ``.../<band>``,
  ``.../<type>``; ``mean_first_episode_s`` likewise; bands
  :data:`~..terrain.STRESS_BANDS`);
* **validation stays flat**: the task's validation mode, as the flat runs';
* **videos** play the terrain in play mode (a random level, seeded);
* the run's terrain spec is written to ``params/terrain_generator.json`` as
  training starts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.osl_gait_library import terrain as osl_terrain
from mjlab.tasks.osl_mimic.rl.runner import EvalPurpose, resumes_training

from .runner import GaitLibraryOnPolicyRunner

TERRAIN_CURRICULUM = "terrain_levels"
"""The curriculum term's name (``config.env_cfgs.TERRAIN_CURRICULUM``)."""


def generator_of(cfg: Any) -> Any:
  """An env config's terrain generator config (it must have one)."""
  if not osl_terrain.is_terrain_cfg(cfg):
    raise ValueError("Not a terrain env config: its scene has no generated terrain.")
  return cfg.scene.terrain.terrain_generator


def freeze_terrain(
  cfg: ManagerBasedRlEnvCfg, seed: int = osl_terrain.STRESS_TERRAIN_SEED
) -> ManagerBasedRlEnvCfg:
  """Make a copy of a terrain training config stress validation's, in place: the
  stress ground (``seed``), no level curriculum, envs at any level (the runner
  then lays them out, :func:`~..terrain.uniform_levels`)."""
  generator_of(cfg).seed = seed
  terrain = cfg.scene.terrain
  assert terrain is not None
  terrain.max_init_terrain_level = None
  cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k != TERRAIN_CURRICULUM}
  return cfg


def curriculum_term(env: ManagerBasedRlEnv) -> Any:
  """The env's level curriculum term, or None."""
  manager = env.curriculum_manager
  if not isinstance(manager, CurriculumManager) and not hasattr(
    manager, "get_term_cfg"
  ):
    return None
  if TERRAIN_CURRICULUM not in getattr(manager, "active_terms", []):
    return None
  return cast(CurriculumManager, manager).get_term_cfg(TERRAIN_CURRICULUM).func


def level_summary(terrain: Any) -> str:
  """Every type's mean level, e.g. ``rough 2.51, waves 2.47, obstacles 2.55``."""
  names = list(terrain.cfg.terrain_generator.sub_terrains)
  levels = terrain.terrain_levels.float()
  parts = []
  for i, name in enumerate(names):
    mask = terrain.terrain_types == i
    if bool(mask.any()):
      parts.append(f"{name} {float(levels[mask].mean()):.2f}")
  return ", ".join(parts)


class TerrainRunnerMixin:
  """Terrain levels in checkpoints, frozen stress validation with level bands,
  terrain videos; see the module docstring. Mix it in before a gait-library
  runner."""

  env: RslRlVecEnvWrapper
  logger: Any

  ##
  # The terrain.
  ##

  def terrain_spec(self) -> dict[str, Any]:
    """The training env's terrain spec (:func:`~..terrain.terrain_spec`)."""
    return osl_terrain.terrain_spec(generator_of(self.env.unwrapped.cfg))

  def _training_terrain(self) -> Any:
    terrain = self.env.unwrapped.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("A terrain runner needs a training env on generated terrain.")
    return terrain

  def save_terrain_spec(self) -> Path | None:
    """Write the training env's spec to ``<log_dir>/params/terrain_generator.json``
    (main process only); returns the file."""
    log_dir = getattr(self.logger, "log_dir", None)
    if log_dir is None or getattr(self.logger, "disable_logs", False):
      return None
    path = Path(log_dir) / "params" / osl_terrain.SPEC_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(osl_terrain.spec_json(self.terrain_spec()))
    return path

  ##
  # Training and checkpoints.
  ##

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Restore the levels of the checkpoint a training resume loaded
    (:meth:`restore_terrain`), write the spec, then the runner's training."""
    pending = getattr(self, "_terrain_pending", None)
    self._terrain_pending = None
    if pending is not None:
      self.restore_terrain(*pending)
    self.save_terrain_spec()
    print(
      f"[INFO] Terrain levels at the start: {level_summary(self._training_terrain())}",
      flush=True,
    )
    super().learn(num_learning_iterations, init_at_random_ep_len)  # type: ignore[misc]

  def save(self, path: str, infos: dict | None = None) -> None:
    """The runner's checkpoint plus the terrain record (``infos["terrain"]``)."""
    record = osl_terrain.terrain_state(self._training_terrain(), self.terrain_spec())
    infos = {**(infos or {}), osl_terrain.CHECKPOINT_KEY: record}
    super().save(path, infos)  # type: ignore[misc]

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """The runner's load. A load of what a training resume needs also notes the
    checkpoint's terrain record, which :meth:`learn` restores before it trains
    (:meth:`restore_terrain`): an evaluation loads the whole checkpoint too, on envs
    of its own."""
    infos = super().load(path, load_cfg, strict, map_location)  # type: ignore[misc]
    self._terrain_pending = (infos, path) if resumes_training(load_cfg) else None
    return infos

  def restore_terrain(self, infos: Mapping[str, Any], path: str) -> bool:
    """Put the training envs on the levels and types ``infos`` records, and restart
    every episode there (a reset whose curriculum step changes nothing). Returns
    whether it restored them. A record of other ground or another env count draws
    a warning; no record (a flat run) an info line; both leave the levels as they
    are (random ``0..max_init_terrain_level``)."""
    env = self.env.unwrapped
    term = curriculum_term(env)
    if term is None:
      return False
    terrain = self._training_terrain()
    record = infos.get(osl_terrain.CHECKPOINT_KEY)
    if not isinstance(record, Mapping):
      print(
        f"[INFO] {path} records no terrain levels (a flat-ground run): the envs keep "
        "their random start levels 0.."
        f"{terrain.cfg.max_init_terrain_level} ({level_summary(terrain)}).",
        flush=True,
      )
      return False
    problems = osl_terrain.restore_problems(record, terrain, self.terrain_spec())
    if problems:
      print(
        f"[WARN] Terrain levels not restored from {path}: {'; '.join(problems)}. "
        f"The envs keep their random start levels ({level_summary(terrain)}).",
        flush=True,
      )
      return False
    osl_terrain.set_levels(
      terrain, torch.as_tensor(record["levels"]), torch.as_tensor(record["types"])
    )
    term.freeze_next()
    self.env.reset()
    print(
      f"[INFO] Terrain levels restored from {path}: {level_summary(terrain)}.",
      flush=True,
    )
    return True

  ##
  # Evaluation envs.
  ##

  def _eval_env_cfg(self, purpose: EvalPurpose) -> ManagerBasedRlEnvCfg:
    """The runner's evaluation config; stress validation's is then frozen
    (:func:`freeze_terrain`), and a video's play config put on the training
    ground (a random level)."""
    from mjlab.tasks.osl_gait_library.config.env_cfgs import apply_terrain

    cfg = super()._eval_env_cfg(purpose)  # type: ignore[misc]
    if purpose == "stress":
      freeze_terrain(cfg)
    elif purpose == "video":
      apply_terrain(cfg, "play", seed=int(generator_of(self.env.unwrapped.cfg).seed))
    return cfg

  def _make_stress_env(self) -> RslRlVecEnvWrapper:
    """The runner's stress env, every env on its fixed (type, level)
    (:func:`~..terrain.uniform_levels`)."""
    vec = super()._make_stress_env()  # type: ignore[misc]
    terrain = vec.unwrapped.scene.terrain
    if terrain is not None and terrain.terrain_origins is not None:
      levels, types = osl_terrain.uniform_levels(
        vec.unwrapped.num_envs,
        int(terrain.terrain_origins.shape[1]),
        int(terrain.terrain_origins.shape[0]),
      )
      osl_terrain.set_levels(terrain, levels, types)
    return vec

  def _stress_groups(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """The stress bands (:func:`~..terrain.band_masks`) of the stress env."""
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      return {}
    generator = terrain.cfg.terrain_generator
    assert generator is not None
    names = list(generator.sub_terrains)
    return osl_terrain.band_masks(terrain.terrain_levels, terrain.terrain_types, names)


class TerrainGaitLibraryRunner(TerrainRunnerMixin, GaitLibraryOnPolicyRunner):
  """The stage-1 terrain runner (module docstring)."""


def read_spec(path: str | Path) -> dict[str, Any]:
  """A spec file (``terrain_generator.json``)."""
  return json.loads(Path(path).read_text())
