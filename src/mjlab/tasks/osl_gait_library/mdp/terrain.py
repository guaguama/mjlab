"""The terrain tasks' command and curriculum (``mjlab.tasks.osl_gait_library.terrain``).

* :class:`TerrainGaitLibraryCommand`: the library command, its episodes starting at
  the ground height under the reference's feet instead of the env origin's z (the
  generator puts a rough tile's origin at half its height range, below many of its
  vertices).
* :class:`terrain_levels_survival`: a per-env, survival-gated level curriculum for
  the library command (G1's ``terrain_levels_vel`` asserts a velocity command and
  freezes only while ``common_step_counter == 0``, which a resumed or stage-2 run
  never sees).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.tasks.osl_gait_library.terrain import (
  SPAWN_RADIUS,
  SPAWN_SITES,
  HeightLookup,
)

from .commands import GaitLibraryCommand, GaitLibraryCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


@dataclass(kw_only=True)
class TerrainGaitLibraryCommandCfg(GaitLibraryCommandCfg):
  """:class:`~.commands.GaitLibraryCommandCfg` plus the start height's rule."""

  spawn_sites: tuple[str, ...] = SPAWN_SITES
  """Bodies among ``site_bodies`` whose ground decides the start height."""
  spawn_radius: float = SPAWN_RADIUS
  """Half-width (m) of the ground square cleared around each spawn site."""

  def build(self, env: ManagerBasedRlEnv) -> TerrainGaitLibraryCommand:
    return TerrainGaitLibraryCommand(self, env)


class TerrainGaitLibraryCommand(GaitLibraryCommand):
  """The library command on generated terrain.

  An episode's reference (root and sites) is raised by the highest ground under
  its start frame's feet: the highest heightfield vertex within ``spawn_radius``
  (per axis) of each spawn site, maximised over the sites
  (:meth:`~mjlab.tasks.osl_gait_library.terrain.HeightLookup.window_max`). Every
  foot collider then starts at least as far above the ground as the flat reference
  puts it above the plane; the other foot may hang above a hollow, and drops into
  it. The height stays fixed for the episode, as on flat ground. A scene without
  heightfields starts at the env origin's z, as the library command does.
  """

  # Narrows the base's cfg: pyright's reportIncompatibleVariableOverride.
  cfg: TerrainGaitLibraryCommandCfg  # pyright: ignore

  def __init__(self, cfg: TerrainGaitLibraryCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    missing = [s for s in cfg.spawn_sites if s not in cfg.site_bodies]
    if missing:
      raise ValueError(f"spawn_sites {missing} are not among site_bodies.")
    self._spawn_ids = torch.tensor(
      [list(cfg.site_bodies).index(s) for s in cfg.spawn_sites],
      dtype=torch.long,
      device=self.device,
    )
    self.heights = HeightLookup.from_model(env.sim.mj_model, self.device)

  def _start_height(
    self,
    clip: torch.Tensor,
    frame: torch.Tensor,
    anchor: torch.Tensor,
    origins: torch.Tensor,
  ) -> torch.Tensor:
    if self.heights is None:
      return super()._start_height(clip, frame, anchor, origins)
    return self.start_heights(clip, frame, anchor)

  def start_heights(
    self, clip: torch.Tensor, frame: torch.Tensor, anchor: torch.Tensor
  ) -> torch.Tensor:
    """(n,) the start height of references of ``clip`` at ``frame`` under
    ``anchor`` (see the class docstring)."""
    assert self.heights is not None
    start = self._gather(clip, frame, anchor, rvel=False)
    feet = start["site_pos"][:, self._spawn_ids, :2]
    ground = self.heights.window_max(feet, float(self.cfg.spawn_radius))
    return ground.amax(dim=1).to(torch.float32)


class terrain_levels_survival:
  """Per-env terrain levels for the gait-library tasks, gated on survival.

  At every episode reset (curriculum terms run first in the reset): an env whose
  episode timed out moves one level up, one that ended early (any termination,
  ``nan_term`` included) one level down, clipped at 0; an env promoted past the
  top level goes to a random level (G1's rule, mjlab's ``update_env_origins``).
  The run's 2 m root-deviation limit makes 'timed out' mean 'kept up with the
  reference', as G1's distance rule does.

  Levels are frozen on the term's first call (the env's first reset, before any
  step: no episode has ended) and on the call after :meth:`freeze_next` (a reset
  that restarts episodes without ending them, e.g. after a checkpoint restores
  the levels). Unlike G1's term, the freeze does not read ``common_step_counter``,
  which a resumed run and a stage-2 run restore.

  Logged per reset (``Curriculum/<term>/<key>``): ``mean`` (every env's level) and
  each type's mean level (by name); and among the envs reset by this call,
  ``promoted``, ``demoted``, ``randomized`` (promoted past the top) and
  ``nan_term`` (their share).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    del cfg
    terrain = env.scene.terrain
    generator = None if terrain is None else terrain.cfg.terrain_generator
    if (
      terrain is None
      or generator is None
      or not generator.curriculum
      or terrain.terrain_origins is None
    ):
      raise ValueError(
        "terrain_levels_survival needs a curriculum terrain generator (one column "
        "per type, levels along the rows)."
      )
    self._names = list(generator.sub_terrains)
    self._freeze = True

  def freeze_next(self) -> None:
    """Leave the levels unchanged at the next call."""
    self._freeze = True

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
  ) -> dict[str, torch.Tensor]:
    terrain = env.scene.terrain
    assert terrain is not None
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    n = max(int(env_ids.numel()), 1)
    zeros = torch.zeros(env_ids.shape, dtype=torch.bool, device=env.device)
    nan = zeros
    if self._freeze:
      self._freeze = False
      up = down = zeros
    else:
      terms = env.termination_manager
      down = terms.terminated[env_ids].bool()
      up = terms.time_outs[env_ids].bool() & ~down
      if "nan_term" in terms.active_terms:
        nan = terms.get_term("nan_term")[env_ids].bool()
    top = int(terrain.max_terrain_level) - 1
    randomized = up & (terrain.terrain_levels[env_ids] >= top)
    terrain.update_env_origins(env_ids, up, down)
    levels = terrain.terrain_levels.float()
    stats: dict[str, Any] = {"mean": levels.mean()}
    for i, name in enumerate(self._names):
      mask = terrain.terrain_types == i
      if bool(mask.any()):
        stats[name] = levels[mask].mean()
    stats["promoted"] = up.float().sum() / n
    stats["demoted"] = down.float().sum() / n
    stats["randomized"] = randomized.float().sum() / n
    stats["nan_term"] = nan.float().sum() / n
    return stats
