"""Generated terrain for the gait-library terrain tasks (round 7; blind policies).

The ``...-Terrain`` tasks (stage 1 and the split twin) train on a seeded curriculum
grid built by mjlab's own :class:`~mjlab.terrains.TerrainGenerator`, with three
heightfield types in one column each and ten levels along the rows:

* ``rough``: random heights per vertex, uniform on 1 mm steps from 0 up to 0-6 cm
  (the top grows linearly with the level);
* ``waves``: ``a (cos + sin)`` waves of 2 m wavelength, amplitude 0-6 cm (a surface
  within +-6 cm at level 9), feathered to 0 over the outer 1 m of each tile, so wave
  tiles meet their neighbours without a seam;
* ``obstacles``: 150 flat-topped boxes of 0.4-1.2 m sides and height 0-4 cm on a
  flat floor, with a 1 m square cleared at the tile centre.

Level 0 of every type is a flat heightfield (no flat box column). Tiles are 16 m
square (a 10 s episode leaves its tile ~3% of the time), with a 20 m flat border.
Every heightfield gets a base of at least ``MIN_BASE_THICKNESS`` below its lowest
point, so a loaded foot never reaches the bottom of a thin flat tile.

The observation is unchanged: no height scan, and the reference keeps its meaning
(a planar pose re-based on the robot, a height fixed per episode). Episodes start
at the ground height under the reference's feet (:class:`HeightLookup`, used by
:class:`~.mdp.TerrainGaitLibraryCommand`).

``TERRAIN_SPEC.md`` beside this module documents the generator as the deploy port
must reproduce it; :func:`terrain_spec` writes the same facts as JSON (a bundle's
and a run's ``params/`` ``terrain_generator.json``, :data:`SPEC_FILE`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np
import torch

from mjlab.terrains import (
  HfDiscreteObstaclesTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

TERRAIN_KIND = "hfield_generator"
"""The spec's ``kind``: generated heightfields (the stairs bundles' ``terrain.json``
is TERRA's box list, which has no ``kind``)."""
TERRAIN_VERSION = "osl_terrain_v1"
"""The generator settings below, as one name: a change to any of them is a new
version (and a new spec)."""
SPEC_SCHEMA = 1
CHECKPOINT_KEY = "terrain"
"""A terrain run's checkpoint ``infos`` key: the version, the spec digest and every
env's level and type (:func:`terrain_state`)."""
SPEC_FILE = "terrain_generator.json"
"""The spec's file in a terrain run's ``params`` folder and in its bundles. Not
``terrain.json``: that name is the stairs runs' and bundles' box list."""
MANIFEST_KEY = "terrain_generator"
"""A terrain bundle's manifest section (:func:`manifest_section`). Not ``terrain``:
osl_deploy reads that key as a single clip's box terrain compiled into model.mjb
and refuses a bundle whose section is anything else, so a generator section there
would stop every panel of the bundle, flat ones included."""

TERRAIN_TYPES = ("rough", "waves", "obstacles")
"""The grid's columns, in order (world +y)."""
NUM_LEVELS = 10
"""The grid's rows (world +x): level 0 is flat, level 9 the hardest."""
TILE_SIZE = 16.0
"""Tile side (m): a 10 s episode leaves an 8 m tile 32% of the time, a 16 m one
3.2% (round-7 terrain audit)."""
BORDER_WIDTH = 20.0
BORDER_HEIGHT = 1.0
HORIZONTAL_SCALE = 0.1
"""Vertex spacing in the generator's grid (m): 160 vertices per tile side."""
VERTICAL_SCALE = 0.001
"""Height unit (m): 1 mm, so levels step evenly (5 mm units would quantize the
wave amplitudes to 0, 0, 1, 1, 2, ... cm)."""
MIN_BASE_THICKNESS = 0.05
"""Smallest heightfield base below a tile's lowest point (m). The generator's own
base is a ratio of the tile's height span, 1 mm on a flat tile, thinner than a
loaded foot's contact penetration."""

ROUGH_NOISE_RANGE = (0.0, 0.06)
ROUGH_NOISE_STEP = 0.001
WAVE_AMPLITUDE_RANGE = (0.0, 0.06)
WAVE_LENGTH = 2.0
WAVE_EDGE_WIDTH = 1.0
"""Waves fade to 0 over this distance (m) from each tile edge (a raised cosine)."""
OBSTACLE_HEIGHT_RANGE = (0.0, 0.04)
OBSTACLE_WIDTH_RANGE = (0.4, 1.25)
"""The generator draws sides on a 4-vertex grid from this range: 0.4, 0.8 or
1.2 m."""
OBSTACLE_COUNT = 150
OBSTACLE_PLATFORM = 1.0

TERRAIN_SEED = 0
"""Training ground: a resume rebuilds the same tiles."""
STRESS_TERRAIN_SEED = 1
"""Stress validation's ground: other tiles of the same generator, the same for every
run and pass."""
MAX_INIT_LEVEL = 5
"""Training envs start at a random level 0..5 (G1's ``max_init_terrain_level``)."""

STRESS_BANDS: tuple[tuple[str, int, int], ...] = (
  ("L0-2", 0, 2),
  ("L3-6", 3, 6),
  ("L7-9", 7, 9),
)
"""Stress survival is reported per band of levels (first and last level), per type
and over all types."""

SPAWN_SITES = (
  "talus_l",
  "toes_l",
  "OSL_KA_L1_osl_ankle_assembly",
  "OSL_KA_L1_osl_foot_assembly",
)
"""The reference's foot bodies (among its sites) whose ground decides the episode's
start height."""
SPAWN_RADIUS = 0.25
"""Half-width (m) of the square of ground around each spawn site that the start
height clears: it holds every foot collider of the site's foot."""

SPAWN_RULE = (
  "per reference foot site (spawn.sites) at world (x, y): take the tile whose "
  "half-open area [x0, x0 + tile_x) x [y0, y0 + tile_y) holds (x, y) "
  "(grid.tile_corner_m; a site off every tile reads 0); u = (x - vx) / sx and "
  "v = (y - vy) / sy in float64, with (vx, vy) the world xy of the tile's vertex "
  "(0, 0) and sx, sy its vertex spacings (mujoco.vertex_xy); the site's value is "
  "the highest world height (float32) of every vertex (row i, column j) with "
  "floor(u - r / sx) <= j <= ceil(u + r / sx) and floor(v - r / sy) <= i <= "
  "ceil(v + r / sy), both ranges clipped to the tile's 0..ncol-1 and 0..nrow-1 "
  "(r = radius_m): every vertex of every heightfield cell that meets the square "
  "[x - r, x + r] x [y - r, y + r], clipped to the site's tile. The start height "
  "is the maximum over the sites; the flat reference is raised rigidly by it and "
  "keeps it for the episode"
)
"""The start-height rule as the spec states it (:meth:`HeightLookup.window_max`
per site; ``tests/osl_terrain_port.window_max`` is a port written from it)."""

TILE_LANGUAGE = (
  "Python 3 statements, run in order with np = numpy, p = types.<name>.params, "
  "d = the level's difficulty (types.<name>.levels[level].difficulty) and rng = "
  "the grid's one numpy.random.default_rng(seed), shared by every tile in "
  "rng.tile_order; tiles.common first, then the type's formula, which leaves the "
  "int16 array noise[W, L] (heights in vertical_scale units; first index along "
  "world y, see mujoco). Normative: tests/test_osl_terrain.py runs these strings "
  "against mjlab's compiled heightfields"
)
TILE_COMMON = (
  "W = int(p['size'][0] / p['horizontal_scale'])",
  "L = int(p['size'][1] / p['horizontal_scale'])",
)
TILE_FORMULAS: dict[str, tuple[str, ...]] = {
  "rough": (
    "s = d if p['scale_with_difficulty'] else 1.0",
    "lo = p['noise_range'][0] * s",
    "hi = p['noise_range'][1] * s",
    "hmin = int(lo / p['vertical_scale'])",
    "hmax = int(hi / p['vertical_scale'])",
    "hstep = int(p['noise_step'] / p['vertical_scale'])",
    "values = np.arange(hmin, hmax + hstep, hstep)",
    "noise = rng.choice(values, size=(W, L)).astype(np.int16)",
  ),
  "waves": (
    "lo, hi = p['amplitude_range']",
    "amp = lo + d * (hi - lo)",
    "a = int(0.5 * amp / p['vertical_scale'])",
    "k = 2 * np.pi / (L / p['num_waves'])",
    "i = np.arange(W).reshape(W, 1)",
    "j = np.arange(L).reshape(1, L)",
    "raw = a * (np.cos(j * k) + np.sin(i * k))",
    "e, hs = p['edge_width'], p['horizontal_scale']",
    "qw = np.arange(W, dtype=np.float64)",
    "qw = np.minimum(qw, W - 1 - qw) * hs",
    "tw = 0.5 - 0.5 * np.cos(np.pi * np.minimum(qw / e, 1.0))",
    "ql = np.arange(L, dtype=np.float64)",
    "ql = np.minimum(ql, L - 1 - ql) * hs",
    "tl = 0.5 - 0.5 * np.cos(np.pi * np.minimum(ql / e, 1.0))",
    "noise = np.rint(raw * np.outer(tw, tl)).astype(np.int16)",
  ),
  "obstacles": (
    "lo, hi = p['obstacle_height_range']",
    "hu = int((lo + d * (hi - lo)) / p['vertical_scale'])",
    "wmin = int(p['obstacle_width_range'][0] / p['horizontal_scale'])",
    "wmax = int(p['obstacle_width_range'][1] / p['horizontal_scale'])",
    "widths = np.arange(wmin, wmax + 1, 4)",
    "widths = widths if len(widths) else np.array([wmin])",
    "noise = np.zeros((W, L), dtype=np.int16)",
    "for _ in range(p['num_obstacles']): "
    "ow = rng.choice(widths); "
    "ol = ow if p['square_obstacles'] else rng.choice(widths); "
    "x = rng.choice(np.arange(0, W, 4)); "
    "y = rng.choice(np.arange(0, L, 4)); "
    "noise[x:min(x + ow, W), y:min(y + ol, L)] = hu",
    "hp = int(p['platform_width'] / p['horizontal_scale']) // 2",
    "cx, cy = W // 2, L // 2",
    "noise[max(cx - hp, 0):min(cx + hp, W), max(cy - hp, 0):min(cy + hp, L)] = 0",
  ),
}
"""Every type's tile as exact statements (:data:`TILE_LANGUAGE`): the spec's
``types.<name>.formula``, so ``terrain_generator.json`` alone rebuilds the tiles."""


##
# The sub-terrains.
##


def _with_min_base(output: TerrainOutput, min_base: float) -> TerrainOutput:
  """Deepen every heightfield's base to at least ``min_base`` (the surface is
  unchanged: only the prism bottom below the lowest point moves)."""
  for geometry in output.geometries:
    field = geometry.hfield
    if field is not None:
      field.size[3] = max(float(field.size[3]), min_base)
  return output


@dataclass(kw_only=True)
class OslHfRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
  """mjlab's random-uniform heightfield with a base of at least ``min_base``."""

  min_base: float = MIN_BASE_THICKNESS

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    return _with_min_base(super().function(difficulty, spec, rng), self.min_base)


@dataclass(kw_only=True)
class OslHfDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg):
  """mjlab's discrete-obstacle heightfield with a base of at least ``min_base``."""

  min_base: float = MIN_BASE_THICKNESS

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    return _with_min_base(super().function(difficulty, spec, rng), self.min_base)


def edge_taper(n: int, horizontal_scale: float, edge_width: float) -> np.ndarray:
  """(n,) the raised-cosine weight of vertex ``i``: 0 on the tile's edge vertices,
  1 from ``edge_width`` inward (``0.5 - 0.5 cos(pi min(d / edge_width, 1))``, ``d``
  the distance to the nearer edge vertex)."""
  i = np.arange(n, dtype=np.float64)
  d = np.minimum(i, n - 1 - i) * horizontal_scale
  if edge_width <= 0.0:
    return np.ones(n)
  return 0.5 - 0.5 * np.cos(np.pi * np.minimum(d / edge_width, 1.0))


@dataclass(kw_only=True)
class FeatheredHfWaveTerrainCfg(HfWaveTerrainCfg):
  """mjlab's wave heightfield (its no-border path), faded to 0 over ``edge_width``
  from every tile edge, with a base of at least ``min_base``.

  The wave is upstream's ``a (cos(k y') + sin(k x'))`` in the generator's pixel
  indices, ``a`` = ``int(0.5 amplitude / vertical_scale)``, multiplied by
  :func:`edge_taper` along both axes before rounding to height units."""

  edge_width: float = WAVE_EDGE_WIDTH
  min_base: float = MIN_BASE_THICKNESS

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # Waves draw nothing.
    if self.num_waves <= 0:
      raise ValueError(f"Number of waves must be positive. Got: {self.num_waves}")
    if self.border_width != 0.0:
      raise ValueError("FeatheredHfWaveTerrainCfg has no border; feather instead.")
    body = spec.body("terrain")
    noise = wave_noise(
      difficulty,
      self.size,
      self.amplitude_range,
      self.num_waves,
      self.horizontal_scale,
      self.vertical_scale,
      self.edge_width,
    )
    elevation_min = int(np.min(noise))
    elevation_max = int(np.max(noise))
    elevation_range = (
      elevation_max - elevation_min if elevation_max != elevation_min else 1
    )
    max_physical_height = elevation_range * self.vertical_scale
    base_thickness = max_physical_height * self.base_thickness_ratio
    if elevation_range > 0:
      normalized = (noise - elevation_min) / elevation_range
    else:  # pragma: no cover - the range is at least 1 by construction.
      normalized = np.zeros_like(noise, dtype=np.float64)
    field = spec.add_hfield(
      name=f"hfield_wave_{_tile_counter()}",
      size=[
        self.size[0] / 2,
        self.size[1] / 2,
        max_physical_height,
        max(base_thickness, self.min_base),
      ],
      nrow=noise.shape[0],
      ncol=noise.shape[1],
      userdata=normalized.flatten().astype(np.float32).tolist(),
    )
    # Upstream offsets the geom down by half the span, so the wave sits about z=0.
    z = -max_physical_height / 2
    physical_heights = normalized * max_physical_height + z
    material = color_by_height(spec, noise, field.name, physical_heights)
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[self.size[0] / 2, self.size[1] / 2, z],
      material=material,
    )
    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
    return TerrainOutput(
      origin=origin, geometries=[TerrainGeometry(geom=geom, hfield=field)]
    )


_COUNTER = [0]


def _tile_counter() -> int:
  """A process-wide counter for unique wave heightfield names (upstream uses
  random uuids; names never reach the physics)."""
  _COUNTER[0] += 1
  return _COUNTER[0]


def wave_noise(
  difficulty: float,
  size: Sequence[float],
  amplitude_range: Sequence[float],
  num_waves: int,
  horizontal_scale: float,
  vertical_scale: float,
  edge_width: float,
) -> np.ndarray:
  """(W, L) int16 heights of a feathered wave tile, in ``vertical_scale`` units, in
  the generator's (first index, second index) order."""
  amplitude = amplitude_range[0] + difficulty * (
    amplitude_range[1] - amplitude_range[0]
  )
  width_pixels = int(size[0] / horizontal_scale)
  length_pixels = int(size[1] / horizontal_scale)
  amplitude_pixels = int(0.5 * amplitude / vertical_scale)
  wave_length = length_pixels / num_waves
  wave_number = 2 * np.pi / wave_length
  x = np.arange(0, width_pixels)
  y = np.arange(0, length_pixels)
  xx, yy = np.meshgrid(x, y, sparse=True)
  xx = xx.reshape(width_pixels, 1)
  yy = yy.reshape(1, length_pixels)
  hf_raw = amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
  taper = np.outer(
    edge_taper(width_pixels, horizontal_scale, edge_width),
    edge_taper(length_pixels, horizontal_scale, edge_width),
  )
  return np.rint(hf_raw * taper).astype(np.int16)


##
# The generator and the scene.
##


def sub_terrain_cfgs() -> dict[str, Any]:
  """The three sub-terrains, in column order (equal spawn shares)."""
  num_waves = round(TILE_SIZE / WAVE_LENGTH)
  # The generator sets every sub-terrain's size to the tile's when it runs; set here
  # too, so a run's saved config and a rebuilt one agree.
  size = (TILE_SIZE, TILE_SIZE)
  return {
    "rough": OslHfRandomUniformTerrainCfg(
      proportion=1.0,
      size=size,
      horizontal_scale=HORIZONTAL_SCALE,
      vertical_scale=VERTICAL_SCALE,
      noise_range=ROUGH_NOISE_RANGE,
      noise_step=ROUGH_NOISE_STEP,
      scale_with_difficulty=True,
    ),
    "waves": FeatheredHfWaveTerrainCfg(
      proportion=1.0,
      size=size,
      horizontal_scale=HORIZONTAL_SCALE,
      vertical_scale=VERTICAL_SCALE,
      amplitude_range=WAVE_AMPLITUDE_RANGE,
      num_waves=num_waves,
    ),
    "obstacles": OslHfDiscreteObstaclesTerrainCfg(
      proportion=1.0,
      size=size,
      horizontal_scale=HORIZONTAL_SCALE,
      vertical_scale=VERTICAL_SCALE,
      obstacle_height_mode="fixed",
      obstacle_width_range=OBSTACLE_WIDTH_RANGE,
      obstacle_height_range=OBSTACLE_HEIGHT_RANGE,
      num_obstacles=OBSTACLE_COUNT,
      platform_width=OBSTACLE_PLATFORM,
    ),
  }


def terrain_generator_cfg(seed: int = TERRAIN_SEED) -> TerrainGeneratorCfg:
  """The curriculum grid: one column per type, ``NUM_LEVELS`` rows, seeded."""
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=True,
    size=(TILE_SIZE, TILE_SIZE),
    border_width=BORDER_WIDTH,
    border_height=BORDER_HEIGHT,
    num_rows=NUM_LEVELS,
    num_cols=len(TERRAIN_TYPES),
    color_scheme="height",
    sub_terrains=sub_terrain_cfgs(),
    difficulty_range=(0.0, 1.0),
    add_lights=False,
  )


def terrain_entity_cfg(
  seed: int = TERRAIN_SEED, max_init_level: int | None = MAX_INIT_LEVEL
) -> TerrainEntityCfg:
  """The scene's terrain: the grid of :func:`terrain_generator_cfg`, envs placed at
  random levels ``0..max_init_level`` (None: every level) and in the types' equal
  shares."""
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=terrain_generator_cfg(seed),
    max_init_terrain_level=max_init_level,
  )


def is_terrain_cfg(cfg: Any) -> bool:
  """Whether an env config's scene is generated terrain."""
  terrain = getattr(getattr(cfg, "scene", None), "terrain", None)
  return bool(
    terrain is not None
    and getattr(terrain, "terrain_type", None) == "generator"
    and getattr(terrain, "terrain_generator", None) is not None
  )


def saved_terrain(saved_env: Mapping[str, Any] | None) -> dict[str, Any] | None:
  """A run's terrain generator settings from its ``params/env.yaml`` (as read), or
  None for a flat-ground run."""
  terrain = ((saved_env or {}).get("scene") or {}).get("terrain") or {}
  if terrain.get("terrain_type") != "generator":
    return None
  generator = terrain.get("terrain_generator")
  return dict(generator) if isinstance(generator, Mapping) else None


##
# The spec (terrain_generator.json).
##


def _plain(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(k): _plain(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_plain(v) for v in value]
  if isinstance(value, np.generic):
    return value.item()
  return value


def level_difficulty(level: int, num_levels: int = NUM_LEVELS) -> float:
  """The generator's difficulty of a row: ``lower + (upper - lower) * t`` with
  ``t = level / max(num_levels - 1, 1)``, range (0, 1)."""
  lower, upper = 0.0, 1.0
  t = level / max(num_levels - 1, 1)
  return lower + (upper - lower) * t


def level_table(cfgs: Mapping[str, Any] | None = None) -> dict[str, list[dict]]:
  """Per type and level, the integers the generator derives from the difficulty
  (in Python floats, in the generator's order of operations): a port's check."""
  cfgs = cfgs or sub_terrain_cfgs()
  rough, waves, obstacles = cfgs["rough"], cfgs["waves"], cfgs["obstacles"]
  table: dict[str, list[dict]] = {name: [] for name in TERRAIN_TYPES}
  for level in range(NUM_LEVELS):
    d = level_difficulty(level)
    lo = rough.noise_range[0] * d
    hi = rough.noise_range[1] * d
    table["rough"].append(
      {
        "level": level,
        "difficulty": d,
        "height_min_units": int(lo / rough.vertical_scale),
        "height_max_units": int(hi / rough.vertical_scale),
        "height_step_units": int(rough.noise_step / rough.vertical_scale),
      }
    )
    amp = waves.amplitude_range[0] + d * (
      waves.amplitude_range[1] - waves.amplitude_range[0]
    )
    table["waves"].append(
      {
        "level": level,
        "difficulty": d,
        "amplitude_m": amp,
        "amplitude_units": int(0.5 * amp / waves.vertical_scale),
      }
    )
    h = obstacles.obstacle_height_range[0] + d * (
      obstacles.obstacle_height_range[1] - obstacles.obstacle_height_range[0]
    )
    w_min = int(obstacles.obstacle_width_range[0] / obstacles.horizontal_scale)
    w_max = int(obstacles.obstacle_width_range[1] / obstacles.horizontal_scale)
    widths = np.arange(w_min, w_max + 1, 4)
    table["obstacles"].append(
      {
        "level": level,
        "difficulty": d,
        "height_m": h,
        "height_units": int(h / obstacles.vertical_scale),
        "width_choices_px": [int(w) for w in widths],
      }
    )
  return table


def border_boxes(
  extent: tuple[float, float],
  centre: tuple[float, float] = (0.0, 0.0),
  width: float = BORDER_WIDTH,
  height: float = BORDER_HEIGHT,
) -> list[dict[str, list[float]]]:
  """The four boxes of the flat border around a block of tiles ``extent`` (x, y)
  metres wide centred on ``centre``, as mjlab's generator builds the grid's
  (:func:`mjlab.terrains.utils.make_border`): +y, -y, -x, +x, each a MuJoCo box
  ``half_size`` at ``pos``, its top at z = 0. The +-y boxes span the whole outer
  width; the +-x boxes fill the gap between them."""
  ex, ey = float(extent[0]), float(extent[1])
  cx, cy = float(centre[0]), float(centre[1])
  w, h = float(width), float(height)
  z = -h / 2
  along_x = [(ex + 2 * w) / 2, w / 2, h / 2]
  along_y = [w / 2, ey / 2, h / 2]
  return [
    {"half_size": along_x, "pos": [cx, cy + ey / 2 + w / 2, z]},
    {"half_size": along_x, "pos": [cx, cy - ey / 2 - w / 2, z]},
    {"half_size": along_y, "pos": [cx - ex / 2 - w / 2, cy, z]},
    {"half_size": along_y, "pos": [cx + ex / 2 + w / 2, cy, z]},
  ]


def terrain_spec(
  generator: TerrainGeneratorCfg | None = None,
  seed: int | None = None,
  stress_seed: int = STRESS_TERRAIN_SEED,
) -> dict[str, Any]:
  """The terrain as a deploy port must rebuild it (``TERRAIN_SPEC.md``): every
  generator setting, the derived per-level integers, the grid layout, the RNG
  scheme and the MuJoCo conversion. ``seed`` overrides the generator's (the
  training seed); ``stress_seed`` is stress validation's."""
  generator = generator or terrain_generator_cfg()
  seed = generator.seed if seed is None else seed
  if seed is None:
    raise ValueError("A terrain spec needs a seeded generator (the grid's seed).")
  seed = int(seed)
  subs = generator.sub_terrains
  names = list(subs)
  if tuple(names) != TERRAIN_TYPES:
    raise ValueError(f"Unknown terrain columns {names}; want {list(TERRAIN_TYPES)}.")
  types: dict[str, Any] = {}
  for name in names:
    cfg = subs[name]
    params = _plain(asdict(cfg))
    params.pop("flat_patch_sampling", None)
    params.pop("proportion", None)
    params["size"] = [float(s) for s in generator.size]
    types[name] = {
      "class": f"{type(cfg).__module__}.{type(cfg).__qualname__}",
      "params": params,
      "formula": list(TILE_FORMULAS[name]),
    }
  table = level_table(subs)
  for name in names:
    types[name]["levels"] = table[name]
  nrows, ncols = generator.num_rows, len(names)
  sx, sy = generator.size
  return {
    "kind": TERRAIN_KIND,
    "schema_version": SPEC_SCHEMA,
    "version": TERRAIN_VERSION,
    "seed": seed,
    "stress_seed": int(stress_seed),
    "rng": {
      "generator": "numpy.random.default_rng(seed) (PCG64), one stream per grid",
      "tile_order": "type (column) outer, level (row) inner: (0,0), (1,0), ...",
      "numpy_version": np.__version__,
      "draws": {
        "rough": "rng.choice(arange(min, max + step, step), size=(W, L))",
        "waves": "none",
        "obstacles": (
          "per obstacle: rng.choice(widths), rng.choice(widths), "
          "rng.choice(arange(0, W, 4)), rng.choice(arange(0, L, 4))"
        ),
      },
    },
    "grid": {
      "types": names,
      "num_levels": nrows,
      "tile_size_m": [float(sx), float(sy)],
      "level_axis": "+x",
      "type_axis": "+y",
      "tile_corner_m": (
        "x = -num_levels * tile_x / 2 + level * tile_x, "
        "y = -num_types * tile_y / 2 + type * tile_y"
      ),
      "extent_m": {
        "x": [-nrows * sx / 2, nrows * sx / 2],
        "y": [-ncols * sy / 2, ncols * sy / 2],
      },
      "border_width_m": float(generator.border_width),
      "border_height_m": float(generator.border_height),
      "border_top_z_m": 0.0,
      "border_boxes_m": border_boxes(
        (nrows * sx, ncols * sy),
        width=float(generator.border_width),
        height=float(generator.border_height),
      ),
      "border_rule": (
        "four boxes (+y, -y, -x, +x; MuJoCo half sizes and positions, metres) of "
        "border_width_m around the tiles, top at z = 0, the default contact "
        "parameters; a deployment surrounds whatever tiles it builds (one, the "
        "grid, a tile repeated) with the same border around their extent, and "
        "never keeps a plane: wave troughs lie below z = 0"
      ),
      "difficulty": "level / (num_levels - 1), range (0, 1)",
    },
    "mujoco": {
      "hfield": (
        "nrow = W (generator first index), ncol = L; data = (noise - min) / "
        "max(max - min, 1) as float32, row-major; size = (tile_x / 2, tile_y / 2, "
        "max(max - min, 1) * vertical_scale, max(that * base_thickness_ratio, "
        "min_base))"
      ),
      "geom_pos": "tile corner + (tile_x / 2, tile_y / 2, z), z per type",
      "geom_z": {
        "rough": "0",
        "waves": "-size[2] / 2",
        "obstacles": "0 (fixed mode)",
      },
      "vertex_xy": "x = -size[0] + c * 2 size[0] / (ncol - 1), y likewise with r",
    },
    "tiles": {"language": TILE_LANGUAGE, "common": list(TILE_COMMON)},
    "types": types,
    "levels_per_type": nrows,
    "spawn": {
      "rule": SPAWN_RULE,
      "sites": list(SPAWN_SITES),
      "radius_m": SPAWN_RADIUS,
      "env_origin": "the tile centre",
    },
    "curriculum": {
      "rule": (
        "per env: +1 level after a time-out, -1 after an early termination "
        "(nan_term included), frozen on the first reset, a promotion past the top "
        "draws a random level"
      ),
      "max_init_level": MAX_INIT_LEVEL,
    },
    "stress": {
      "seed": int(stress_seed),
      "levels": "env i: type i % num_types, level (i // num_types) % num_levels",
      "bands": [list(b) for b in STRESS_BANDS],
    },
  }


def spec_digest(spec: Mapping[str, Any]) -> str:
  """SHA-256 of the spec's canonical JSON, without its ``rng.numpy_version`` (the
  generator's numbers, not the machine that wrote them)."""
  data = json.loads(json.dumps(spec))
  data.get("rng", {}).pop("numpy_version", None)
  text = json.dumps(data, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(text.encode()).hexdigest()


def spec_json(spec: Mapping[str, Any]) -> str:
  return json.dumps(spec, indent=2, sort_keys=False) + "\n"


def check_saved_generator(saved: Mapping[str, Any], seed: int | None = None) -> None:
  """Refuse a run whose ``params/env.yaml`` terrain generator differs from this
  checkout's (``TERRAIN_VERSION``): its bundle's spec would describe other ground.

  Raises:
    ValueError: Naming the differing settings.
  """
  want = _plain(asdict(terrain_generator_cfg(TERRAIN_SEED if seed is None else seed)))
  got = _plain(dict(saved))
  if seed is None:
    want["seed"] = got.get("seed")
  diffs = _differences(got, want)
  if diffs:
    raise ValueError(
      f"The run's terrain generator differs from this checkout's {TERRAIN_VERSION} "
      f"in: {', '.join(diffs[:8])}"
    )


def _differences(a: Any, b: Any, prefix: str = "") -> list[str]:
  if isinstance(a, Mapping) and isinstance(b, Mapping):
    out: list[str] = []
    for key in sorted(set(a) | set(b), key=str):
      out += _differences(a.get(key), b.get(key), f"{prefix}{key}.")
    return out
  if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
    if len(a) != len(b):
      return [prefix[:-1]]
    out = []
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
      out += _differences(x, y, f"{prefix}{i}.")
    return out
  numbers = (int, float)
  if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool):
    if math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12):
      return []
  return [] if a == b else [prefix[:-1]]


##
# Heights.
##


class HeightLookup:
  """The compiled scene's heightfields as one static lookup (world x, y -> height),
  on any device.

  Tiles form a regular grid (the generator's). Queries inside a tile read that
  tile alone; outside every tile they read the border's top, z = 0.

  :meth:`window_max` bounds the surface from above: the highest vertex of every
  heightfield cell that meets the square ``[x - r, x + r] x [y - r, y + r]``
  (MuJoCo's surface is planar on each half cell, so it never rises above a cell's
  corners). :meth:`surface` is MuJoCo's surface itself (its two triangles per cell,
  split along the (r, c) - (r + 1, c + 1) diagonal).
  """

  def __init__(
    self,
    heights: torch.Tensor,
    corner: torch.Tensor,
    spacing: torch.Tensor,
    grid_origin: tuple[float, float],
    tile_size: tuple[float, float],
    tile_of: torch.Tensor,
  ) -> None:
    self.heights = heights  # (T, nrow, ncol) world z of every vertex
    self.corner = corner  # (T, 2) world xy of vertex (0, 0)
    self.spacing = spacing  # (T, 2) vertex spacing along x (columns), y (rows)
    self.grid_origin = grid_origin
    self.tile_size = tile_size
    self.tile_of = tile_of  # (grid rows along x, grid cols along y) -> tile or -1

  @classmethod
  def from_model(
    cls, model: mujoco.MjModel, device: str | torch.device = "cpu"
  ) -> HeightLookup | None:
    """The lookup of ``model``'s heightfield geoms, or None when it has none.

    Raises:
      ValueError: If the heightfields do not tile a regular grid of equal tiles.
    """
    geoms = [
      g
      for g in range(model.ngeom)
      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD
    ]
    if not geoms:
      return None
    # World poses of the (static) heightfield geoms, whatever bodies hold them.
    kinematics = mujoco.MjData(model)
    mujoco.mj_kinematics(model, kinematics)
    heights, corners, spacings, sizes = [], [], [], []
    shape = None
    for g in geoms:
      h = int(model.geom_dataid[g])
      nrow, ncol = int(model.hfield_nrow[h]), int(model.hfield_ncol[h])
      if shape is None:
        shape = (nrow, ncol)
      elif shape != (nrow, ncol):
        raise ValueError("Heightfield tiles of different resolutions.")
      adr = int(model.hfield_adr[h])
      size = model.hfield_size[h]
      data = np.asarray(model.hfield_data[adr : adr + nrow * ncol], dtype=np.float64)
      pos = kinematics.geom_xpos[g]
      if np.any(np.abs(kinematics.geom_xmat[g] - np.eye(3).ravel()) > 1e-9):
        raise ValueError("A rotated heightfield geom: not a generator grid.")
      heights.append(data.reshape(nrow, ncol) * size[2] + pos[2])
      corners.append((pos[0] - size[0], pos[1] - size[1]))
      spacings.append((2 * size[0] / (ncol - 1), 2 * size[1] / (nrow - 1)))
      sizes.append((2 * size[0], 2 * size[1]))
    sizes_np = np.asarray(sizes)
    if np.ptp(sizes_np[:, 0]) > 1e-9 or np.ptp(sizes_np[:, 1]) > 1e-9:
      raise ValueError("Heightfield tiles of different sizes.")
    tile = (float(sizes_np[0, 0]), float(sizes_np[0, 1]))
    corners_np = np.asarray(corners)
    x0, y0 = float(corners_np[:, 0].min()), float(corners_np[:, 1].min())
    rows = np.rint((corners_np[:, 0] - x0) / tile[0]).astype(int)
    cols = np.rint((corners_np[:, 1] - y0) / tile[1]).astype(int)
    off = np.abs(corners_np[:, 0] - (x0 + rows * tile[0])) + np.abs(
      corners_np[:, 1] - (y0 + cols * tile[1])
    )
    if off.max() > 1e-6:
      raise ValueError("Heightfield tiles off a regular grid.")
    tile_of = -np.ones((rows.max() + 1, cols.max() + 1), dtype=np.int64)
    tile_of[rows, cols] = np.arange(len(geoms))

    def as_t(a: Any, dtype: torch.dtype) -> torch.Tensor:
      return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    return cls(
      heights=as_t(np.stack(heights), torch.float32),
      corner=as_t(corners_np, torch.float64),
      spacing=as_t(np.asarray(spacings), torch.float64),
      grid_origin=(x0, y0),
      tile_size=tile,
      tile_of=as_t(tile_of, torch.long),
    )

  def tile_index(self, xy: torch.Tensor) -> torch.Tensor:
    """(...) the tile under each world point of ``xy`` (..., 2), -1 off the grid."""
    xy = xy.to(torch.float64)
    r = torch.floor((xy[..., 0] - self.grid_origin[0]) / self.tile_size[0]).long()
    c = torch.floor((xy[..., 1] - self.grid_origin[1]) / self.tile_size[1]).long()
    nr, nc = self.tile_of.shape
    inside = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
    tile = self.tile_of[r.clamp(0, nr - 1), c.clamp(0, nc - 1)]
    return torch.where(inside, tile, torch.full_like(tile, -1))

  def window_max(self, xy: torch.Tensor, radius: float) -> torch.Tensor:
    """(...) an upper bound of the surface over the square of half-width ``radius``
    around each point of ``xy`` (..., 2): see the class docstring. Points off the
    grid read 0; a square that crosses its tile's edge is clipped to the tile."""
    shape = xy.shape[:-1]
    flat = xy.reshape(-1, 2).to(torch.float64)
    tile = self.tile_index(flat)
    t = tile.clamp(min=0)
    _, nrow, ncol = self.heights.shape
    local = (flat - self.corner[t]) / self.spacing[t]  # (N, 2): column, row
    reach = radius / self.spacing[t]
    lo = torch.floor(local - reach).long()
    hi = torch.ceil(local + reach).long()
    c0, c1 = lo[:, 0].clamp(0, ncol - 1), hi[:, 0].clamp(0, ncol - 1)
    r0, r1 = lo[:, 1].clamp(0, nrow - 1), hi[:, 1].clamp(0, nrow - 1)
    width = int(math.ceil(2 * radius / float(self.spacing.min()))) + 3
    steps = torch.arange(width, device=flat.device)
    cols = torch.minimum(c0[:, None] + steps, c1[:, None])  # (N, W)
    rows = torch.minimum(r0[:, None] + steps, r1[:, None])
    patch = self.heights[t[:, None, None], rows[:, :, None], cols[:, None, :]]
    top = patch.amax(dim=(1, 2))
    top = torch.where(tile >= 0, top, torch.zeros_like(top))
    return top.reshape(shape)

  def surface(self, xy: torch.Tensor) -> torch.Tensor:
    """(...) MuJoCo's surface height under each point of ``xy`` (..., 2); 0 off
    the grid."""
    shape = xy.shape[:-1]
    flat = xy.reshape(-1, 2).to(torch.float64)
    tile = self.tile_index(flat)
    t = tile.clamp(min=0)
    _, nrow, ncol = self.heights.shape
    local = (flat - self.corner[t]) / self.spacing[t]
    c = local[:, 0].clamp(0, ncol - 1 - 1e-9)
    r = local[:, 1].clamp(0, nrow - 1 - 1e-9)
    c0, r0 = c.floor().long(), r.floor().long()
    fc, fr = c - c0, r - r0
    h = self.heights.to(torch.float64)
    z00 = h[t, r0, c0]
    z01 = h[t, r0, c0 + 1]
    z10 = h[t, r0 + 1, c0]
    z11 = h[t, r0 + 1, c0 + 1]
    # Triangles (r, c)-(r+1, c+1)-(r, c+1) below the diagonal, (r, c)-(r+1, c)-
    # (r+1, c+1) above it.
    lower = z00 + fc * (z01 - z00) + fr * (z11 - z01)
    upper = z00 + fr * (z10 - z00) + fc * (z11 - z10)
    z = torch.where(fc >= fr, lower, upper)
    z = torch.where(tile >= 0, z, torch.zeros_like(z))
    return z.to(torch.float32).reshape(shape)


##
# Levels: the stress layout, the bands and checkpoint state.
##


def uniform_levels(
  num_envs: int,
  num_types: int = len(TERRAIN_TYPES),
  num_levels: int = NUM_LEVELS,
  device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
  """Stress validation's fixed layout: env ``i`` on type ``i % num_types`` at level
  ``(i // num_types) % num_levels``: every (type, level) pair gets the same number
  of envs, give or take one."""
  i = torch.arange(num_envs, device=device)
  return (i // num_types) % num_levels, i % num_types


def set_levels(terrain: Any, levels: torch.Tensor, types: torch.Tensor) -> None:
  """Put every env of a generated-terrain entity on ``levels`` and ``types``, in
  place: the terrain's level and type tensors and the env origins."""
  device = terrain.terrain_levels.device
  terrain.terrain_levels[:] = levels.to(
    device=device, dtype=terrain.terrain_levels.dtype
  )
  terrain.terrain_types[:] = types.to(device=device, dtype=terrain.terrain_types.dtype)
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ]


def band_masks(
  levels: torch.Tensor, types: torch.Tensor, names: Sequence[str] = TERRAIN_TYPES
) -> dict[str, torch.Tensor]:
  """The stress groups: ``<type>_<band>``, ``<band>`` (every type) and ``<type>``
  (every level), each a boolean env mask (empty ones included)."""
  groups: dict[str, torch.Tensor] = {}
  for band, lo, hi in STRESS_BANDS:
    in_band = (levels >= lo) & (levels <= hi)
    for i, name in enumerate(names):
      groups[f"{name}_{band}"] = in_band & (types == i)
    groups[band] = in_band
  for i, name in enumerate(names):
    groups[name] = types == i
  return groups


def terrain_state(terrain: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
  """A checkpoint's terrain record: the version, the spec digest, the grid shape and
  every env's level and type (CPU int16)."""
  return {
    "version": TERRAIN_VERSION,
    "spec_sha256": spec_digest(spec),
    "seed": int(spec["seed"]),
    "num_levels": int(terrain.terrain_origins.shape[0]),
    "num_types": int(terrain.terrain_origins.shape[1]),
    "levels": terrain.terrain_levels.detach().to("cpu", torch.int16).clone(),
    "types": terrain.terrain_types.detach().to("cpu", torch.int16).clone(),
  }


def restore_problems(
  record: Mapping[str, Any], terrain: Any, spec: Mapping[str, Any]
) -> list[str]:
  """Why a checkpoint's terrain record cannot be restored onto ``terrain`` (empty:
  it can)."""
  problems = []
  if record.get("spec_sha256") != spec_digest(spec):
    problems.append(
      f"its terrain spec ({record.get('version')}, seed {record.get('seed')}) is "
      f"not this env's ({spec['version']}, seed {spec['seed']})"
    )
  levels = record.get("levels")
  n = int(terrain.terrain_levels.shape[0])
  if levels is None or int(len(levels)) != n:
    got = None if levels is None else len(levels)
    problems.append(f"it holds {got} envs' levels, this env has {n}")
  return problems


##
# Bundles.
##

SPEC_DOC = "mjlab src/mjlab/tasks/osl_gait_library/TERRAIN_SPEC.md"
SPEC_FIXTURE = "mjlab tests/fixtures/osl_terrain_heights.npz"
"""Heightfields of a few (seed, type, level) a port must reproduce."""


def run_terrain_spec(saved_env: Mapping[str, Any] | None) -> dict[str, Any] | None:
  """The terrain spec of a run, from its ``params/env.yaml`` (as read): None for a
  flat-ground run.

  Raises:
    ValueError: If the run's generator is not this checkout's
      (:func:`check_saved_generator`; the seed may be any).
  """
  generator = saved_terrain(saved_env)
  if generator is None:
    return None
  check_saved_generator(generator)
  return terrain_spec(terrain_generator_cfg(int(generator["seed"])))


def manifest_section(spec: Mapping[str, Any]) -> dict[str, Any]:
  """A terrain bundle's ``manifest.terrain_generator`` (:data:`MANIFEST_KEY`): what
  the bundle's spec file holds and how a deployment builds the ground from it."""
  return {
    "kind": spec["kind"],
    "file": SPEC_FILE,
    "version": spec["version"],
    "schema_version": spec["schema_version"],
    "seed": spec["seed"],
    "stress_seed": spec["stress_seed"],
    "types": list(spec["grid"]["types"]),
    "num_levels": spec["grid"]["num_levels"],
    "tile_size_m": list(spec["grid"]["tile_size_m"]),
    "spec_sha256": spec_digest(spec),
    "documentation": SPEC_DOC,
    "reference_fixture": SPEC_FIXTURE,
    "model": (
      "model.mjb is the flat training scene: its ground is the plane geom "
      f"'terrain'. A deployment generates the heightfields of {SPEC_FILE} (any "
      "seed, type and level), puts them and the z = 0 border boxes around them "
      "(grid.border_rule) in place of that plane and compiles; nothing of the "
      "terrain is compiled into model.mjb, and the bundle deploys on flat ground "
      "as it is."
    ),
    "start_height": spec["spawn"]["rule"],
    "training": (
      "trained on the curriculum grid at seed `seed` with flat validation; stress "
      "validation used `stress_seed`"
    ),
  }


def readme_lines(section: Mapping[str, Any]) -> str:
  """A terrain bundle's README addition."""
  return (
    f"\nTerrain: `{section['file']}` ({section['kind']}, {section['version']}; "
    f"types {', '.join(section['types'])}, {section['num_levels']} levels, "
    f"training seed {section['seed']}, stress seed {section['stress_seed']}; "
    f"manifest `{MANIFEST_KEY}`). model.mjb is flat: generate the heightfields "
    f"from {section['file']} (see {section['documentation']}) and put them, with "
    "their z = 0 border, in place of its plane.\n"
  )
