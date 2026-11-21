"""A NumPy-only port of the OSL terrain generator, written from the spec file
(``terrain_generator.json``) and ``src/mjlab/tasks/osl_gait_library/TERRAIN_SPEC.md``
alone (no mjlab import).

It proves the spec complete: ``tests/test_osl_terrain.py`` checks it against the
heightfields mjlab compiles and against the reference fixture
(``fixtures/osl_terrain_heights.npz``). osl_deploy's port must match the same
fixture. :func:`formula_noise` runs the spec's own statements (``tiles.common`` and
``types.<name>.formula``) instead, and :func:`window_max` is the start-height rule
(``spawn.rule``) written from its text. Regenerate the fixture (after a deliberate
generator change, which is a new terrain version, or any spec change, since the
fixture embeds the spec) with::

  PYTHONPATH=src python tests/osl_terrain_port.py --write-fixture
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE = Path(__file__).parent / "fixtures" / "osl_terrain_heights.npz"
FIXTURE_TILES: tuple[tuple[int, str, int], ...] = (
  (0, "rough", 9),
  (0, "waves", 9),
  (0, "obstacles", 9),
  (1, "rough", 4),
  (1, "waves", 3),
  (1, "obstacles", 6),
  (7, "obstacles", 2),
)
"""(seed, type, level) of the fixture's tiles: every type, the training and stress
seeds and one other, the last tile of each column (whose draws follow every earlier
tile's)."""


def difficulty(level: int, num_levels: int) -> float:
  lower, upper = 0.0, 1.0
  t = level / max(num_levels - 1, 1)
  return lower + (upper - lower) * t


def _rough(p: dict[str, Any], d: float, rng: np.random.Generator) -> np.ndarray:
  vs, hs = p["vertical_scale"], p["horizontal_scale"]
  w, length = int(p["size"][0] / hs), int(p["size"][1] / hs)
  scale = d if p["scale_with_difficulty"] else 1.0
  lo, hi = p["noise_range"][0] * scale, p["noise_range"][1] * scale
  hmin, hmax, step = int(lo / vs), int(hi / vs), int(p["noise_step"] / vs)
  assert p["downsampled_scale"] is None and p["border_width"] == 0.0
  heights = np.arange(hmin, hmax + step, step)
  return rng.choice(heights, size=(w, length)).astype(np.int16)


def _taper(n: int, hs: float, edge: float) -> np.ndarray:
  i = np.arange(n, dtype=np.float64)
  dist = np.minimum(i, n - 1 - i) * hs
  if edge <= 0.0:
    return np.ones(n)
  return 0.5 - 0.5 * np.cos(np.pi * np.minimum(dist / edge, 1.0))


def _waves(p: dict[str, Any], d: float) -> np.ndarray:
  vs, hs = p["vertical_scale"], p["horizontal_scale"]
  w, length = int(p["size"][0] / hs), int(p["size"][1] / hs)
  lo, hi = p["amplitude_range"]
  amp = lo + d * (hi - lo)
  a = int(0.5 * amp / vs)
  k = 2 * np.pi / (length / p["num_waves"])
  i = np.arange(w).reshape(w, 1)
  j = np.arange(length).reshape(1, length)
  raw = a * (np.cos(j * k) + np.sin(i * k))
  taper = np.outer(_taper(w, hs, p["edge_width"]), _taper(length, hs, p["edge_width"]))
  return np.rint(raw * taper).astype(np.int16)


def _obstacles(p: dict[str, Any], d: float, rng: np.random.Generator) -> np.ndarray:
  vs, hs = p["vertical_scale"], p["horizontal_scale"]
  assert p["obstacle_height_mode"] == "fixed" and p["border_width"] == 0.0
  w, length = int(p["size"][0] / hs), int(p["size"][1] / hs)
  lo, hi = p["obstacle_height_range"]
  hu = int((lo + d * (hi - lo)) / vs)
  wmin = int(p["obstacle_width_range"][0] / hs)
  wmax = int(p["obstacle_width_range"][1] / hs)
  widths = np.arange(wmin, wmax + 1, 4)
  if len(widths) == 0:
    widths = np.array([wmin])
  noise = np.zeros((w, length), dtype=np.int16)
  for _ in range(p["num_obstacles"]):
    ow = rng.choice(widths)
    ol = ow if p["square_obstacles"] else rng.choice(widths)
    x = rng.choice(np.arange(0, w, 4))
    y = rng.choice(np.arange(0, length, 4))
    noise[x : min(x + ow, w), y : min(y + ol, length)] = hu
  plat = int(p["platform_width"] / hs)
  cx, cy, half = w // 2, length // 2, plat // 2
  noise[
    max(cx - half, 0) : min(cx + half, w), max(cy - half, 0) : min(cy + half, length)
  ] = 0
  return noise


def grid_noise(spec: dict[str, Any], seed: int) -> dict[tuple[str, int], np.ndarray]:
  """Every tile's int16 ``noise`` of the grid at ``seed``, by (type, level)."""
  rng = np.random.default_rng(seed)
  levels = spec["grid"]["num_levels"]
  out: dict[tuple[str, int], np.ndarray] = {}
  for name in spec["grid"]["types"]:
    p = spec["types"][name]["params"]
    for level in range(levels):
      d = difficulty(level, levels)
      if name == "rough":
        out[(name, level)] = _rough(p, d, rng)
      elif name == "waves":
        out[(name, level)] = _waves(p, d)
      elif name == "obstacles":
        out[(name, level)] = _obstacles(p, d, rng)
      else:
        raise ValueError(f"Unknown terrain type {name!r}")
  return out


def formula_noise(spec: dict[str, Any], seed: int) -> dict[tuple[str, int], np.ndarray]:
  """Every tile's ``noise`` from the spec's statements alone (``tiles.language``):
  ``tiles.common`` then ``types.<name>.formula``, one rng for the grid."""
  rng = np.random.default_rng(seed)
  common = "\n".join(spec["tiles"]["common"])
  out: dict[tuple[str, int], np.ndarray] = {}
  for name in spec["grid"]["types"]:
    entry = spec["types"][name]
    code = compile(common + "\n" + "\n".join(entry["formula"]), name, "exec")
    for row in entry["levels"]:
      scope: dict[str, Any] = {
        "np": np,
        "p": entry["params"],
        "d": row["difficulty"],
        "rng": rng,
      }
      exec(code, scope)  # The spec's own statements (a test helper).
      out[(name, row["level"])] = scope["noise"]
  return out


def window_max(
  spec: dict[str, Any],
  tiles: dict[tuple[str, int], dict[str, np.ndarray]],
  x: float,
  y: float,
  radius: float,
) -> float:
  """One site's start-height value by ``spawn.rule``: ``tiles`` maps (type, level)
  to :func:`mujoco_tile`'s output (its world ``heights``, ``size`` and ``pos``)."""
  types = spec["grid"]["types"]
  levels = spec["grid"]["num_levels"]
  sx_tile, sy_tile = spec["grid"]["tile_size_m"]
  level = math.floor((x - (-levels * sx_tile / 2)) / sx_tile)
  kind = math.floor((y - (-len(types) * sy_tile / 2)) / sy_tile)
  if not (0 <= level < levels and 0 <= kind < len(types)):
    return 0.0
  t = tiles[(types[kind], level)]
  heights, size, pos = t["heights"], t["size"], t["pos"]
  nrow, ncol = heights.shape
  vx, vy = pos[0] - size[0], pos[1] - size[1]
  sx, sy = 2 * size[0] / (ncol - 1), 2 * size[1] / (nrow - 1)
  u, v = (x - vx) / sx, (y - vy) / sy
  j0 = min(max(math.floor(u - radius / sx), 0), ncol - 1)
  j1 = min(max(math.ceil(u + radius / sx), 0), ncol - 1)
  i0 = min(max(math.floor(v - radius / sy), 0), nrow - 1)
  i1 = min(max(math.ceil(v + radius / sy), 0), nrow - 1)
  return float(heights[i0 : i1 + 1, j0 : j1 + 1].astype(np.float32).max())


def mujoco_tile(
  spec: dict[str, Any], name: str, level: int, noise: np.ndarray
) -> dict[str, np.ndarray]:
  """The tile's MuJoCo heightfield: ``data`` (nrow, ncol) float32, ``size`` (4,),
  ``pos`` (3,) and the world ``heights`` (nrow, ncol) of its vertices."""
  p = spec["types"][name]["params"]
  vs = p["vertical_scale"]
  emin, emax = int(noise.min()), int(noise.max())
  span = emax - emin if emax != emin else 1
  top = span * vs
  data = ((noise - emin) / span).astype(np.float32)
  size = np.array(
    [
      p["size"][0] / 2,
      p["size"][1] / 2,
      top,
      max(top * p["base_thickness_ratio"], p["min_base"]),
    ]
  )
  types = spec["grid"]["types"]
  levels = spec["grid"]["num_levels"]
  sx, sy = spec["grid"]["tile_size_m"]
  x0 = -levels * sx / 2 + level * sx
  y0 = -len(types) * sy / 2 + types.index(name) * sy
  z = -top / 2 if name == "waves" else 0.0
  pos = np.array([x0 + sx / 2, y0 + sy / 2, z])
  heights = data.astype(np.float64) * size[2] + z
  return {"data": data, "size": size, "pos": pos, "heights": heights}


def tile(spec: dict[str, Any], seed: int, name: str, level: int) -> dict[str, Any]:
  noise = grid_noise(spec, seed)[(name, level)]
  return {"noise": noise, **mujoco_tile(spec, name, level, noise)}


def write_fixture() -> None:
  from mjlab.tasks.osl_gait_library import terrain

  spec = json.loads(json.dumps(terrain.terrain_spec()))
  arrays: dict[str, Any] = {"spec_json": np.array(terrain.spec_json(spec))}
  for seed, name, level in FIXTURE_TILES:
    t = tile(spec, seed, name, level)
    key = f"s{seed}_{name}_L{level}"
    arrays[f"{key}_noise"] = t["noise"]
    arrays[f"{key}_data"] = t["data"]
    arrays[f"{key}_size"] = t["size"]
    arrays[f"{key}_pos"] = t["pos"]
  FIXTURE.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(FIXTURE, **arrays)
  print(f"{FIXTURE} ({FIXTURE.stat().st_size / 1024:.0f} KiB, numpy {np.__version__})")


def check_close(a: float, b: float) -> bool:
  return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


if __name__ == "__main__":
  if "--write-fixture" in sys.argv:
    write_fixture()
