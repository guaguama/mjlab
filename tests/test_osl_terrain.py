"""The OSL terrain tasks (round 7): generator, spec and port, start height, level
curriculum, checkpointed levels, stress layout, the flat-teacher refusal and the
terrain bundle's spec (CPU only, no Warp simulation)."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
from osl_terrain_port import (
  FIXTURE,
  FIXTURE_TILES,
  formula_noise,
  grid_noise,
  mujoco_tile,
  window_max,
)
from test_osl_export import ROBOT, _make_run
from test_osl_export import library as library  # noqa: F401  (the fixture)

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.tasks.osl_gait_library import terrain as T
from mjlab.tasks.osl_gait_library.config.env_cfgs import (
  NAN_TERMINATION,
  TERRAIN_CURRICULUM,
  apply_terrain,
  osl_gait_library_env_cfg,
  osl_gait_library_terrain_env_cfg,
)
from mjlab.tasks.osl_gait_library.mdp import (
  TerrainGaitLibraryCommand,
  TerrainGaitLibraryCommandCfg,
  terrain_levels_survival,
)
from mjlab.tasks.osl_gait_library.rl.terrain_runner import (
  TerrainRunnerMixin,
  freeze_terrain,
)
from mjlab.tasks.osl_mimic.rl.runner import stress_group_metrics
from mjlab.tasks.osl_split.rl.runner import SplitOnPolicyRunner
from mjlab.tasks.osl_split.rl.terrain_runner import (
  TerrainSplitRunner,
  check_terrain_teacher,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.terrains import TerrainEntity
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab.utils.os import dump_yaml

STAGE1 = "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
_TERM_CFG = cast(Any, SimpleNamespace(params={}))
SPLIT = "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"


def _compile_grid(seed: int) -> mujoco.MjModel:
  spec = mujoco.MjSpec()
  TerrainGenerator(T.terrain_generator_cfg(seed)).compile(spec)
  return spec.compile()


def _model_tiles(model: mujoco.MjModel) -> dict[tuple[str, int], dict[str, Any]]:
  """Every heightfield tile of a compiled grid by (type, level): data, size, pos."""
  out = {}
  x0 = -T.NUM_LEVELS * T.TILE_SIZE / 2
  y0 = -len(T.TERRAIN_TYPES) * T.TILE_SIZE / 2
  for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_HFIELD:
      continue
    h = model.geom_dataid[g]
    nrow, ncol, adr = model.hfield_nrow[h], model.hfield_ncol[h], model.hfield_adr[h]
    pos = model.geom_pos[g]
    level = int(round((pos[0] - x0 - T.TILE_SIZE / 2) / T.TILE_SIZE))
    kind = T.TERRAIN_TYPES[int(round((pos[1] - y0 - T.TILE_SIZE / 2) / T.TILE_SIZE))]
    out[(kind, level)] = {
      "data": model.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol).copy(),
      "size": model.hfield_size[h].copy(),
      "pos": pos.copy(),
    }
  return out


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
  return json.loads(json.dumps(T.terrain_spec()))


##
# Tasks.
##


def test_terrain_tasks_are_twins_of_the_unicycle_tasks() -> None:
  tasks = set(list_tasks())
  stage1 = sorted(t for t in tasks if t.startswith("Mjlab-Gait-Library-Unicycle"))
  split = sorted(t for t in tasks if t.startswith("Mjlab-Split-Unicycle"))
  assert sum(t.endswith("-Terrain") for t in stage1) == 8
  assert sum(t.endswith("-Terrain") for t in split) == 8
  for task in (STAGE1, SPLIT):
    flat = cast(Any, load_env_cfg(task))
    terrain = cast(Any, load_env_cfg(task + "-Terrain"))
    assert flat.scene.terrain.terrain_type == "plane"
    assert terrain.scene.terrain.terrain_type == "generator"
    assert terrain.scene.terrain.max_init_terrain_level == T.MAX_INIT_LEVEL
    assert terrain.scene.terrain.terrain_generator.seed == T.TERRAIN_SEED
    assert list(terrain.curriculum) == [TERRAIN_CURRICULUM]
    assert NAN_TERMINATION in terrain.terminations
    assert NAN_TERMINATION not in flat.terminations
    assert isinstance(terrain.commands["motion"], TerrainGaitLibraryCommandCfg)
    # Everything the policy sees and is scored on is the flat task's.
    assert terrain.observations == flat.observations
    assert terrain.rewards == flat.rewards
    assert terrain.actions == flat.actions
    assert terrain.sim == flat.sim
    assert terrain.scene.num_envs == flat.scene.num_envs
    play = cast(Any, load_env_cfg(task + "-Terrain", play=True))
    assert play.scene.terrain.terrain_type == "generator"
    assert play.scene.terrain.max_init_terrain_level is None and not play.curriculum
    rl_flat = cast(Any, load_rl_cfg(task))
    rl = cast(Any, load_rl_cfg(task + "-Terrain"))
    assert rl.experiment_name == rl_flat.experiment_name + "_terrain"
    assert rl.wandb_tags == (*rl_flat.wandb_tags, "terrain")
    assert rl.env_stages == rl_flat.env_stages
    assert rl.algorithm == rl_flat.algorithm and rl.validation == rl_flat.validation


def test_terrain_validates_on_flat_ground() -> None:
  cfg = cast(
    Any, osl_gait_library_terrain_env_cfg("msk", "compliant", "clf", "validation")
  )
  assert cfg.scene.terrain.terrain_type == "plane" and not cfg.curriculum
  with pytest.raises(ValueError, match="flat ground"):
    apply_terrain(osl_gait_library_env_cfg(mode="validation"), "validation")


##
# Generator, lookup, spec and port.
##


def test_levels_span_and_lookup_matches_mujoco() -> None:
  model = _compile_grid(T.TERRAIN_SEED)
  lookup = T.HeightLookup.from_model(model)
  assert lookup is not None and tuple(lookup.tile_of.shape) == (10, 3)
  tops = {"rough": 0.06, "waves": 0.06, "obstacles": 0.04}
  for (kind, level), tile in _model_tiles(model).items():
    z = tile["data"] * tile["size"][2] + tile["pos"][2]
    span = float(z.max() - z.min())
    if level == 0:
      assert span == 0.0, (kind, level)
    # Heights are whole millimetres (a wave's 4 x its truncated amplitude unit).
    want = tops[kind] * level / 9 * (2 if kind == "waves" else 1)
    slack = 0.0041 if kind == "waves" else 0.0011
    assert want - slack <= span <= want + 1e-9, (kind, level)
    assert tile["size"][3] >= T.MIN_BASE_THICKNESS
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  rng = np.random.default_rng(0)
  pts = np.c_[rng.uniform(-80, 80, 300), rng.uniform(-24, 24, 300)]
  ray = np.array(
    [
      2.0
      - mujoco.mj_ray(
        model,
        data,
        np.array([x, y, 2.0]),
        np.array([0, 0, -1.0]),
        None,
        True,
        -1,
        np.zeros(1, np.int32),
      )
      for x, y in pts
    ]
  )
  surface = lookup.surface(torch.tensor(pts)).numpy()
  np.testing.assert_allclose(surface, ray, atol=1e-6)
  # The window bound covers the surface anywhere inside the square (squares inside
  # their tile: one that crosses a tile edge is clipped to its tile).
  local = np.mod(pts - np.array(lookup.grid_origin), T.TILE_SIZE)
  keep = np.all((local > 0.3) & (local < T.TILE_SIZE - 0.3), axis=1)
  centre = torch.tensor(pts[keep][:50])
  bound = lookup.window_max(centre, 0.25)
  for dx in np.linspace(-0.25, 0.25, 6):
    for dy in np.linspace(-0.25, 0.25, 6):
      inside = lookup.surface(centre + torch.tensor([dx, dy]))
      assert bool((inside <= bound + 1e-6).all())
  off_grid = lookup.window_max(torch.tensor([[200.0, 0.0]]), 0.25)
  assert float(off_grid[0]) == 0.0


def test_the_spec_lists_the_border_boxes(spec) -> None:
  """The spec's border boxes are the compiled grid's (a deployment rebuilds them
  around its tiles instead of keeping a plane)."""
  model = _compile_grid(T.TERRAIN_SEED)
  boxes = [
    g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
  ]
  listed = spec["grid"]["border_boxes_m"]
  assert len(boxes) == len(listed) == 4
  for g, box in zip(boxes, listed, strict=True):
    np.testing.assert_allclose(model.geom_size[g], box["half_size"], atol=1e-12)
    np.testing.assert_allclose(model.geom_pos[g], box["pos"], atol=1e-12)
    assert box["pos"][2] + box["half_size"][2] == 0.0  # Top at z = 0.
  one = T.border_boxes((16.0, 16.0), centre=(8.0, 8.0))
  assert one[0] == {"half_size": [28.0, 10.0, 0.5], "pos": [8.0, 26.0, -0.5]}
  assert one[3] == {"half_size": [10.0, 8.0, 0.5], "pos": [26.0, 8.0, -0.5]}


@pytest.mark.parametrize("seed", [T.TERRAIN_SEED, T.STRESS_TERRAIN_SEED, 7])
def test_the_numpy_port_rebuilds_every_tile(spec, seed) -> None:
  """The port written from the spec alone (``tests/osl_terrain_port.py``) gives
  mjlab's compiled heightfields: the spec is complete. So do the spec's own
  statements (``tiles.common`` + ``types.<name>.formula``), run as written."""
  ported = grid_noise(spec, seed)
  compiled = _model_tiles(_compile_grid(seed))
  assert set(ported) == set(compiled)
  stated = formula_noise(spec, seed)
  assert set(stated) == set(ported)
  for key, noise in stated.items():
    assert noise.dtype == np.int16
    np.testing.assert_array_equal(noise, ported[key], err_msg=f"formula {key}")
  for (kind, level), noise in ported.items():
    tile = mujoco_tile(spec, kind, level, noise)
    got = compiled[(kind, level)]
    np.testing.assert_array_equal(tile["data"], got["data"], err_msg=f"{kind} {level}")
    np.testing.assert_allclose(tile["size"], got["size"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(tile["pos"], got["pos"], rtol=0, atol=1e-12)
  # The per-level integers the spec lists are the ported tiles' own.
  for row in spec["types"]["rough"]["levels"]:
    noise = ported[("rough", row["level"])]
    assert int(noise.max()) <= row["height_max_units"]
  for row in spec["types"]["obstacles"]["levels"]:
    noise = ported[("obstacles", row["level"])]
    assert int(noise.max()) == row["height_units"]


def test_the_reference_fixture_is_this_generator(spec) -> None:
  """The fixture a deploy port must match holds mjlab's tiles and spec."""
  fixture = np.load(FIXTURE)
  saved = json.loads(str(fixture["spec_json"]))
  assert T.spec_digest(saved) == T.spec_digest(spec)
  grids = {
    seed: _model_tiles(_compile_grid(seed)) for seed in {s for s, _, _ in FIXTURE_TILES}
  }
  for seed, kind, level in FIXTURE_TILES:
    key = f"s{seed}_{kind}_L{level}"
    got = grids[seed][(kind, level)]
    np.testing.assert_array_equal(fixture[f"{key}_data"], got["data"], err_msg=key)
    np.testing.assert_allclose(fixture[f"{key}_size"], got["size"], atol=1e-12)
    np.testing.assert_allclose(fixture[f"{key}_pos"], got["pos"], atol=1e-12)
    noise = fixture[f"{key}_noise"]
    assert noise.dtype == np.int16 and noise.shape == (160, 160)


def test_spec_is_json_and_versioned(spec) -> None:
  assert spec["kind"] == T.TERRAIN_KIND and spec["version"] == T.TERRAIN_VERSION
  assert spec["grid"]["types"] == list(T.TERRAIN_TYPES)
  assert spec["seed"] == T.TERRAIN_SEED and spec["stress_seed"] == T.STRESS_TERRAIN_SEED
  assert spec["types"]["waves"]["params"]["edge_width"] == T.WAVE_EDGE_WIDTH
  other = T.terrain_spec(seed=5)
  assert other["seed"] == 5 and T.spec_digest(other) != T.spec_digest(spec)
  assert T.spec_digest(json.loads(T.spec_json(spec))) == T.spec_digest(spec)


##
# Start height.
##


def _ramp_lookup() -> T.HeightLookup:
  """One 2 m tile centred on the origin, rising 3 cm along +x, plus a 5 cm vertex."""
  n = 21
  heights = np.tile(np.linspace(0.0, 0.03, n), (n, 1))
  heights[15, 5] = 0.05  # x = -0.5, y = +0.5.
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="terrain")
  top = float(heights.max())
  field = spec.add_hfield(
    name="ramp",
    size=[1.0, 1.0, top, 0.05],
    nrow=n,
    ncol=n,
    userdata=(heights / top).flatten().astype(np.float32).tolist(),
  )
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name)
  lookup = T.HeightLookup.from_model(spec.compile())
  assert lookup is not None
  return lookup


def test_start_height_clears_the_ground_under_every_foot() -> None:
  from test_osl_gait_library import _fake_command

  base = _fake_command(num_envs=3)
  cmd = TerrainGaitLibraryCommand.__new__(TerrainGaitLibraryCommand)
  cmd.__dict__.update(base.__dict__)
  cmd.cfg = TerrainGaitLibraryCommandCfg(
    library_dir="synthetic",
    rel_static_envs=0.0,
    max_snap_error=0.2,
    site_bodies=("root", "foot"),
    spawn_sites=("foot",),
  )
  cmd._spawn_ids = torch.tensor([1])
  cmd.heights = _ramp_lookup()
  cmd.robot = cast(
    Any,
    SimpleNamespace(
      data=base.robot.data,
      write_joint_state_to_sim=lambda *a, **k: None,
      write_root_state_to_sim=lambda *a, **k: None,
      reset=lambda *a, **k: None,
      set_joint_position_target=lambda *a, **k: None,
    ),
  )
  # Env origins under the foot's start (frame 0: the foot is at the root's xy).
  fake_env = cast(Any, cmd._env)
  fake_env.scene.env_origins = torch.tensor(
    [[0.0, 0.0, -9.0], [-0.5, 0.45, -9.0], [0.8, -0.8, -9.0]]
  )
  cmd.cfg.fixed_clips = True  # Clip i % 3 from frame 0, no random heading.
  ids = torch.arange(3)
  cmd._start_episode(ids)
  # The highest vertex of every cell meeting the 0.25 m square: x up to 0.3 at the
  # centre (ramp 0.03 * 1.3 / 2), the 5 cm vertex for the second env, the tile edge
  # (clipped to the tile) for the third.
  want = torch.tensor([0.03 * 1.3 / 2, 0.05, 0.03])
  torch.testing.assert_close(cmd.z_offset, want, atol=1e-6, rtol=0)
  served = cmd.reference(0)
  clip_z = cmd.data["root_pos"][cmd.clip, 0, 2]
  torch.testing.assert_close(served.root_pos_w[:, 2], clip_z + want, atol=1e-6, rtol=0)
  # Without heightfields the start height is the env origin's z, as on flat ground.
  cmd.heights = None
  cmd._start_episode(ids)
  torch.testing.assert_close(cmd.z_offset, torch.full((3,), -9.0))


def test_the_start_height_reads_only_the_sites_tile() -> None:
  """Two 1 m tiles side by side along x, flat at 0 and 4 cm: a site 0.1 m from the
  seam reads its own tile alone (the spec's spawn rule)."""
  heights = torch.zeros(2, 5, 5)
  heights[1] = 0.04
  lookup = T.HeightLookup(
    heights=heights,
    corner=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
    spacing=torch.full((2, 2), 0.25, dtype=torch.float64),
    grid_origin=(0.0, 0.0),
    tile_size=(1.0, 1.0),
    tile_of=torch.tensor([[0], [1]]),
  )
  sites = torch.tensor([[0.9, 0.5], [1.1, 0.5], [-0.1, 0.5]])
  torch.testing.assert_close(
    lookup.window_max(sites, 0.25), torch.tensor([0.0, 0.04, 0.0])
  )
  assert "clipped to the site's tile" in T.terrain_spec()["spawn"]["rule"]


def test_the_window_max_takes_every_vertex_of_the_cells_meeting_the_square() -> None:
  """On a hand-made tile of random heights, the start-height window is every vertex
  of every cell whose area overlaps the square, clipped to the tile: a geometric
  brute force over the cells, independent of the floor/ceil index rule."""
  gen = np.random.default_rng(3)
  nrow, ncol, spacing = 9, 12, (0.13, 0.11)  # Columns along x, rows along y.
  heights = gen.uniform(-0.05, 0.05, size=(nrow, ncol)).astype(np.float32)
  corner = (0.4, -0.3)
  lookup = T.HeightLookup(
    heights=torch.as_tensor(heights)[None],
    corner=torch.tensor([corner], dtype=torch.float64),
    spacing=torch.tensor([spacing], dtype=torch.float64),
    grid_origin=corner,
    tile_size=(spacing[0] * (ncol - 1), spacing[1] * (nrow - 1)),
    tile_of=torch.tensor([[0]]),
  )
  points = gen.uniform((0.3, -0.4), (1.9, 0.7), size=(400, 2))
  got = lookup.window_max(torch.as_tensor(points), 0.2).numpy()
  x_edges = corner[0] + np.arange(ncol) * spacing[0]
  y_edges = corner[1] + np.arange(nrow) * spacing[1]
  for (x, y), value in zip(points, got, strict=True):
    if not (x_edges[0] <= x < x_edges[-1] and y_edges[0] <= y < y_edges[-1]):
      assert value == 0.0  # Off the (one-tile) grid.
      continue
    best = -np.inf
    for i in range(nrow - 1):
      for j in range(ncol - 1):
        meets_x = x_edges[j] < x + 0.2 and x_edges[j + 1] > x - 0.2
        meets_y = y_edges[i] < y + 0.2 and y_edges[i + 1] > y - 0.2
        if meets_x and meets_y:
          best = max(best, float(heights[i : i + 2, j : j + 2].max()))
    assert value == best, (x, y)


def test_the_spawn_rule_port_matches_the_lookup(spec) -> None:
  """``tests/osl_terrain_port.window_max``, written from the spec's ``spawn.rule``
  text, gives the start-height lookup's value at random sites over the training
  grid, near the tile seams and off the grid."""
  noise = grid_noise(spec, T.TERRAIN_SEED)
  tiles = {key: mujoco_tile(spec, *key, n) for key, n in noise.items()}
  lookup = T.HeightLookup.from_model(_compile_grid(T.TERRAIN_SEED))
  assert lookup is not None
  gen = np.random.default_rng(0)
  points = gen.uniform((-82.0, -26.0), (82.0, 26.0), size=(3000, 2))
  near = gen.uniform(-0.3, 0.3, size=(2, 600))
  x_seams = np.stack(
    [
      gen.choice(np.arange(-80.0, 81.0, 16.0), 600) + near[0],
      gen.uniform(-24.0, 24.0, 600),
    ],
    axis=1,
  )
  y_seams = np.stack(
    [
      gen.uniform(-80.0, 80.0, 600),
      gen.choice(np.arange(-24.0, 25.0, 16.0), 600) + near[1],
    ],
    axis=1,
  )
  points = np.concatenate([points, x_seams, y_seams])
  for radius in (T.SPAWN_RADIUS, 0.37):
    got = lookup.window_max(torch.as_tensor(points), radius).numpy()
    want = [window_max(spec, tiles, float(x), float(y), radius) for x, y in points]
    np.testing.assert_array_equal(got, np.asarray(want, dtype=np.float32))
  assert spec["spawn"]["rule"] == T.SPAWN_RULE


def test_a_flat_command_starts_at_the_origin_height() -> None:
  from test_osl_gait_library import _fake_command

  cmd = _fake_command(num_envs=2)
  origins = torch.tensor([[0.0, 0.0, 0.25], [1.0, 0.0, -0.5]])
  heights = cmd._start_height(
    torch.zeros(2), torch.zeros(2), torch.zeros(2, 3), origins
  )
  torch.testing.assert_close(heights, origins[:, 2])


##
# The level curriculum.
##


def _terrain(num_envs: int = 12) -> TerrainEntity:
  cfg = T.terrain_entity_cfg()
  cfg.num_envs = num_envs
  return TerrainEntity(cfg, device="cpu")


def _curriculum_env(num_envs: int = 12, nan: bool = True) -> Any:
  terrain = _terrain(num_envs)
  terms = SimpleNamespace(
    terminated=torch.zeros(num_envs, dtype=torch.bool),
    time_outs=torch.zeros(num_envs, dtype=torch.bool),
    active_terms=["time_out", "site_deviation"] + (["nan_term"] if nan else []),
    nan=torch.zeros(num_envs, dtype=torch.bool),
  )
  terms.get_term = lambda name: terms.nan
  return SimpleNamespace(
    scene=SimpleNamespace(terrain=terrain),
    termination_manager=terms,
    num_envs=num_envs,
    device="cpu",
  )


def test_survival_curriculum_moves_levels() -> None:
  torch.manual_seed(0)
  env = _curriculum_env()
  terrain = env.scene.terrain
  assert int(terrain.terrain_levels.max()) <= T.MAX_INIT_LEVEL
  term = terrain_levels_survival(_TERM_CFG, env)
  terrain.terrain_levels[:] = torch.tensor([0, 3, 3, 3, 9, 9, 5, 5, 5, 0, 1, 2])
  ids = torch.arange(12)
  # Every env is flagged, yet the first call (the env's first reset) moves nothing.
  env.termination_manager.time_outs[:] = True
  stats = term(env, ids)
  assert terrain.terrain_levels.tolist() == [0, 3, 3, 3, 9, 9, 5, 5, 5, 0, 1, 2]
  assert float(stats["promoted"]) == 0.0
  # A time-out promotes, an early end demotes (nan_term too), clipped at 0; a reset
  # neither moves nothing.
  tm = env.termination_manager
  tm.time_outs[:] = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
  tm.terminated[:] = torch.tensor(
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0], dtype=torch.bool
  )
  tm.nan[:] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool)
  stats = term(env, ids)
  assert terrain.terrain_levels.tolist() == [1, 4, 2, 3, 9, 9, 5, 4, 5, 0, 1, 2]
  assert float(stats["promoted"]) == pytest.approx(2 / 12)
  assert float(stats["demoted"]) == pytest.approx(3 / 12)
  assert float(stats["nan_term"]) == pytest.approx(1 / 12)
  assert float(stats["randomized"]) == 0.0
  for i, name in enumerate(T.TERRAIN_TYPES):
    mask = terrain.terrain_types == i
    assert float(stats[name]) == pytest.approx(
      float(terrain.terrain_levels[mask].float().mean())
    )
  # Only the reset envs move, and their origins follow their tiles.
  tm.time_outs[:] = True
  tm.terminated[:] = False
  term(env, torch.tensor([3]))
  assert terrain.terrain_levels.tolist() == [1, 4, 2, 4, 9, 9, 5, 4, 5, 0, 1, 2]
  torch.testing.assert_close(
    terrain.env_origins,
    terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
  )
  # Past the top: a random level.
  levels = []
  for _ in range(40):
    terrain.terrain_levels[4] = 9
    stats = term(env, torch.tensor([4]))
    levels.append(int(terrain.terrain_levels[4]))
    assert float(stats["randomized"]) == 1.0
  assert all(0 <= v < 10 for v in levels) and len(set(levels)) > 3
  # freeze_next: the next call moves nothing (a reset after restored levels).
  term.freeze_next()
  before = terrain.terrain_levels.clone()
  term(env, ids)
  assert torch.equal(terrain.terrain_levels, before)
  term(env, slice(None))
  assert not torch.equal(terrain.terrain_levels, before)


def test_curriculum_needs_a_curriculum_grid() -> None:
  env = _curriculum_env()
  env.scene.terrain = None
  with pytest.raises(ValueError, match="curriculum terrain generator"):
    terrain_levels_survival(_TERM_CFG, env)


##
# Checkpointed levels and the frozen stress layout.
##


class _FakeBase:
  """The runner methods the mixin wraps."""

  def __init__(self, env: Any) -> None:
    self.env = env
    self.logger = SimpleNamespace(log_dir=None, disable_logs=True)
    self.saved: dict[str, Any] = {}
    self.learned = False

  def save(self, path: str, infos: dict | None = None) -> None:
    self.saved[path] = deepcopy(infos)

  def load(self, path: str, load_cfg=None, strict=True, map_location=None) -> dict:
    return deepcopy(self.saved[path])

  def learn(self, num_learning_iterations: int, init_at_random_ep_len=False) -> None:
    self.learned = True


class _FakeTerrainRunner(TerrainRunnerMixin, _FakeBase):
  pass


def _runner_env(num_envs: int = 12) -> Any:
  cfg = osl_gait_library_terrain_env_cfg()
  env = _curriculum_env(num_envs)
  term = terrain_levels_survival(_TERM_CFG, env)
  term._freeze = False
  env.cfg = cfg
  env.resets = 0

  def reset() -> None:
    env.resets += 1
    term(env, torch.arange(num_envs))  # The env's reset runs the curriculum first.

  env.curriculum_manager = SimpleNamespace(
    active_terms=[TERRAIN_CURRICULUM],
    get_term_cfg=lambda name: SimpleNamespace(func=term),
  )
  env.unwrapped = env
  env.reset = reset
  env.term = term
  return env


def test_levels_round_trip_through_a_checkpoint(capsys) -> None:
  env = _runner_env()
  runner = _FakeTerrainRunner(env)
  terrain = env.scene.terrain
  terrain.terrain_levels[:] = torch.arange(12) % 10
  runner.save("a.pt")
  record = runner.saved["a.pt"][T.CHECKPOINT_KEY]
  assert record["levels"].dtype == torch.int16
  assert record["levels"].tolist() == (torch.arange(12) % 10).tolist()
  assert record["spec_sha256"] == T.spec_digest(T.terrain_spec())
  # A resume: random start levels, then the checkpoint's once training starts, with
  # the episodes restarted on them (the reset's curriculum step moves nothing even
  # though every env carries a time-out flag).
  T.set_levels(terrain, torch.zeros(12, dtype=torch.long), terrain.terrain_types)
  env.termination_manager.time_outs[:] = True
  runner.load("a.pt")
  assert int(terrain.terrain_levels.sum()) == 0  # Restored only when it trains.
  runner.learn(1)
  assert runner.learned and env.resets == 1
  assert terrain.terrain_levels.tolist() == (torch.arange(12) % 10).tolist()
  torch.testing.assert_close(
    terrain.env_origins,
    terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
  )
  assert "Terrain levels restored" in capsys.readouterr().out
  # An evaluation's load (actors only) restores nothing.
  T.set_levels(terrain, torch.zeros(12, dtype=torch.long), terrain.terrain_types)
  runner.load("a.pt", load_cfg={"actor": True})
  runner.learn(1)
  assert int(terrain.terrain_levels.sum()) == 0


def test_a_flat_checkpoint_keeps_the_random_start_levels(capsys) -> None:
  env = _runner_env()
  runner = _FakeTerrainRunner(env)
  start = env.scene.terrain.terrain_levels.clone()
  runner.saved["flat.pt"] = {"env_state": {"common_step_counter": 240_000}}
  runner.load("flat.pt")
  runner.learn(1)
  assert torch.equal(env.scene.terrain.terrain_levels, start) and env.resets == 0
  assert "records no terrain levels (a flat-ground run)" in capsys.readouterr().out


def test_a_checkpoint_of_other_ground_or_envs_is_not_restored(capsys) -> None:
  env = _runner_env()
  runner = _FakeTerrainRunner(env)
  runner.save("a.pt")
  start = env.scene.terrain.terrain_levels.clone()
  other = deepcopy(runner.saved["a.pt"])
  other[T.CHECKPOINT_KEY]["spec_sha256"] = "0" * 64
  other[T.CHECKPOINT_KEY]["levels"] = torch.full((12,), 9, dtype=torch.int16)
  runner.saved["b.pt"] = other
  runner.load("b.pt")
  runner.learn(1)
  assert torch.equal(env.scene.terrain.terrain_levels, start)
  fewer = deepcopy(runner.saved["a.pt"])
  fewer[T.CHECKPOINT_KEY]["levels"] = torch.full((6,), 9, dtype=torch.int16)
  runner.saved["c.pt"] = fewer
  runner.load("c.pt")
  runner.learn(1)
  out = capsys.readouterr().out
  assert "is not this env's" in out and "this env has 12" in out


def test_stress_is_frozen_on_its_own_ground_with_level_bands() -> None:
  cfg = cast(Any, freeze_terrain(deepcopy(osl_gait_library_terrain_env_cfg())))
  assert cfg.scene.terrain.terrain_generator.seed == T.STRESS_TERRAIN_SEED
  assert cfg.scene.terrain.max_init_terrain_level is None and not cfg.curriculum
  assert NAN_TERMINATION in cfg.terminations
  levels, types = T.uniform_levels(512)
  counts = torch.zeros(10, 3, dtype=torch.long)
  counts.index_put_((levels, types), torch.ones(512, dtype=torch.long), accumulate=True)
  assert int(counts.min()) == 17 and int(counts.max()) == 18
  groups = T.band_masks(levels, types)
  assert sorted(groups) == sorted(
    [f"{t}_{b}" for b, _, _ in T.STRESS_BANDS for t in T.TERRAIN_TYPES]
    + [b for b, _, _ in T.STRESS_BANDS]
    + list(T.TERRAIN_TYPES)
  )
  bands = torch.stack([groups[b] for b, _, _ in T.STRESS_BANDS])
  assert bool((bands.sum(0) == 1).all())  # The bands partition the envs.
  failed = levels >= 7
  length = torch.where(failed, 50.0, 100.0)
  env = cast(Any, SimpleNamespace(step_dt=0.01))
  metrics = stress_group_metrics(groups, failed, length, env)
  assert metrics["survival_rate/L0-2"] == 1.0 and metrics["survival_rate/L7-9"] == 0.0
  assert metrics["survival_rate/rough_L3-6"] == 1.0
  assert metrics["mean_first_episode_s/L7-9"] == pytest.approx(0.5)
  assert metrics["survival_rate/waves"] == pytest.approx(
    1.0 - float(failed[types == 1].float().mean())
  )


##
# Stage 2: the teacher must have trained on this terrain.
##


def _stage1_run(root: Path, cfg: Any) -> dict[str, Any]:
  run = root / "run"
  dump_yaml(run / "params" / "env.yaml", asdict(cfg))
  return {"stage1_run_dir": str(run), "stage1_checkpoint": str(run / "model_9.pt")}


def test_a_flat_teacher_is_refused(tmp_path) -> None:
  generator = T.terrain_generator_cfg()
  flat = _stage1_run(
    tmp_path / "flat",
    osl_gait_library_env_cfg("msk", "compliant", "clf", "train", "unicycle"),
  )
  with pytest.raises(ValueError, match="FLAT-ground stage-1 run"):
    check_terrain_teacher(flat, generator)
  with pytest.raises(ValueError, match="no readable"):
    check_terrain_teacher({"stage1_run_dir": str(tmp_path / "none")}, generator)
  terrain = _stage1_run(tmp_path / "terrain", osl_gait_library_terrain_env_cfg())
  assert check_terrain_teacher(terrain, generator) is None  # No checkpoint file.
  other = cast(Any, osl_gait_library_terrain_env_cfg())
  other.scene.terrain.terrain_generator.sub_terrains["rough"].noise_range = (0.0, 0.1)
  changed = _stage1_run(tmp_path / "other", other)
  with pytest.raises(ValueError, match="other terrain"):
    check_terrain_teacher(changed, generator)


def test_a_flat_stage2_checkpoint_does_not_resume_on_terrain(monkeypatch) -> None:
  """A terrain stage 2 resumes training only from a terrain stage-2 checkpoint of
  this terrain version; play.py's actor-only load is not checked."""
  saved: dict[str, dict] = {
    "flat.pt": {"split": {"stage_start": 5999}},
    "old.pt": {T.CHECKPOINT_KEY: {"version": "osl_terrain_v0", "levels": None}},
    "terrain.pt": {T.CHECKPOINT_KEY: {"version": T.TERRAIN_VERSION, "levels": None}},
  }

  def load(self, path, load_cfg=None, strict=True, map_location=None) -> dict:
    return deepcopy(saved[path])

  monkeypatch.setattr(SplitOnPolicyRunner, "load", load)
  runner = TerrainSplitRunner.__new__(TerrainSplitRunner)
  with pytest.raises(ValueError, match="FLAT-ground stage-2 checkpoint"):
    runner.load("flat.pt")
  with pytest.raises(ValueError, match="osl_terrain_v0"):
    runner.load("old.pt", load_cfg={"actor": True, "critic": True})
  assert runner.load("flat.pt", load_cfg={"actor": True}) == saved["flat.pt"]
  assert runner.load("terrain.pt") == saved["terrain.pt"]


##
# Bundles.
##


def test_a_run_spec_follows_its_env_yaml(tmp_path) -> None:
  from mjlab.tasks.osl_gait_library.export import read_config

  flat = _stage1_run(
    tmp_path / "flat", osl_gait_library_env_cfg(mode="train", variant="unicycle")
  )
  assert (
    T.run_terrain_spec(read_config(Path(flat["stage1_run_dir"]) / "params/env.yaml"))
    is None
  )
  cfg = cast(Any, osl_gait_library_terrain_env_cfg())
  cfg.scene.terrain.terrain_generator.seed = 3
  run = _stage1_run(tmp_path / "terrain", cfg)
  spec = T.run_terrain_spec(
    read_config(Path(run["stage1_run_dir"]) / "params/env.yaml")
  )
  assert spec is not None and spec["seed"] == 3
  section = T.manifest_section(spec)
  assert section["kind"] == "hfield_generator"
  assert section["file"] == "terrain_generator.json" == T.SPEC_FILE
  assert section["spec_sha256"] == T.spec_digest(spec)


def test_a_terrain_run_exports_its_spec(tmp_path, library) -> None:  # noqa: F811
  """A stage-1 terrain run's bundle carries terrain_generator.json (checksummed) and
  ``manifest.terrain_generator``; the rest is the flat bundle's. Neither the
  ``terrain`` section nor a ``terrain.json`` (the stairs bundles' box terrain, which
  osl_deploy compiles and checks) is written, so osl_deploy loads the bundle as a
  flat one."""
  from mjlab.tasks.osl_gait_library import export as gl_export

  flat_ckpt = _make_run(tmp_path / "flat", ROBOT, library)
  terrain_ckpt = _make_run(tmp_path / "terrain", ROBOT, library)
  env_cfg = osl_gait_library_terrain_env_cfg("robot", "rigid", "mimic")
  env_cfg.commands["motion"].library_dir = str(library)  # type: ignore[attr-defined]
  dump_yaml(terrain_ckpt.parent / "params" / "env.yaml", asdict(env_cfg))
  flat = gl_export.export_bundle(flat_ckpt)
  bundle = gl_export.export_bundle(terrain_ckpt)
  manifest = json.loads((bundle / "manifest.json").read_text())
  flat_manifest = json.loads((flat / "manifest.json").read_text())
  assert T.MANIFEST_KEY == "terrain_generator" != "terrain"
  for folder, m in ((flat, flat_manifest), (bundle, manifest)):
    assert "terrain" not in m and not (folder / "terrain.json").exists()
  assert T.MANIFEST_KEY not in flat_manifest
  assert not (flat / T.SPEC_FILE).exists()
  spec = json.loads((bundle / T.SPEC_FILE).read_text())
  assert spec["kind"] == "hfield_generator" and spec["seed"] == T.TERRAIN_SEED
  assert manifest[T.MANIFEST_KEY] == T.manifest_section(spec)
  assert manifest["files"][T.SPEC_FILE] == gl_export.sha256(bundle / T.SPEC_FILE)
  assert manifest["run"]["task"] == flat_manifest["run"]["task"] + "-Terrain"
  for key in ("model.mjb", "reference.npz", "parity.npz"):
    assert manifest["files"][key] == flat_manifest["files"][key], key
  drop = {T.MANIFEST_KEY, "files", "run", "provenance"}
  assert {k: v for k, v in manifest.items() if k not in drop} == {
    k: v for k, v in flat_manifest.items() if k not in drop
  }


class _Built(Exception):
  """Raised where the evaluation would build its runner."""


def test_evaluate_follows_the_run_or_the_terrain_flag(
  tmp_path,
  library,  # noqa: F811
  monkeypatch,
) -> None:
  """evaluate.py builds the terrain twin and its runner for a terrain run, the flat
  task for a flat run, and ``--terrain`` overrides either way (a flat control on
  the frozen terrain stress, a terrain run on flat stress)."""
  import mjlab.envs
  import mjlab.rl
  from mjlab.tasks.osl_gait_library import evaluate as ev
  from mjlab.tasks.osl_gait_library.rl import runner as library_runner
  from mjlab.tasks.osl_gait_library.rl import terrain_runner

  flat = _make_run(tmp_path / "flat", ROBOT, library)
  terrain = _make_run(tmp_path / "terrain", ROBOT, library)
  env_cfg = osl_gait_library_terrain_env_cfg("robot", "rigid", "mimic")
  env_cfg.commands["motion"].library_dir = str(library)  # type: ignore[attr-defined]
  dump_yaml(terrain.parent / "params" / "env.yaml", asdict(env_cfg))
  built: list[Any] = []

  def runner(name: str) -> Any:
    def make(env: Any, train_cfg: dict, *args: Any, **kwargs: Any) -> Any:
      raise _Built(name)

    return make

  monkeypatch.setattr(
    mjlab.envs, "ManagerBasedRlEnv", lambda cfg, **k: built.append(cfg)
  )
  monkeypatch.setattr(mjlab.rl, "RslRlVecEnvWrapper", lambda env: env)
  monkeypatch.setattr(library_runner, "GaitLibraryOnPolicyRunner", runner("flat"))
  monkeypatch.setattr(terrain_runner, "TerrainGaitLibraryRunner", runner("terrain"))
  cases = ((flat, None, False), (flat, True, True), (terrain, None, True))
  for checkpoint, flag, on_terrain in (*cases, (terrain, False, False)):
    cfg = ev.EvaluateConfig(checkpoint, library=library, device="cpu", terrain=flag)
    with pytest.raises(_Built) as raised:
      ev.evaluate(cfg)
    assert str(raised.value) == ("terrain" if on_terrain else "flat")
    assert T.is_terrain_cfg(built[-1]) is on_terrain
    assert (TERRAIN_CURRICULUM in built[-1].curriculum) is on_terrain
