# OSL terrain spec (`hfield_generator`, `osl_terrain_v1`)

The `...-Terrain` tasks (stage 1 `Mjlab-Gait-Library-Unicycle-OSL-Fullbody-*-Terrain`,
stage 2 `Mjlab-Split-Unicycle-OSL-Fullbody-*-Terrain`) train on heightfields made by
mjlab's `TerrainGenerator` from the settings in `terrain.py`. A terrain run writes
them to `params/terrain_generator.json`, and its bundles (stage-1 or dual) ship the
same file, checksummed in `manifest.files`, with the section
`manifest.terrain_generator` (`kind = "hfield_generator"`).

`terrain_generator.json` is self-contained: with NumPy and nothing else (no mjlab,
not this document) it rebuilds the heightfield of any `(seed, type, level)`
exactly. `tiles.common` and `types.<name>.formula` are each type's tile as exact
Python statements (`tiles.language` says how to run them; `tests/test_osl_terrain.py`
runs them against mjlab's compiled heightfields), `rng` the draw order, `mujoco` the
heightfield conversion, `grid` the layout and border, and `spawn.rule` the start
height. This document explains the same rules in prose. `tests/osl_terrain_port.py`
is a NumPy-only port written from them (its `window_max` from `spawn.rule`), and
`tests/fixtures/osl_terrain_heights.npz` holds tiles a port must match.

**Names.** The section is deliberately not `manifest.terrain` and the file not
`terrain.json`: those are the stairs bundles' (TERRA's box list, compiled into
model.mjb), and osl_deploy (main 6038d1e) refuses a bundle whose `terrain` section
is anything else. Under the names above, today's osl_deploy loads a terrain bundle
as a flat bundle (its model.mjb is the flat training scene), so flat panels of a
terrain run work before osl_deploy learns the generator; the osl_deploy terrain
work adds it under `terrain_generator`.

## Grid

- Types (columns, world +y): `rough`, `waves`, `obstacles` (`grid.types`).
- Levels (rows, world +x): 0..9 (`grid.num_levels`). Level 0 of every type is flat.
- Tiles are `16 x 16` m (`grid.tile_size_m`). The grid is centred on the world origin:
  tile `(level r, type c)` has its corner at
  `x0 = -num_levels * 16 / 2 + r * 16`, `y0 = -num_types * 16 / 2 + c * 16`, i.e. x in
  [-80, 80], y in [-24, 24].
- A flat border 20 m wide surrounds the grid: four boxes 1 m tall whose top is z = 0
  (`grid.border_boxes_m`, MuJoCo half sizes and positions: `+y` and `-y` boxes of
  half size `(100, 10, 0.5)` at `(0, +-34, -0.5)`, `-x` and `+x` boxes of half size
  `(10, 24, 0.5)` at `(-+90, 0, -0.5)`), with the plane's contact parameters.
- difficulty `d = 0.0 + (1.0 - 0.0) * t`, `t = level / max(num_levels - 1, 1)`, in
  Python floats (so `d(3) = 0.3333333333333333`).

## Random numbers

One `numpy.random.default_rng(seed)` (PCG64) for the whole grid. Tiles are generated
type by type, level 0..9 within each type: `(rough,0) .. (rough,9), (waves,0) ..
(waves,9), (obstacles,0) .. (obstacles,9)`. Only `rough` and `obstacles` draw; to get
one tile, generate the grid up to it. The calls are NumPy `Generator.choice` exactly as
below (`rng.numpy_version` records the NumPy that wrote it; the fixture test fails if
another NumPy draws differently). The training ground is `seed` (0), stress
validation's `stress_seed` (1); a deployment may use any seed.

## Tiles (generator arrays)

Each tile is an int16 array `noise[W, L]` of heights in units of `vertical_scale`
(0.001 m), `W = int(16 / horizontal_scale) = 160` along the generator's first index and
`L = 160` along its second; `horizontal_scale = 0.1`. `types.<name>.params` holds every
setting, `types.<name>.levels` the integers each level derives (check your port
against them). All arithmetic below is Python float arithmetic, in this order. The
JSON's `types.<name>.formula` statements are the normative form of the same rules.

**rough** (mjlab `HfRandomUniformTerrainCfg`, `scale_with_difficulty`):
`lo = noise_range[0] * d`, `hi = noise_range[1] * d`; `hmin = int(lo / vs)`,
`hmax = int(hi / vs)`, `hstep = int(noise_step / vs)`;
`noise = rng.choice(np.arange(hmin, hmax + hstep, hstep), size=(W, L)).astype(int16)`.
(mjlab passes this through a cubic spline evaluated at its own nodes, which returns
the same integers.)

**waves** (`FeatheredHfWaveTerrainCfg`; draws nothing):
`amp = amplitude_range[0] + d * (amplitude_range[1] - amplitude_range[0])`;
`a = int(0.5 * amp / vs)`; `k = 2 pi / (L / num_waves)`;
`raw[i, j] = a * (cos(j * k) + sin(i * k))` for `i < W`, `j < L`;
taper `w(n)[i] = 0.5 - 0.5 cos(pi * min(min(i, n - 1 - i) * horizontal_scale /
edge_width, 1))`; `noise = rint(raw * outer(w(W), w(L))).astype(int16)`.
So the surface spans `+-amp` (`2a` units): level 9 is `+-6 cm`, 12 cm peak to
peak (mjlab's amplitude convention, in which G1's `amplitude_range (0, 0.2)` is
"waves 0-20 cm").

**obstacles** (mjlab `HfDiscreteObstaclesTerrainCfg`, `obstacle_height_mode =
"fixed"`): `h = obstacle_height_range[0] + d * (range[1] - range[0])`,
`hu = int(h / vs)`; `wmin = int(width_range[0] / horizontal_scale)`,
`wmax = int(width_range[1] / horizontal_scale)`, `widths = arange(wmin, wmax + 1, 4)`
(`[wmin]` if empty); `noise = zeros((W, L), int16)`; then `num_obstacles` times:
`w = rng.choice(widths)`, `l = w if square_obstacles else rng.choice(widths)`,
`x = rng.choice(arange(0, W, 4))`, `y = rng.choice(arange(0, L, 4))`,
`noise[x:min(x + w, W), y:min(y + l, L)] = hu`. Finally the centre platform:
`p = int(platform_width / horizontal_scale)`, `cx, cy = W // 2, L // 2`, `hp = p // 2`,
`noise[max(cx - hp, 0):min(cx + hp, W), max(cy - hp, 0):min(cy + hp, L)] = 0`.

## MuJoCo heightfield of a tile

- `emin, emax = noise.min(), noise.max()`; `span = emax - emin if emax != emin else 1`.
- `nrow = W`, `ncol = L`; `data = ((noise - emin) / span).astype(float32)`, row-major
  (MuJoCo `hfield_data`).
- `size = (8.0, 8.0, span * vs, max(span * vs * base_thickness_ratio, min_base))`
  (`min_base` 0.05 m: a thin flat tile would let a loaded foot reach its bottom).
- geom `pos = (x0 + 8, y0 + 8, z)`, identity orientation, with `z = 0` (rough,
  obstacles) or `z = -size[2] / 2` (waves).
- MuJoCo puts vertex `(row i, column j)` at `x = pos.x - size[0] + j * 2 size[0] /
  (ncol - 1)`, `y = pos.y - size[1] + i * 2 size[1] / (nrow - 1)` (so the generator's
  first index runs along world y), height `pos.z + data[i * ncol + j] * size[2]`.
  Each cell is two triangles split along the `(i, j)`-`(i + 1, j + 1)` diagonal.
- Geom friction, solref and solimp are MuJoCo's defaults, as the training plane's.

A deployment builds the tiles it needs (one tile, the grid, or a tile repeated), adds
them in place of the bundle model's plane geom `terrain`, and compiles. **It removes
the plane and surrounds its tiles with the same border**: for tiles covering
`[x0, x1] x [y0, y1]`, four boxes 20 m wide and 1 m tall, top at z = 0, laid out as
`grid.border_boxes_m` around that extent (mjlab `make_border`: the `+-y` boxes span
`x1 - x0 + 40` m, the `+-x` boxes fill `y1 - y0` between them; `terrain.border_boxes`
computes them), with the default contact parameters. A 20 s episode at 1.5 m/s
covers ~30 m and a tile centre sits 8 m from its edge, so episodes do reach the
border. A z = 0 plane is not equivalent: wave tiles reach 6 cm below z = 0 and a
plane would cut through their troughs.

## Start height

An episode's reference is the flat library pose (on the plane at z = 0), raised
rigidly by `h`: for each foot site (`spawn.sites`: `talus_l`, `toes_l`,
`OSL_KA_L1_osl_ankle_assembly`, `OSL_KA_L1_osl_foot_assembly`) take the heightfield
vertices of every cell that meets the square of half-width `r = spawn.radius_m`
(0.25) around the site's xy, and `h` = the highest of them over all sites.

In indices (`spawn.rule`): the site's tile is the one whose half-open area
`[x0, x0 + 16) x [y0, y0 + 16)` holds its `(x, y)`; with `(vx, vy)` that tile's
vertex `(0, 0)` and `sx = sy = 16 / 159` its vertex spacing, `u = (x - vx) / sx`,
`v = (y - vy) / sy` (float64), and the site's value is the highest world height
(float32) over rows `i` in `floor(v - r / sy) .. ceil(v + r / sy)` and columns `j`
in `floor(u - r / sx) .. ceil(u + r / sx)`, both ranges clipped to `0 .. 159`. The
square's reach is therefore up to one vertex spacing (0.1 m) beyond `r`: a port
that takes only the vertices inside the square reads up to ~4 cm lower next to an
obstacle edge or a wave crest.

**Only cells of the site's own tile count**: the square is clipped to that tile,
never reaching into a neighbour, and a site off the grid reads 0. (Including the
neighbour's cells changes `h` by up to 4 cm within 0.25 m of a seam.)
The height stays fixed for the episode (as on flat ground). Training starts envs at
their tile's centre.

## Training facts recorded for comparison

Curriculum: +1 level after a time-out, -1 after an early termination, frozen on the
first reset, a promotion past level 9 draws a random level, start levels random 0..5.
Stress validation: `stress_seed`, env `i` on type `i % 3`, level `(i // 3) % 10`,
survival reported per band of levels 0-2 / 3-6 / 7-9.

## Known differences for the deploy port (measured 2026-11-20, CPU checks)

- **Touch zones on heightfields.** A capsule-heightfield contact is made per prism
  along the capsule, so part of the OSL foot's load lands between its `r_foot`
  (heel) and `r_toes` zones; on the plane only the end caps touch. OSL-foot capture
  (zone touch sum over the foot's normal load; 1.00 on the plane in both engines):
  - mujoco_warp (training), on the terrain levels (round-6 `s2_r6s7_C` dual policy
    on the frozen stress layout, 60 envs x 200 steps): levels 0-2 **0.93** (29% of
    loaded states below 0.95), 3-6 **0.88** (33%), 7-9 **0.91** (26%); per type
    rough 0.90, waves 0.90, obstacles 0.91. Same states of the round-6 seed-7
    teacher on level-0 heightfields: 0.93-0.94, with the OSL touch sum ~6% above
    the plane's (median), mostly in the toe zone (toe 181 vs 142 N, heel 386 vs
    365 N);
  - C MuJoCo (osl_deploy): 0.87-0.91 per band (22-61% of loaded states below 0.95),
    0.87 on the level-0 states above (73% below 0.95).

  The left foot captures 1.00 on both engines. So **on terrain the OSL heel/toe
  touch reads ~7-12% below the foot's load in both engines**: the shift from flat
  ground (1.00) is the same order in training and in a C-MuJoCo deploy above level
  0 (C keeps every per-prism contact, warp at most 4 per heightfield pair; only on
  level 0 does warp capture visibly more, 0.93 vs 0.87). It feeds the human's touch
  input and stage 2's heel/toe inputs. The touch zones are an observation contract
  (`touch_sites`) and are unchanged.
- **Contact force.** For the same state the normal force on a level-0 heightfield is
  1.08-1.25x (OSL) and 1.17-1.20x (left) the plane's in warp, 1.25x and 1.32x in C
  MuJoCo (the heightfield contact sits ~1.5 mm deeper than the plane's).
- **Root height.** The actor's root height input is world z (`root_pos_no_xy`), as on
  flat ground, so on terrain it rises with the ground: a weak ground-height cue to a
  blind policy (start heights reach 3.3 cm at levels 0-5, 6.0 cm at level 9; the
  round-6 normaliser's root-z std is 4.4 cm). The reference keeps its episode start
  height, so at the top levels it sits ~3-5 cm above where the robot walks.
