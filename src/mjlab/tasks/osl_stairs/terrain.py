"""Static box terrain from a TERRA TerrainSpec JSON, added to the scene as geometry.

A stairs clip comes as two files written side by side: clipper's ``--format mjlab``
motion NPZ ``<clip>.npz`` and TERRA's terrain sidecar ``<clip>_terrain.json`` (TERRA's
own pairing convention, which clipper's ``terra`` source also reads). The JSON is read
here directly, in clipper's and TERRA's plain schema:

* ``boxes``: each a world-frame centre ``pos`` (m), MuJoCo half-extents ``size`` (m)
  and ``yaw``/``pitch`` (rad), oriented ``Rz(yaw) @ Ry(pitch)``, plus a ``name``;
* ``provenance``: TERRA's reconstruction record, not read here;
* an implicit floor at z = 0, which is mjlab's terrain plane.

No coordinate is moved. The clip's root and the boxes share TERRA's world frame, so
the task puts both exactly there (see ``mdp.StairsMotionCommand``). A malformed file
raises rather than silently dropping obstacles, as clipper's loader does.

The boxes join mjlab's terrain body (``terrain``, which holds the floor plane), so the
floor and the stairs are one static body with the plane's contact parameters: MuJoCo's
defaults, friction 1.0. The OSL tasks' foot friction randomization reaches stair
contacts through the foot geoms' contact priority, as it reaches floor contacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

TERRAIN_SUFFIX = "_terrain.json"
"""The sidecar of ``<clip>.npz`` is ``<clip>_terrain.json`` in the same folder."""
TERRAIN_BODY = "terrain"
"""mjlab's terrain body, holding the floor plane (``TerrainEntity``)."""
BOX_PREFIX = "terrain_box_"
"""Box ``i`` is the geom ``terrain_box_<i>``, in file order (TERRA's own naming)."""
BOX_RGBA = (0.62, 0.58, 0.52, 1.0)


@dataclass(frozen=True)
class TerrainBox:
  """One static box: TERRA's ``BoxSpec``."""

  pos: tuple[float, float, float]
  """World-frame centre (m)."""
  size: tuple[float, float, float]
  """Half-extents (m), as MuJoCo's box ``size``."""
  yaw: float = 0.0
  pitch: float = 0.0
  """Orientation ``Rz(yaw) @ Ry(pitch)`` (rad)."""
  name: str = ""

  def __post_init__(self) -> None:
    for key in ("pos", "size"):
      value = np.asarray(getattr(self, key), dtype=np.float64)
      if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"box {key} must hold three finite values")
      object.__setattr__(self, key, tuple(float(v) for v in value))
    if min(self.size) <= 0.0:
      raise ValueError("box size must hold positive half-extents")
    for key in ("yaw", "pitch"):
      value = float(getattr(self, key))
      if not math.isfinite(value):
        raise ValueError(f"box {key} must be finite")
      object.__setattr__(self, key, value)

  @property
  def quat(self) -> np.ndarray:
    """(4,) MuJoCo (w, x, y, z) of ``Rz(yaw) @ Ry(pitch)``."""
    cy, sy = math.cos(self.yaw / 2.0), math.sin(self.yaw / 2.0)
    cp, sp = math.cos(self.pitch / 2.0), math.sin(self.pitch / 2.0)
    # (cy, 0, 0, sy) * (cp, 0, sp, 0)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


def load_terrain(path: str | Path) -> tuple[TerrainBox, ...]:
  """The boxes of a TerrainSpec JSON. ``{"boxes": []}`` is valid flat terrain."""
  path = Path(path)
  try:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
      raise ValueError("expected a TerrainSpec object with a 'boxes' list")
    boxes = []
    for i, box in enumerate(value["boxes"]):
      if not isinstance(box, dict):
        raise ValueError(f"box {i}: expected an object")
      try:
        boxes.append(TerrainBox(**box))
      except (TypeError, ValueError) as err:
        raise ValueError(f"box {i}: {err}") from err
    return tuple(boxes)
  except (TypeError, ValueError) as err:
    raise ValueError(f"{path}: invalid terrain: {err}") from err


def terrain_sidecar(motion_file: str | Path) -> Path:
  """``<clip>_terrain.json`` beside ``<clip>.npz``."""
  motion = Path(motion_file)
  return motion.with_name(motion.stem + TERRAIN_SUFFIX)


def resolve_terrain_file(motion_file: str, terrain_file: str = "") -> Path:
  """The terrain a clip is played on: ``terrain_file`` if given, else the motion's
  sidecar. A missing file raises: a stairs clip must never fall back to a flat floor
  without saying so."""
  if terrain_file:
    path = Path(terrain_file)
  elif motion_file:
    path = terrain_sidecar(motion_file)
  else:
    raise ValueError(
      "No terrain: pass --env.commands.motion.motion-file <clip.npz> (its "
      f"<clip>{TERRAIN_SUFFIX} is loaded) or set the task's motion environment "
      "variable (see the task's docstring)."
    )
  if not path.is_file():
    raise FileNotFoundError(
      f"Terrain file not found: {path}. A stairs clip <clip>.npz needs its "
      f"<clip>{TERRAIN_SUFFIX} beside it, or --env.commands.motion.terrain-file."
    )
  return path


def add_terrain_boxes(spec: mujoco.MjSpec, boxes: tuple[TerrainBox, ...]) -> None:
  """Add ``boxes`` to the scene's terrain body as static, colliding geoms."""
  body = spec.body(TERRAIN_BODY)
  if body is None:
    raise ValueError(
      f"The scene has no '{TERRAIN_BODY}' body: box terrain needs mjlab's terrain."
    )
  for i, box in enumerate(boxes):
    body.add_geom(
      name=f"{BOX_PREFIX}{i}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=box.pos,
      quat=box.quat.tolist(),
      size=box.size,
      rgba=BOX_RGBA,
      # Static: a mass would only inflate stat.meanmass (as in TerrainGenerator).
      mass=0.0,
    )


@dataclass(frozen=True)
class TerrainBoxes:
  """``SceneCfg.spec_fn`` that adds a clip's box terrain to the scene.

  It holds the motion command's two paths, not the boxes, and reads the file when the
  scene is built. The env config keeps it in step with the command
  (``config.env_cfgs.sync_terrain``), and the command checks the built scene against
  its own file, so a mismatch fails loudly.
  """

  motion_file: str = ""
  terrain_file: str = ""

  def resolve(self) -> Path:
    return resolve_terrain_file(self.motion_file, self.terrain_file)

  def __call__(self, spec: mujoco.MjSpec) -> None:
    add_terrain_boxes(spec, load_terrain(self.resolve()))


def scene_boxes(model: mujoco.MjModel) -> list[int]:
  """Geom ids of the scene's terrain boxes, in box order."""
  ids = []
  while (
    gid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{BOX_PREFIX}{len(ids)}")
  ) >= 0:
    ids.append(gid)
  return ids


def check_scene_terrain(
  model: mujoco.MjModel, boxes: tuple[TerrainBox, ...], source: Path
) -> None:
  """Raise unless the compiled scene holds exactly ``boxes``, in place."""
  ids = scene_boxes(model)
  problem = ""
  if len(ids) != len(boxes):
    problem = f"the scene has {len(ids)} terrain boxes, the file {len(boxes)}"
  else:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TERRAIN_BODY)
    at_origin = np.allclose(model.body_pos[body], 0.0) and np.allclose(
      model.body_quat[body], (1.0, 0.0, 0.0, 0.0)
    )
    for i, (gid, box) in enumerate(zip(ids, boxes, strict=True)):
      quat = model.geom_quat[gid]
      same = (
        at_origin
        and np.allclose(model.geom_pos[gid], box.pos, atol=1e-6)
        and np.allclose(model.geom_size[gid], box.size, atol=1e-6)
        and abs(float(np.dot(quat, box.quat))) > 1.0 - 1e-9
      )
      if not same:
        problem = f"box {i} differs"
        break
  if problem:
    raise ValueError(
      f"The scene's terrain does not match {source}: {problem}. The scene's terrain "
      "follows the motion command's motion_file/terrain_file when the env config is "
      "built; after changing them on a built config, call "
      "mjlab.tasks.osl_stairs.config.env_cfgs.sync_terrain(cfg)."
    )
