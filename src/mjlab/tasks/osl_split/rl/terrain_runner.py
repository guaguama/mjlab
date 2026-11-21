"""The stage-2 terrain runner: unicycle stage 2's, on stage 1's generated terrain.

:class:`TerrainSplitRunner` is :class:`~.runner.SplitOnPolicyRunner` with stage 1's
terrain additions (:class:`~mjlab.tasks.osl_gait_library.rl.terrain_runner.
TerrainRunnerMixin`: terrain levels in checkpoints and restored on resume, frozen
stress validation with level bands, flat validation, terrain videos). The envs
start at random levels 0..5, as stage 1's, whatever the teacher reached; the
curriculum is stage 1's survival rule.

**The warm start must come from a terrain teacher** (a stage-1 ``...-Terrain`` run of
the same terrain version): stage 2 cannot add conditions the human never trained on
(the user's rule, round 7). A teacher whose ``params/env.yaml`` trains on flat
ground, or that cannot be read, is refused, with no override; another terrain
version is refused too (the ground's distribution would change under the human).
For the same reason a training resume (``--agent.resume``) must load a terrain
stage-2 checkpoint of this terrain version: a flat stage-2 run's checkpoint, which
records no terrain levels, is refused (:func:`check_resume_record`).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mjlab.tasks.osl_gait_library import terrain as osl_terrain
from mjlab.tasks.osl_gait_library.export import read_config
from mjlab.tasks.osl_gait_library.rl.terrain_runner import (
  TerrainRunnerMixin,
  generator_of,
)
from mjlab.tasks.osl_mimic.rl.runner import resumes_training

from .runner import SplitOnPolicyRunner
from .warmstart import WarmStart

STAGE1_TERRAIN_TASKS = "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-*-Terrain"
RULE = (
  "stage 2 cannot add conditions to the warm-started human (the user's rule, round "
  "7): train the human on terrain in stage 1 first"
)


def check_terrain_teacher(
  lineage: Mapping[str, Any], generator: Any
) -> dict[str, Any] | None:
  """Check that the warm start's stage-1 run trained on this terrain; returns the
  teacher checkpoint's terrain record (its final levels), if it has one.

  Raises:
    ValueError: For a teacher on flat ground, one whose ``params/env.yaml`` is
      missing, or one on another terrain version than ``generator`` (the seed
      alone may differ: it changes the tiles, not their distribution).
  """
  run_dir = lineage.get("stage1_run_dir")
  source = run_dir or "the warm start's stage-1 run"
  env_yaml = Path(run_dir) / "params" / "env.yaml" if run_dir else None
  if env_yaml is None or not env_yaml.is_file():
    raise ValueError(
      f"A terrain stage 2 needs a terrain teacher, and {source} has no readable "
      f"params/env.yaml to show its ground: {RULE} ({STAGE1_TERRAIN_TASKS}), then "
      "warm-start from that run's directory."
    )
  saved = osl_terrain.saved_terrain(read_config(env_yaml))
  if saved is None:
    raise ValueError(
      f"The warm start {lineage.get('stage1_checkpoint', source)} comes from a "
      f"FLAT-ground stage-1 run ({env_yaml} has no generated terrain), and this "
      f"stage 2 trains on terrain: {RULE} ({STAGE1_TERRAIN_TASKS}), then "
      "warm-start from that run. A flat teacher trains the flat split task."
    )
  try:
    osl_terrain.check_saved_generator(saved)
  except ValueError as err:
    raise ValueError(
      f"The warm start's stage-1 run {source} trained on other terrain than this "
      f"stage 2 ({osl_terrain.TERRAIN_VERSION}): {err}"
    ) from err
  if int(saved.get("seed", -1)) != int(generator.seed):
    print(
      f"[WARN] The teacher's terrain seed {saved.get('seed')} differs from stage 2's "
      f"{generator.seed}: the same generator on other tiles.",
      flush=True,
    )
  checkpoint = lineage.get("stage1_checkpoint")
  if not checkpoint or not Path(checkpoint).is_file():
    return None
  data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  record = (data.get("infos") or {}).get(osl_terrain.CHECKPOINT_KEY)
  return dict(record) if isinstance(record, Mapping) else None


def check_resume_record(infos: Mapping[str, Any], path: str) -> None:
  """Check that a stage-2 checkpoint a terrain stage 2 resumes training from trained
  on this terrain: it records terrain levels (``infos["terrain"]``) of this
  terrain version.

  Raises:
    ValueError: For a checkpoint without the record (a flat stage-2 run's) or
      with another terrain version's.
  """
  record = infos.get(osl_terrain.CHECKPOINT_KEY)
  if not isinstance(record, Mapping):
    raise ValueError(
      f"{path} records no terrain levels: it is a FLAT-ground stage-2 checkpoint, "
      f"and this stage 2 trains on terrain. {RULE[0].upper()}{RULE[1:]} "
      f"({STAGE1_TERRAIN_TASKS}), then warm-start a terrain stage 2 from that run. "
      "Resume a flat stage-2 run with its flat split task."
    )
  version = record.get("version")
  if version != osl_terrain.TERRAIN_VERSION:
    raise ValueError(
      f"{path} trained on terrain {version!r}, this stage 2 on "
      f"{osl_terrain.TERRAIN_VERSION!r}: the human's ground would change ({RULE})."
    )


class TerrainSplitRunner(TerrainRunnerMixin, SplitOnPolicyRunner):
  """The stage-2 terrain runner (module docstring)."""

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """The terrain runner's load; a load of what a training resume needs must be
    of a terrain stage-2 checkpoint (:func:`check_resume_record`). ``play.py``'s
    actor-only load is not checked."""
    infos = super().load(path, load_cfg, strict, map_location)
    if resumes_training(load_cfg):
      check_resume_record(infos, path)
    return infos

  def warm_start(self, ws: WarmStart) -> None:
    """Refuse a teacher that did not train on this terrain
    (:func:`check_terrain_teacher`), then unicycle stage 2's warm start."""
    generator = generator_of(self.env.unwrapped.cfg)
    record = check_terrain_teacher(ws.lineage, generator)
    super().warm_start(ws)
    if record is not None and record.get("levels") is not None:
      levels = torch.as_tensor(record["levels"]).float()
      types = torch.as_tensor(record["types"]).long()
      parts = [
        f"{name} {float(levels[types == i].mean()):.2f}"
        for i, name in enumerate(generator.sub_terrains)
        if bool((types == i).any())
      ]
      print(
        f"[INFO] The teacher ended at mean terrain levels {', '.join(parts)}; stage 2 "
        f"starts at random levels 0..{osl_terrain.MAX_INIT_LEVEL}.",
        flush=True,
      )
