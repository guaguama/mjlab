"""The stairs stage-2 runner: unicycle stage 2's, on one clip and its terrain.

:class:`~mjlab.tasks.osl_split.rl.runner.SplitOnPolicyRunner` assumes a motion
library in four places, each inside its evaluation guard: without the overrides here
a stairs run would train with stress validation alone, printing only warnings. On
the stairs, as in stairs stage 1:

* validation and video envs are this task's validation and play modes (videos on
  the nominal plant) on the training env's clip and terrain, with the training run's
  settings and the training OSL observation read exactly, each bus at the middle of
  its training lag, as unicycle stage 2's; stress validation plays a copy of the
  training config, as unicycle stage 2's;
* validation plays ``validation.num_envs`` envs (100) from frame 0, for one clip
  length unless ``validation.num_steps`` says otherwise, and reports no clip groups.

The run's ``params/env.yaml`` records only the terrain's path, so ``learn`` copies
the terrain JSON beside it as ``params/terrain.json``, as stairs stage 1 does, and
every checkpoint records the clip and the terrain's digest (``infos["stairs"]``); a
resume or play on another clip or terrain draws a warning.

The warm start must come from a stairs stage-1 run (``Mjlab-Stairs-OSL-Fullbody-*``):
flat ``Mjlab-Mimic-OSL-*`` runs share its 2333 human inputs and would load. A run
whose ``params/env.yaml`` trains without a terrain, or that has no
``params/terrain.json``, is refused; another clip or terrain than stage 2's draws a
warning (the run directory is the one the warm start records).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.osl_gait_library.config.dr import nominal_plant
from mjlab.tasks.osl_gait_library.export import read_config
from mjlab.tasks.osl_mimic.rl.runner import (
  EvalPurpose,
  OslMimicOnPolicyRunner,
  eval_mode,
)
from mjlab.tasks.osl_split.mdp import (
  OSL_OBS_GROUP,
  osl_validation_group,
  osl_validation_lags,
)
from mjlab.tasks.osl_split.rl.runner import SplitOnPolicyRunner
from mjlab.tasks.osl_split.rl.warmstart import WarmStart
from mjlab.tasks.osl_split_stairs.config.env_cfgs import osl_split_stairs_env_cfg
from mjlab.tasks.osl_stairs.config.env_cfgs import sync_terrain
from mjlab.tasks.osl_stairs.mdp import StairsMotionCommandCfg
from mjlab.tasks.osl_stairs.rl.runner import TERRAIN_COPY, copy_terrain
from mjlab.tasks.osl_stairs.terrain import resolve_terrain_file

STAIRS_INFO_KEY = "stairs"
"""A checkpoint's ``infos`` key for the clip and terrain it trained on
(:func:`stairs_identity`)."""
STAGE1_TASKS = "Mjlab-Stairs-OSL-Fullbody-*"


class StairsSplitRunner(SplitOnPolicyRunner):
  """The stairs stage-2 runner; see the module docstring."""

  ##
  # Warm start.
  ##

  def warm_start(self, ws: WarmStart) -> None:
    """Check that ``ws`` comes from a stairs stage-1 run on this run's clip and
    terrain (:func:`check_stairs_warm_start`), then start as unicycle stage 2.

    Raises:
      ValueError: For a stage-1 run that is not a stairs run, and for unicycle
        stage 2's refusals.
    """
    for warning in check_stairs_warm_start(ws.lineage, self.env.unwrapped.cfg):
      print(f"[WARN] {warning}", flush=True)
    super().warm_start(ws)

  def _check_stage1_variant(self, lineage: Mapping[str, Any]) -> None:
    """Nothing: stairs stage-1 runs record no variant, and :meth:`warm_start` has
    checked the run itself."""
    del lineage

  ##
  # Training and checkpoints.
  ##

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    copy_terrain(self.env.unwrapped, self.logger)
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos: dict | None = None) -> None:
    """Unicycle stage 2's checkpoint, plus the clip and terrain the env trains on
    (``infos["stairs"]``)."""
    identity = stairs_identity(self.env.unwrapped.cfg)
    super().save(path, {**(infos or {}), STAIRS_INFO_KEY: identity})

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Unicycle stage 2's load, with a warning when the checkpoint trained on
    another clip or terrain than the env's."""
    infos = super().load(path, load_cfg, strict, map_location)
    recorded = infos.get(STAIRS_INFO_KEY)
    if isinstance(recorded, Mapping):
      current = stairs_identity(self.env.unwrapped.cfg)
      for line in identity_differences(recorded, current, "The checkpoint", "the env"):
        print(f"[WARN] {line}", flush=True)
    return infos

  ##
  # Validation: stairs stage 1's, past the gait library's.
  ##

  def _eval_env_cfg(self, purpose: EvalPurpose) -> ManagerBasedRlEnvCfg:
    """An evaluation env's config, made to score and observe what training does.

    Stress validation plays a copy of the training config (unicycle stage 2's
    rule). Validation is the task's validation mode and a video its play mode on
    validation's nominal plant, each on the training env's clip and terrain with
    its settings (:meth:`_inherit_training_settings`), then the training OSL
    observation read exactly (:func:`~mjlab.tasks.osl_split.mdp.
    osl_validation_group`), each bus at the middle of its training lag
    (:func:`~mjlab.tasks.osl_split.mdp.osl_validation_lags`).

    Raises:
      TypeError: If the training env does not play a stairs clip.
    """
    if purpose == "stress":
      return super()._eval_env_cfg(purpose)
    train = self.env.unwrapped.cfg
    train_motion = train.commands.get("motion")
    if not isinstance(train_motion, StairsMotionCommandCfg):
      raise TypeError(
        "Not a stairs training config: its motion command is a "
        f"{type(train_motion).__name__}."
      )
    v = self.validation_cfg
    cfg = osl_split_stairs_env_cfg(
      v["model"], v["socket"], v["reward"], eval_mode(purpose)
    )
    if purpose == "video":
      nominal_plant(cfg)
    motion = cfg.commands["motion"]
    assert isinstance(motion, StairsMotionCommandCfg)
    motion.motion_file = train_motion.motion_file
    motion.terrain_file = train_motion.terrain_file
    sync_terrain(cfg)
    self._inherit_training_settings(cfg)
    train_osl = train.observations[OSL_OBS_GROUP]
    cfg.observations[OSL_OBS_GROUP] = osl_validation_group(
      train_osl, osl_validation_lags(train_osl)
    )
    return cfg

  def _validation_num_envs(self) -> int:
    return OslMimicOnPolicyRunner._validation_num_envs(self)

  def _default_validation_steps(self, env: ManagerBasedRlEnv) -> int:
    return OslMimicOnPolicyRunner._default_validation_steps(self, env)

  def _validation_groups(self, env: ManagerBasedRlEnv) -> dict:
    return OslMimicOnPolicyRunner._validation_groups(self, env)


##
# The clip and terrain.
##


def stairs_identity(cfg: ManagerBasedRlEnvCfg) -> dict[str, str]:
  """The clip (its file name) and terrain (its JSON's SHA-256 digest) a stairs
  env config plays.

  Raises:
    TypeError: If ``cfg`` plays no stairs clip.
    FileNotFoundError: If the terrain file is missing.
  """
  motion = cfg.commands.get("motion")
  if not isinstance(motion, StairsMotionCommandCfg):
    raise TypeError(
      f"Not a stairs env config: its motion command is a {type(motion).__name__}."
    )
  terrain = resolve_terrain_file(motion.motion_file, motion.terrain_file)
  return {"clip": Path(motion.motion_file).name, "terrain_sha256": _sha256(terrain)}


def identity_differences(
  recorded: Mapping[str, Any], current: Mapping[str, Any], source: str, here: str
) -> list[str]:
  """How ``recorded`` (by ``source``) differs from ``current`` (``here``'s), as
  :func:`stairs_identity` gives them: one line per differing field it records."""
  lines = []
  clip = recorded.get("clip")
  if clip is not None and clip != current["clip"]:
    lines.append(
      f"{source} trained on the clip {clip}; {here} plays {current['clip']}."
    )
  digest = recorded.get("terrain_sha256")
  if digest is not None and digest != current["terrain_sha256"]:
    lines.append(
      f"{source} trained on another terrain (SHA-256 {str(digest)[:12]}) than "
      f"{here}'s ({current['terrain_sha256'][:12]}): the boxes moved or changed."
    )
  return lines


def check_stairs_warm_start(
  lineage: Mapping[str, Any], cfg: ManagerBasedRlEnvCfg
) -> list[str]:
  """Check that a warm start comes from a stairs stage-1 run, reading the run
  directory it records; returns warnings where that run's clip or terrain differs
  from ``cfg``'s.

  Stairs stage 1 copies its terrain into ``params/terrain.json`` as it starts, and
  its ``params/env.yaml`` records the clip and the terrain file's path.

  Raises:
    ValueError: If the run's ``params/env.yaml`` records a motion command without a
      terrain (a gait-library or flat-ground run), or the run has no
      ``params/terrain.json`` (not a stairs run, or not on this machine).
  """
  hint = f"Stairs stage 2 starts from a stairs stage-1 run ({STAGE1_TASKS})."
  run_dir = Path(str(lineage.get("stage1_run_dir") or ""))
  motion = _stage1_motion(run_dir)
  if motion is not None and "terrain_file" not in motion:
    library = motion.get("library_dir")
    trained = (
      f"on the motion library {library}" if library else "on flat ground (one clip)"
    )
    raise ValueError(
      f"The warm start's stage-1 run {run_dir} trained {trained}: its "
      f"params/env.yaml records no terrain. {hint}"
    )
  terrain = run_dir / "params" / TERRAIN_COPY
  if not terrain.is_file():
    raise ValueError(
      f"The warm start's stage-1 run directory {run_dir} has no "
      f"params/{TERRAIN_COPY}, which every stairs stage-1 run writes as it starts: "
      "it is not a stairs run (a flat Mjlab-Mimic-OSL-* run reads the same "
      f"2333 inputs), or its run directory is not on this machine. {hint}"
    )
  recorded: dict[str, Any] = {"terrain_sha256": _sha256(terrain)}
  if motion is not None and motion.get("motion_file"):
    recorded["clip"] = Path(str(motion["motion_file"])).name
  return identity_differences(recorded, stairs_identity(cfg), "Stage 1", "stage 2")


def _stage1_motion(run_dir: Path) -> Mapping[str, Any] | None:
  """The motion command of a stage-1 run's ``params/env.yaml``, as plain data; None
  without a readable one."""
  try:
    env = read_config(run_dir / "params" / "env.yaml")
  except (OSError, yaml.YAMLError, ValueError):
    return None
  commands = env.get("commands") if isinstance(env, Mapping) else None
  motion = commands.get("motion") if isinstance(commands, Mapping) else None
  return motion if isinstance(motion, Mapping) else None


def _sha256(path: Path) -> str:
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()
