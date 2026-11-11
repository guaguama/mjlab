"""The stairs environment: musclemimic-style imitation of one clip on its terrain.

Everything not listed here is ``osl_mimic_env_cfg(model, socket, reward, mode)``:
2 ms Euler steps at 100 Hz control, musclemimic's observation (blind: the stairs
show only through the reference goal), the MimicReward or CLF reward, random start
frames, the clip-end termination and musclemimic's training terminations. The
changes:

* **Terrain.** The clip's TERRA box terrain joins the floor plane as static scene
  geometry (:mod:`~mjlab.tasks.osl_stairs.terrain`), read from the motion's
  ``<clip>_terrain.json`` or ``--env.commands.motion.terrain-file``. Every env has the
  same boxes at their TERRA world coordinates, so the env origins all sit at the
  world origin (``env_spacing`` 0) and every episode starts at the reference's own
  world pose (:class:`~..mdp.StairsMotionCommand`).
* **Domain randomization**, the gait-library tasks' stage-1 set
  (``osl_gait_library.config.dr``): physical properties, pushes in training, OSL
  and human actuator delays; validation runs the nominal plant at mid-range delays.

The terrain follows the motion command: :class:`OslStairsEnvCfg` rebuilds the scene's
``spec_fn`` from the command's paths whenever the config is constructed, which tyro
does after applying command-line overrides. Code that changes the paths on a built
config calls :func:`sync_terrain`; the command rejects a scene whose terrain does not
match its own file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.osl_gait_library.config.dr import OslDrCfg, apply_osl_dr
from mjlab.tasks.osl_mimic.config.env_cfgs import Mode, Reward, osl_mimic_env_cfg
from mjlab.tasks.osl_stairs import mdp
from mjlab.tasks.osl_stairs.terrain import TerrainBoxes

MOTION_ENV_VAR = "MJLAB_OSL_STAIRS_MOTION"
"""Default motion NPZ when ``--env.commands.motion.motion-file`` is not given (play
takes no ``--env`` overrides, so it reads the clip from here)."""


def sync_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  """Point the scene's terrain at the motion command's clip, in place."""
  motion = cfg.commands.get("motion")
  if not isinstance(motion, mdp.StairsMotionCommandCfg):
    raise TypeError(
      "A stairs env needs a StairsMotionCommandCfg 'motion' command; got "
      f"{type(motion).__name__}."
    )
  cfg.scene.spec_fn = TerrainBoxes(motion.motion_file, motion.terrain_file)


@dataclass(kw_only=True)
class OslStairsEnvCfg(ManagerBasedRlEnvCfg):
  """An env config whose scene terrain follows its motion command."""

  def __post_init__(self) -> None:
    sync_terrain(self)


def osl_stairs_env_cfg(
  model: Model = "robot",
  socket: Socket = "rigid",
  reward: Reward = "mimic",
  mode: Mode = "train",
  dr_cfg: OslDrCfg | None = None,
) -> OslStairsEnvCfg:
  """Create a stairs imitation environment.

  Args:
    model: ``"robot"`` (27 position servos + 2 OSL motors) or ``"msk"`` (339
      muscles + 2 OSL motors, fingerless).
    socket: ``"rigid"`` or ``"compliant"``. Clips are rigid; the compliant socket's
      joints are held at zero in the reference.
    reward: ``"mimic"`` (MimicReward) or ``"clf"`` (CLFReward).
    mode: ``"train"`` starts at random frames, with musclemimic's training
      terminations, pushes and physical draws; ``"validation"`` starts at frame 0 on
      the nominal plant and ends only on the 0.5 m site check or the clip's end;
      ``"play"`` starts at frame 0 with the physical draws, no pushes.
    dr_cfg: Randomization ranges (default ``OslDrCfg()``, the gait-library tasks').
  """
  base = osl_mimic_env_cfg(model, socket, reward, mode)
  base.commands = {
    "motion": mdp.StairsMotionCommandCfg(
      motion_file=os.environ.get(MOTION_ENV_VAR, ""),
      random_start=mode == "train",
    )
  }
  # Every env's boxes sit at their TERRA world coordinates; so do its robot and
  # reference, with the env origin at the world origin.
  base.scene.env_spacing = 0.0
  apply_osl_dr(base, model, socket, mode, dr_cfg)
  return OslStairsEnvCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
