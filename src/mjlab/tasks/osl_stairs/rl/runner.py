"""The validating runner, with evaluation envs built from the stairs task.

osl_mimic's runner: validation plays the clip from frame 0 for one clip length,
deterministically, ending early only on the 0.5 m site check. Here the evaluation
envs are the stairs task's own modes (validation on the nominal plant at mid-range
delays; videos in play mode on the nominal plant), on the training env's clip and
terrain. Stress validation plays the training config itself.

The run's ``params/env.yaml`` records only the terrain's path, so the terrain JSON the
env loaded is copied beside it as ``params/terrain.json``: the run keeps its scene
even if the clip's sidecar is refit later.
"""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.osl_mimic.rl.runner import (
  EvalPurpose,
  OslMimicOnPolicyRunner,
  eval_mode,
)
from mjlab.tasks.osl_stairs.mdp import StairsMotionCommand, StairsMotionCommandCfg

TERRAIN_COPY = "terrain.json"
"""The terrain's copy in the run's ``params`` folder."""


class OslStairsOnPolicyRunner(OslMimicOnPolicyRunner):
  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    self.save_terrain()
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save_terrain(self) -> Path | None:
    """Copy the training env's terrain JSON to ``<log_dir>/params/terrain.json``
    (main process only); returns the copy."""
    log_dir = self.logger.log_dir
    if log_dir is None or self.logger.disable_logs:
      return None
    cmd = self.env.unwrapped.command_manager.get_term("motion")
    params = Path(log_dir) / "params"
    params.mkdir(parents=True, exist_ok=True)
    return Path(
      shutil.copyfile(
        cast(StairsMotionCommand, cmd).terrain_path, params / TERRAIN_COPY
      )
    )

  def _eval_env_cfg(self, purpose: EvalPurpose) -> ManagerBasedRlEnvCfg:
    # Local imports: config imports this runner to register the tasks.
    from mjlab.tasks.osl_gait_library.config.dr import nominal_plant
    from mjlab.tasks.osl_stairs.config.env_cfgs import osl_stairs_env_cfg, sync_terrain

    if purpose == "stress":
      return deepcopy(self.env.unwrapped.cfg)
    v = self.validation_cfg
    cfg = osl_stairs_env_cfg(v["model"], v["socket"], v["reward"], eval_mode(purpose))
    if purpose == "video":
      nominal_plant(cfg)
    train_motion = self.env.unwrapped.cfg.commands["motion"]
    motion = cfg.commands["motion"]
    assert isinstance(train_motion, StairsMotionCommandCfg)
    assert isinstance(motion, StairsMotionCommandCfg)
    motion.motion_file = train_motion.motion_file
    motion.terrain_file = train_motion.terrain_file
    sync_terrain(cfg)
    self._inherit_training_settings(cfg)
    return cfg
