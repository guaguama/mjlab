"""The validating runner, with validation over every library clip.

Validation plays each clip (the idle one included) in its own env from phase 0 at
its own twist, deterministically, for one 10 s episode, ending early only on the
0.5 m site check. The metrics are the training diagnostics (CLF and twist) averaged
over the run, plus the early-termination rate and episode length and return.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic.rl.runner import OslMimicOnPolicyRunner


class GaitLibraryOnPolicyRunner(OslMimicOnPolicyRunner):
  def _make_validation_env(self) -> RslRlVecEnvWrapper:
    # Local import: config imports this runner to register the tasks.
    from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg

    v = self.validation_cfg
    cfg = osl_gait_library_env_cfg(v["socket"], mode="validation")
    train_motion = self.env.unwrapped.cfg.commands["motion"]
    motion = cfg.commands["motion"]
    assert isinstance(train_motion, GaitLibraryCommandCfg)
    assert isinstance(motion, GaitLibraryCommandCfg)
    motion.library_dir = train_motion.library_dir
    clips = len(list(Path(motion.library_dir).glob("*.npz")))
    cfg.scene.num_envs = int(v.get("num_envs", 0)) or clips
    return RslRlVecEnvWrapper(
      ManagerBasedRlEnv(cfg=cfg, device=self.env.unwrapped.device)
    )

  def _default_validation_steps(self, env: ManagerBasedRlEnv) -> int:
    """One episode."""
    return int(env.max_episode_length)
