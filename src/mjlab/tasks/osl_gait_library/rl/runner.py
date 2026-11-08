"""The validating runner, with validation over every library clip.

Validation plays each clip (the idle one included) in its own env from phase 0 at
its own twist, deterministically, for one 10 s episode, ending early only on the
0.5 m site check. The metrics are the training diagnostics (imitation reward, CLF
when on, and twist) averaged over the run, plus the early-termination rate and
episode length and return. Key diagnostics are also reported per clip family
(``fwd``, ``fwd_tight``, ``bck``, ``turn``, ``idle``), e.g.
``Validation/twist_err_wz/turn``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommand, GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic.rl.runner import (
  EvalPurpose,
  OslMimicOnPolicyRunner,
  eval_mode,
)


class GaitLibraryOnPolicyRunner(OslMimicOnPolicyRunner):
  validation_group_keys = (
    "err_site_abs",
    "reward_clipped",
    "clf_V_norm",
    "twist_reward",
    "twist_err_vx",
    "twist_err_wz",
  )

  def _eval_env_cfg(self, purpose: EvalPurpose) -> ManagerBasedRlEnvCfg:
    """The task's validation mode; for videos its play mode (twists drawn from the
    library, the reference ghost) on validation's nominal plant; both with the
    training env's library and its command-line settings (simulation, rewards; see
    ``_inherit_training_settings``). Stress validation plays the training config
    itself (random switches and headings, physical draws, pushes, random delays)."""
    # Local import: config imports this runner to register the tasks.
    from mjlab.tasks.osl_gait_library.config.dr import nominal_plant
    from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg

    if purpose == "stress":
      # The training config itself: every command-line override included.
      return deepcopy(self.env.unwrapped.cfg)
    v = self.validation_cfg
    variant = v.get("variant", "walk_fwd")
    cfg = osl_gait_library_env_cfg(
      v["model"], v["socket"], v["reward"], mode=eval_mode(purpose), variant=variant
    )
    if purpose == "video" and variant == "unicycle":
      nominal_plant(cfg)
    train_motion = self.env.unwrapped.cfg.commands["motion"]
    motion = cfg.commands["motion"]
    assert isinstance(train_motion, GaitLibraryCommandCfg)
    assert isinstance(motion, GaitLibraryCommandCfg)
    motion.library_dir = train_motion.library_dir
    self._inherit_training_settings(cfg)
    return cfg

  def _validation_num_envs(self) -> int:
    """One env per clip unless set."""
    train_motion = self.env.unwrapped.cfg.commands["motion"]
    assert isinstance(train_motion, GaitLibraryCommandCfg)
    clips = len(list(Path(train_motion.library_dir).glob("*.npz")))
    return int(self.validation_cfg.get("num_envs", 0)) or clips

  def _default_validation_steps(self, env: ManagerBasedRlEnv) -> int:
    """One episode."""
    return int(env.max_episode_length)

  def _validation_groups(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Validation env ``i`` plays clip ``i % num_clips``: group it by that clip's
    family. A single-family library reports no groups."""
    cmd = cast(GaitLibraryCommand, env.command_manager.get_term("motion"))
    names = sorted(set(cmd.clip_groups))
    if len(names) < 2:
      return {}
    of_clip = torch.tensor(
      [names.index(g) for g in cmd.clip_groups], dtype=torch.long, device=env.device
    )
    group = of_clip[torch.arange(env.num_envs, device=env.device) % cmd.num_clips]
    return {name: group == i for i, name in enumerate(names)}
