"""The validating runner, with validation over every library clip.

Validation plays each clip (the idle one included) in its own env from phase 0 at
its own twist, deterministically, for one 20 s episode (osl_deploy's run length;
training episodes last 10 s), ending early only on the 0.5 m site check. The
metrics are the training diagnostics (imitation reward, CLF when on, and twist)
averaged over the run, plus the early-termination rate and episode length and
return. Key diagnostics are also reported per clip family (``fwd``, ``fwd_tight``,
``bck``, ``turn``, ``idle``) and for the slow forward arcs (``slow``, see
:func:`slow_clips`), e.g. ``Validation/twist_err_wz/turn``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.osl_gait_library.mdp import (
  TIGHT_ARC_WZ,
  GaitLibraryCommand,
  GaitLibraryCommandCfg,
  apply_sampler,
  describe_sampler,
)
from mjlab.tasks.osl_mimic.reward_settings import read_params_yaml
from mjlab.tasks.osl_mimic.rl.runner import (
  EvalPurpose,
  OslMimicOnPolicyRunner,
  eval_mode,
)

SLOW_VX = 0.6
"""The ``slow`` validation group's top speed (m/s): see :func:`slow_clips`."""


def slow_clips(twists: torch.Tensor) -> torch.Tensor:
  """(C,) mask of the slow forward arcs among clips of twist labels ``twists``
  (C, 3): 0 < vx <= ``SLOW_VX`` and |wz| <= ``TIGHT_ARC_WZ``. On unicycle_v001 these
  are the deploy panel's slow band (vx 0.5-0.6, 63 clips), where stage 2 regressed
  most; they are ``fwd`` clips too."""
  eps = 1e-6
  vx, wz = twists[:, 0], twists[:, 2]
  return (vx > eps) & (vx <= SLOW_VX + eps) & (wz.abs() <= TIGHT_ARC_WZ + eps)


def follow_training_command(
  cfg: ManagerBasedRlEnvCfg, train: ManagerBasedRlEnvCfg
) -> None:
  """Give an evaluation config rebuilt from the task the training env's gait library
  and its command sampler (:func:`~mjlab.tasks.osl_gait_library.mdp.apply_sampler`),
  in place: a run on a forward-only library sets its pivot and backward fractions to
  0, and a command rebuilt with the task's default split would refuse that library.
  Validation plays fixed clips and never draws; it takes the sampler all the same,
  so that every command of a run loads under the same check."""
  train_motion = train.commands["motion"]
  motion = cfg.commands["motion"]
  assert isinstance(train_motion, GaitLibraryCommandCfg)
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = train_motion.library_dir
  apply_sampler(motion, train_motion)


def configure_sampler_for_checkpoint(
  env_cfg: ManagerBasedRlEnvCfg, checkpoint: str | Path
) -> None:
  """Give ``env_cfg``'s gait-library command the sampler ``checkpoint``'s run trained
  with, from the run's ``params/env.yaml``, in place, and say what changed. A config
  without a gait-library command, or a run whose ``params/`` is not beside the
  checkpoint, keeps the task's sampler."""
  motion = env_cfg.commands.get("motion")
  if not isinstance(motion, GaitLibraryCommandCfg):
    return
  saved = read_params_yaml(Path(checkpoint).parent / "params" / "env.yaml")
  saved_motion = ((saved or {}).get("commands") or {}).get("motion")
  if not isinstance(saved_motion, dict):
    return
  if apply_sampler(motion, saved_motion):
    print(f"[INFO] Twist sampler as the run's: {describe_sampler(motion)}.")


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
    """The task's validation mode; for videos its play mode (twists drawn as in
    training, the reference ghost) on validation's nominal plant; both with the
    training env's library and command sampler (:func:`follow_training_command`)
    and its command-line settings (simulation, rewards; see
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
    follow_training_command(cfg, self.env.unwrapped.cfg)
    self._inherit_training_settings(cfg)
    return cfg

  @classmethod
  def configure_env_for_checkpoint(
    cls, env_cfg: ManagerBasedRlEnvCfg, checkpoint: str | Path
  ) -> None:
    """The imitation runner's (the OSL touch sites), then the twist sampler the
    checkpoint's run trained with (:func:`configure_sampler_for_checkpoint`):
    ``play`` takes no ``--env`` overrides, and the task's default split does not
    load on a forward-only library."""
    super().configure_env_for_checkpoint(env_cfg, checkpoint)
    configure_sampler_for_checkpoint(env_cfg, checkpoint)

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
    family, and the slow forward arcs (:func:`slow_clips`) also as ``slow``, a
    subset of ``fwd``. A single-family library reports no groups."""
    cmd = cast(GaitLibraryCommand, env.command_manager.get_term("motion"))
    names = sorted(set(cmd.clip_groups))
    if len(names) < 2:
      return {}
    of_clip = torch.tensor(
      [names.index(g) for g in cmd.clip_groups], dtype=torch.long, device=env.device
    )
    clip = torch.arange(env.num_envs, device=env.device) % cmd.num_clips
    group = of_clip[clip]
    groups = {name: group == i for i, name in enumerate(names)}
    slow = slow_clips(torch.as_tensor(cmd.twists, device=env.device))
    if bool(slow.any()):
      groups["slow"] = slow[clip]
    return groups
