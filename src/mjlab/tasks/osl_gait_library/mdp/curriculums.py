"""A twist-command schedule for the gait-library command (not used by default)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from .commands import GaitLibraryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class TwistLimitStage(TypedDict):
  step: int
  """First env step (``env.common_step_counter``) of the stage."""
  max_twist: tuple[float, float, float] | None
  """(forward vx, backward |vx|, |wz|) cap, or None for the whole library."""


class twist_limit_curriculum:
  """Raise the library command's twist cap in stages of env steps.

  For example, slow and gently turning walks for the first 500 iterations, then the
  whole library::

    CurriculumTermCfg(
      func=mdp.twist_limit_curriculum,
      params={
        "command_name": "motion",
        "stages": [
          {"step": 0, "max_twist": (0.8, 0.6, 0.5)},
          {"step": 500 * 80, "max_twist": None},
        ],
      },
    )

  ``step`` counts env steps, i.e. iterations times ``num_steps_per_env`` (80 here).
  Checkpoints store the counter and a resumed run restores it, so the schedule
  continues from the stage its step count has reached. Curriculum terms run at
  episode resets, and a new cap reaches an env at its next draw (episode start or
  timer resample).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._stages: list[TwistLimitStage] = list(cfg.params["stages"])
    steps = [stage["step"] for stage in self._stages]
    if steps != sorted(steps):
      raise ValueError(f"Stage steps must be nondecreasing, got {steps}.")
    self._cmd = cast(
      GaitLibraryCommand, env.command_manager.get_term(cfg.params["command_name"])
    )
    self._applied: int | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[TwistLimitStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, stages
    current = None
    for i, stage in enumerate(self._stages):
      if env.common_step_counter >= stage["step"]:
        current = i
    if current is not None and current != self._applied:
      self._cmd.set_twist_limit(self._stages[current]["max_twist"])
      self._applied = current
    limit = self._cmd.twist_limit
    names = ("max_vx_fwd", "max_vx_bck", "max_wz")
    if limit is None:
      return {name: torch.tensor(float("inf")) for name in names}
    return {name: torch.tensor(value) for name, value in zip(names, limit, strict=True)}
