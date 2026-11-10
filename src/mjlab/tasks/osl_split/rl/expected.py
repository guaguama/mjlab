"""The noise-averaged action of a clipped Gaussian policy, for validation.

Every sampled command of these policies is clipped: to [-1, 1] by the action term,
and a muscle's excitation again to [0, 1] by MuJoCo's activation dynamics. With a
large std the clipping makes the noise itself drive the plant: at musclemimic's
muscle std of about 2.45, a muscle commanded a mean of 0 receives about 0.42 on
average, and 0 from the raw mean, so a policy trained there leans on its noise and
its raw mean falls. ``E[clip(mu + sigma eps, lo, hi)]`` is the action such a policy
is trained to expect on average, and still deterministic. The stage-1 recipe ends
at a muscle std of 0.07, where it is nearly the raw mean: stage 2 plays it once, at
zero-shot, on the unified stage-1 policy (``ValidationTeacherExpected/``, every row
averaged, as stage 1 does), whose gap to the raw-mean teacher is the warm start's
own noise reliance, and validates the raw mean (``Validation/``), the deployment
metric.

The closed form lives in :mod:`mjlab.tasks.osl_mimic.rl.expected` (stage 1
validates with it too); this module adds the per-agent bounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.osl_mimic.rl.expected import expected_clipped_gaussian, term_bounds

from .layout import ActionLayout, Agent

if TYPE_CHECKING:
  from mjlab.managers.action_manager import ActionManager

__all__ = ["action_bounds", "expected_clipped_gaussian"]


def action_bounds(
  action_manager: ActionManager,
  layout: ActionLayout,
  agent: Agent,
  device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
  """The range each of ``agent``'s action rows is effectively clipped to.

  In ``layout.rows(agent)`` order: (0, 1) for muscles (``TendonEffortAction``:
  the term clips to [-1, 1], MuJoCo's activation dynamics to [0, 1]) and (-1, 1)
  for everything else (the OSL motors scale by 2.88 and clip to +-2.88 ctrl; the
  servos map [-1, 1] onto the joint range and clip to it).
  """
  lows: list[float] = []
  for name, dim in layout.terms:
    if (name == layout.osl_term) != (agent == "osl"):
      continue
    term = action_manager.get_term(name)
    if int(term.action_dim) != dim:
      raise ValueError(
        f"Action term {name!r} has {term.action_dim} dims; the layout has {dim}."
      )
    lows.extend([term_bounds(term)[0]] * dim)
  lo = torch.tensor(lows, dtype=torch.float32, device=device)
  return lo, torch.ones_like(lo)
