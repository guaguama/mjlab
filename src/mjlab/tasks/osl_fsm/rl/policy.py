"""The human actor as one policy over the env's whole action, for stage 1.5.

``play.py`` and the in-runner evaluations step the env with a single policy's output
over every action row. In stage 1.5 the human actor owns its rows and the FSM action
term drives the OSL from the prosthesis's own sensors, ignoring the policy's two OSL
rows (:mod:`mjlab.tasks.osl_fsm.mdp.actions`). The policy writes zeros there, so its
output never looks like an OSL command; the rows stay because the env's action keeps
stage 1's layout, which the warm start, the reward's owner penalties and the
exporters check.

The human acts with its mean (stage 1's validation protocol, the deployment metric)
or with the noise-averaged action of :mod:`mjlab.tasks.osl_split.rl.expected`, as in
stage 2. The actor is the training module itself, not a copy, so a policy built
mid-training plays the current weights; its observation normalizer is put in
``"frozen"`` mode at construction and again on every call, so inference never moves
the statistics training normalizes with.
"""

from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDict
from torch import nn

from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_split.rl.expected import expected_clipped_gaussian
from mjlab.tasks.osl_split.rl.layout import ActionLayout
from mjlab.tasks.osl_split.rl.policy import HumanMode, _bounds, distribution_std


class HumanOnlyInferencePolicy(nn.Module):
  """``obs -> (B, layout.total_dim)`` env action: the human's action in its rows,
  zeros in the OSL rows.

  Args:
    human_actor: The human actor (shared with the algorithm, not copied).
    layout: Which action rows are the human's.
    human_mode: ``"mean"``, or ``"expected"`` for ``E[clip(mu + sigma eps, lo,
      hi)]`` per human row.
    human_bounds: ``(lo, hi)``, each ``(layout.dim("human"),)``; required for
      ``"expected"`` (see :func:`~mjlab.tasks.osl_split.rl.expected.action_bounds`).
  """

  human_index: torch.Tensor
  human_lo: torch.Tensor | None
  human_hi: torch.Tensor | None

  def __init__(
    self,
    human_actor: MuscleMimicMLPModel,
    layout: ActionLayout,
    human_mode: HumanMode = "mean",
    human_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> None:
    super().__init__()
    if human_mode not in ("mean", "expected"):
      raise ValueError(f"Unknown human_mode {human_mode!r}.")
    self.human_actor = human_actor
    self.layout = layout
    self.human_mode: HumanMode = human_mode
    self.osl_mode: Literal["mean"] = "mean"
    """What stage 2's runner reads to seed a sampled OSL family. Stage 1.5 has no
    OSL actor to sample: its OSL rows are always the FSM's to fill."""
    device = next(human_actor.parameters()).device
    self.register_buffer("human_index", layout.index("human", device), persistent=False)
    lo = hi = None
    if human_mode == "expected":
      lo, hi = _bounds(human_bounds, layout.dim("human"), "human_bounds", device)
    self.register_buffer("human_lo", lo, persistent=False)
    self.register_buffer("human_hi", hi, persistent=False)
    self.human_actor.set_norm_mode("frozen")

  def forward(self, obs: TensorDict) -> torch.Tensor:
    self.human_actor.set_norm_mode("frozen")
    human = self.human_actor(obs)  # The distribution mean.
    if self.human_mode == "expected":
      assert self.human_lo is not None and self.human_hi is not None
      sigma = distribution_std(self.human_actor)
      human = expected_clipped_gaussian(human, sigma, self.human_lo, self.human_hi)
    actions = human.new_zeros(human.shape[0], self.layout.total_dim)
    actions.index_copy_(1, self.human_index, human)
    return actions
