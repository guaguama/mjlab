"""Both actors as one deterministic policy over the env's whole action.

``play.py`` and the in-runner validation call a single policy on the env's
observation and step the env with its output. This module is that policy for stage
2: the human actor's action and the OSL actor's mean, scattered into action-manager
order through the :class:`~.layout.ActionLayout` (the OSL rows are 0:2 on the MSK
model and 27:29 on the robot).

The human acts either with its mean (stage 1's validation protocol) or with the
noise-averaged action of :mod:`.expected`. The OSL always acts with its mean: its
std is small and its commands are rarely clipped.

The actors are the training modules themselves, not copies, so a policy built
mid-training plays the current weights. Their observation normalizers are put in
``"frozen"`` mode at construction and again on every call: inference never moves
the statistics training normalizes with, so validation needs no snapshot.
"""

from __future__ import annotations

from typing import Literal

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import (
  GaussianDistribution,
  HeteroscedasticGaussianDistribution,
)
from tensordict import TensorDict
from torch import nn

from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel

from .expected import expected_clipped_gaussian
from .layout import ActionLayout

HumanMode = Literal["mean", "expected"]


def gaussian_distribution(model: MLPModel) -> GaussianDistribution:
  """The model's state-independent Gaussian (``log_std_param`` or ``std_param``)."""
  dist = model.distribution
  if not isinstance(dist, GaussianDistribution) or isinstance(
    dist, HeteroscedasticGaussianDistribution
  ):
    raise TypeError(
      f"Expected a state-independent GaussianDistribution, got {type(dist).__name__}."
    )
  return dist


def std_parameter(model: MLPModel) -> nn.Parameter:
  """The learnable std parameter, in the distribution's own parameterization."""
  dist = gaussian_distribution(model)
  return dist.log_std_param if dist.std_type == "log" else dist.std_param


def distribution_std(model: MLPModel) -> torch.Tensor:
  """The std the distribution samples with: the parameter clamped to its
  ``std_range`` (``exp`` of the clamped ``log_std`` for ``std_type="log"``)."""
  dist = gaussian_distribution(model)
  if dist.std_type == "log":
    low, high = dist.log_std_range
    return torch.exp(dist.log_std_param.clamp(low, high))
  low, high = dist.std_range
  return dist.std_param.clamp(low, high)


class SplitInferencePolicy(nn.Module):
  """``obs -> (B, layout.total_dim)`` env action, deterministic.

  Args:
    human_actor, osl_actor: The two actors (shared with the algorithm, not copied).
    layout: Which action rows each agent owns.
    human_mode: ``"mean"``, or ``"expected"`` for ``E[clip(mu + sigma eps, lo,
      hi)]`` per human row.
    human_bounds: ``(lo, hi)``, each ``(layout.dim("human"),)``; required for
      ``"expected"`` (see :func:`~.expected.action_bounds`).
  """

  human_index: torch.Tensor
  osl_index: torch.Tensor
  human_lo: torch.Tensor | None
  human_hi: torch.Tensor | None

  def __init__(
    self,
    human_actor: MuscleMimicMLPModel,
    osl_actor: MuscleMimicMLPModel,
    layout: ActionLayout,
    human_mode: HumanMode = "mean",
    human_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> None:
    super().__init__()
    if human_mode not in ("mean", "expected"):
      raise ValueError(f"Unknown human_mode {human_mode!r}.")
    self.human_actor = human_actor
    self.osl_actor = osl_actor
    self.layout = layout
    self.human_mode: HumanMode = human_mode
    device = next(human_actor.parameters()).device
    self.register_buffer("human_index", layout.index("human", device), persistent=False)
    self.register_buffer("osl_index", layout.index("osl", device), persistent=False)
    lo = hi = None
    if human_mode == "expected":
      if human_bounds is None:
        raise ValueError("human_mode='expected' needs human_bounds.")
      lo, hi = (b.to(device=device, dtype=torch.float32) for b in human_bounds)
      dim = layout.dim("human")
      if lo.shape != (dim,) or hi.shape != (dim,):
        raise ValueError(
          f"human_bounds must be ({dim},) each, got {tuple(lo.shape)} and "
          f"{tuple(hi.shape)}."
        )
      if bool((lo > hi).any()):
        raise ValueError("human_bounds need lo <= hi.")
    self.register_buffer("human_lo", lo, persistent=False)
    self.register_buffer("human_hi", hi, persistent=False)
    self._freeze_normalizers()

  def _freeze_normalizers(self) -> None:
    self.human_actor.set_norm_mode("frozen")
    self.osl_actor.set_norm_mode("frozen")

  def forward(self, obs: TensorDict) -> torch.Tensor:
    self._freeze_normalizers()
    human = self.human_actor(obs)  # The distribution mean.
    if self.human_mode == "expected":
      assert self.human_lo is not None and self.human_hi is not None
      sigma = distribution_std(self.human_actor)
      human = expected_clipped_gaussian(human, sigma, self.human_lo, self.human_hi)
    osl = self.osl_actor(obs)
    actions = human.new_empty(human.shape[0], self.layout.total_dim)
    actions.index_copy_(1, self.human_index, human)
    actions.index_copy_(1, self.osl_index, osl.to(human.dtype))
    return actions
