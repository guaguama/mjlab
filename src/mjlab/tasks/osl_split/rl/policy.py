"""Both actors as one policy over the env's whole action.

``play.py`` and the in-runner evaluations call a single policy on the env's
observation and step the env with its output. This module is that policy for stage
2: the human actor's action and an OSL action, scattered into action-manager order
through the :class:`~.layout.ActionLayout` (the OSL rows are 0:2 on the MSK model and
27:29 on the robot).

The human acts with its mean (stage 1's validation protocol, the deployment
metric) or with the noise-averaged action of :mod:`.expected`. The OSL rows come
from one of three sources (``osl_mode``):

* ``"mean"``: the OSL actor's mean, the deployable controller;
* ``"sampled"``: a sample of the OSL actor's Gaussian, to see whether the actor
  relies on its own action noise (the caller seeds it);
* ``"teacher"``: the unified stage-1 policy's OSL head on the human trunk's
  penultimate features (:class:`~.ppo.TeacherHead`). While the trunk is the stage-1
  one, human plus teacher is exactly the unified stage-1 policy. With the human's
  ``"expected"`` mode the teacher rows are noise-averaged too, at the unified std
  of those rows (``TeacherHead.log_std``): stage 1's noise-averaged policy.

The actors are the training modules themselves, not copies, so a policy built
mid-training plays the current weights. Their observation normalizers are put in
``"frozen"`` mode at construction and again on every call: inference never moves
the statistics training normalizes with, so validation needs no snapshot.
"""

from __future__ import annotations

from typing import Literal, cast

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
OslMode = Literal["mean", "sampled", "teacher"]


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
  """``obs -> (B, layout.total_dim)`` env action.

  Args:
    human_actor, osl_actor: The two actors (shared with the algorithm, not copied).
    layout: Which action rows each agent owns.
    human_mode: ``"mean"``, or ``"expected"`` for ``E[clip(mu + sigma eps, lo,
      hi)]`` per human row.
    human_bounds: ``(lo, hi)``, each ``(layout.dim("human"),)``; required for
      ``"expected"`` (see :func:`~.expected.action_bounds`).
    osl_mode: ``"mean"``, ``"sampled"`` or ``"teacher"`` (see the module docstring).
    teacher_head: The unified policy's OSL output rows (a linear map of the human
      trunk's penultimate features, with the unified ``log_std`` of those rows);
      required for ``"teacher"``.
    osl_bounds: ``(lo, hi)``, each ``(layout.dim("osl"),)``; required for the
      noise-averaged teacher (``"expected"`` with ``"teacher"``).
  """

  human_index: torch.Tensor
  osl_index: torch.Tensor
  human_lo: torch.Tensor | None
  human_hi: torch.Tensor | None
  osl_lo: torch.Tensor | None
  osl_hi: torch.Tensor | None

  def __init__(
    self,
    human_actor: MuscleMimicMLPModel,
    osl_actor: MuscleMimicMLPModel,
    layout: ActionLayout,
    human_mode: HumanMode = "mean",
    human_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    osl_mode: OslMode = "mean",
    teacher_head: nn.Module | None = None,
    osl_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> None:
    super().__init__()
    if human_mode not in ("mean", "expected"):
      raise ValueError(f"Unknown human_mode {human_mode!r}.")
    if osl_mode not in ("mean", "sampled", "teacher"):
      raise ValueError(f"Unknown osl_mode {osl_mode!r}.")
    if osl_mode == "teacher" and teacher_head is None:
      raise ValueError("osl_mode='teacher' needs the teacher_head.")
    self.human_actor = human_actor
    self.osl_actor = osl_actor
    self.teacher_head = teacher_head if osl_mode == "teacher" else None
    self.layout = layout
    self.human_mode: HumanMode = human_mode
    self.osl_mode: OslMode = osl_mode
    device = next(human_actor.parameters()).device
    self.register_buffer("human_index", layout.index("human", device), persistent=False)
    self.register_buffer("osl_index", layout.index("osl", device), persistent=False)
    lo = hi = osl_lo = osl_hi = None
    if human_mode == "expected":
      lo, hi = _bounds(human_bounds, layout.dim("human"), "human_bounds", device)
      if osl_mode == "teacher":
        osl_lo, osl_hi = _bounds(osl_bounds, layout.dim("osl"), "osl_bounds", device)
    self.register_buffer("human_lo", lo, persistent=False)
    self.register_buffer("human_hi", hi, persistent=False)
    self.register_buffer("osl_lo", osl_lo, persistent=False)
    self.register_buffer("osl_hi", osl_hi, persistent=False)
    self._freeze_normalizers()

  def _freeze_normalizers(self) -> None:
    self.human_actor.set_norm_mode("frozen")
    self.osl_actor.set_norm_mode("frozen")

  def forward(self, obs: TensorDict) -> torch.Tensor:
    self._freeze_normalizers()
    if self.teacher_head is not None:
      # One trunk pass feeds both heads, as in the unified policy.
      features = self.human_actor.mlp[:-1](self.human_actor.get_latent(obs))
      human = self.human_actor.mlp[-1](features)  # The distribution mean.
      osl = self.teacher_head(features)
      if self.osl_lo is not None and self.osl_hi is not None:
        # Stage 1 averages every row at its own std, clamped to the std range the
        # human actor shares with the unified one.
        low, high = gaussian_distribution(self.human_actor).log_std_range
        log_std = cast(torch.Tensor, self.teacher_head.log_std).clamp(low, high)
        osl = expected_clipped_gaussian(osl, log_std.exp(), self.osl_lo, self.osl_hi)
    else:
      human = self.human_actor(obs)  # The distribution mean.
      osl = self.osl_actor(obs, stochastic_output=self.osl_mode == "sampled")
    if self.human_mode == "expected":
      assert self.human_lo is not None and self.human_hi is not None
      sigma = distribution_std(self.human_actor)
      human = expected_clipped_gaussian(human, sigma, self.human_lo, self.human_hi)
    actions = human.new_empty(human.shape[0], self.layout.total_dim)
    actions.index_copy_(1, self.human_index, human)
    actions.index_copy_(1, self.osl_index, osl.to(human.dtype))
    return actions


def _bounds(
  bounds: tuple[torch.Tensor, torch.Tensor] | None,
  dim: int,
  name: str,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  """``(lo, hi)`` as float32 on ``device``, checked: ``(dim,)`` each, lo <= hi."""
  if bounds is None:
    raise ValueError(f"The noise-averaged action needs {name}.")
  lo, hi = (b.to(device=device, dtype=torch.float32) for b in bounds)
  if lo.shape != (dim,) or hi.shape != (dim,):
    raise ValueError(
      f"{name} must be ({dim},) each, got {tuple(lo.shape)} and {tuple(hi.shape)}."
    )
  if bool((lo > hi).any()):
    raise ValueError(f"{name} need lo <= hi.")
  return lo, hi
