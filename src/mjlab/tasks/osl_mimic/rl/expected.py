"""The noise-averaged action of a clipped Gaussian policy.

Every sampled command of these policies is clipped: to [-1, 1] by the action terms,
and a muscle's excitation again to [0, 1] by MuJoCo's activation dynamics. With a
large std the clipping makes the noise itself drive the plant: a muscle commanded a
mean of 0 receives about 0.42 on average at std 2.5, and 0 from the mean action. A
policy trained that way leans on the noise, and its raw mean drives every muscle far
more weakly than training did. ``E[clip(mu + sigma eps, lo, hi)]`` is the action
the policy is trained to expect on average, and still deterministic.

``expected_clipped_gaussian`` is the closed form; ``action_bounds`` the range each
output of a stage-1 policy is effectively clipped to, in action-manager order.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import TendonEffortAction

if TYPE_CHECKING:
  from mjlab.managers.action_manager import ActionManager

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_MIN_SIGMA = 1e-12
"""Below this the clipped mean is the clipped value; the floor keeps ``z`` finite."""
_PSI_FLOOR = -40.0
"""``psi(z)`` underflows to 0 well before this in float64; clamping there avoids
``-inf * 0`` for an infinite bound."""


def _psi(z: torch.Tensor) -> torch.Tensor:
  """``psi(z) = z Phi(z) + phi(z)``, so that ``E[(X - c)^+] = sigma psi((mu - c) /
  sigma)`` for ``X ~ N(mu, sigma^2)``."""
  z = z.clamp(min=_PSI_FLOOR)
  return z * torch.special.ndtr(z) + torch.exp(-0.5 * z.square()) * _INV_SQRT_2PI


def expected_clipped_gaussian(
  mu: torch.Tensor,
  sigma: torch.Tensor | float,
  lo: torch.Tensor | float,
  hi: torch.Tensor | float,
) -> torch.Tensor:
  """``E[clip(X, lo, hi)]`` for ``X ~ N(mu, sigma^2)``, elementwise (broadcast).

  ``lo + sigma (psi((mu - lo) / sigma) - psi((mu - hi) / sigma))``, evaluated as
  the equal ``mu + sigma (psi((lo - mu) / sigma) - psi((mu - hi) / sigma))`` (by
  ``psi(z) - psi(-z) = z``): both arguments are very negative when ``mu`` sits
  well inside the bounds, so the result is ``mu`` exactly there, and an infinite
  bound contributes nothing instead of ``inf - inf``. Computed in float64 and
  returned in ``mu``'s dtype. Requires ``lo <= hi``.
  """
  m = mu.to(torch.float64)
  s = torch.as_tensor(sigma, dtype=torch.float64, device=mu.device)
  s = s.clamp(min=_MIN_SIGMA)
  a = torch.as_tensor(lo, dtype=torch.float64, device=mu.device)
  b = torch.as_tensor(hi, dtype=torch.float64, device=mu.device)
  out = m + s * (_psi((a - m) / s) - _psi((m - b) / s))
  return out.to(mu.dtype)


def term_bounds(term: object) -> tuple[float, float]:
  """The range an action term's outputs are effectively clipped to, in policy
  units: (0, 1) for muscles (``TendonEffortAction``: the term clips to [-1, 1],
  MuJoCo's activation dynamics to [0, 1]) and (-1, 1) for everything else (the OSL
  motors scale by 2.88 and clip to +-2.88 ctrl; the servos map [-1, 1] onto the
  joint range and clip to it)."""
  return (0.0, 1.0) if isinstance(term, TendonEffortAction) else (-1.0, 1.0)


def action_bounds(
  action_manager: ActionManager, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
  """(lo, hi), each (action_dim,), for every action term in action-manager order."""
  lows: list[float] = []
  highs: list[float] = []
  for name in action_manager.active_terms:
    term = action_manager.get_term(name)
    lo, hi = term_bounds(term)
    lows.extend([lo] * int(term.action_dim))
    highs.extend([hi] * int(term.action_dim))
  return (
    torch.tensor(lows, dtype=torch.float32, device=device),
    torch.tensor(highs, dtype=torch.float32, device=device),
  )


def action_term_slices(action_manager: ActionManager) -> dict[str, slice]:
  """Each action term's columns of the policy action, in action-manager order."""
  slices: dict[str, slice] = {}
  start = 0
  for name in action_manager.active_terms:
    dim = int(action_manager.get_term(name).action_dim)
    slices[name] = slice(start, start + dim)
    start += dim
  return slices
