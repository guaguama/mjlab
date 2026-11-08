"""Stage 2's schedules: the freeze curriculum, the per-module learning rates, the
OSL blend and the reward channels' group weights.

Every schedule is a function of the *stage-local* iteration ``k`` (0 at stage 2's
first update), never of the global iteration number stage 2 logs under and never of
an optimizer's own step count. Stage 1's ``MuscleMimicPPO`` decays its single rate
over ``lr_schedule_updates`` counted by ``num_updates_done``, which its checkpoint
carries, so resuming from the final stage-1 checkpoint would train at a rate of 0.

With ``S`` the stage length, ``N`` the human freeze and ``R`` the human's ramp:

* both critics learn from ``k = 0`` at ``critic_lr * (1 - k / S)``;
* the OSL actor acts but does not learn during the critic-only warm-up
  (``k < critic_warmup``), then follows the critics' decay, ``osl_actor_lr * (1 -
  k / S)``; the warm-up does not restart it;
* the human actor is frozen for ``k < N``, then ramps linearly over ``R``
  iterations to ``human_peak_lr``, times its own linear decay to 0 at ``S``. Muon's
  step length is set by the rate rather than by the gradient's size, so the rate
  is the only brake on how far the warm-started human moves.

Nothing requires ``N >= critic_warmup``: a shorter human freeze lets the human
learn from critics that are still warming up.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Module = Literal["human_actor", "human_critic", "osl_actor", "osl_critic"]
MODULES: tuple[Module, Module, Module, Module] = (
  "human_actor",
  "human_critic",
  "osl_actor",
  "osl_critic",
)
"""The four separately optimized networks, each with its own ``MuonAdam``."""


@dataclass(frozen=True)
class SplitSchedule:
  """Which modules learn at stage-local iteration ``k``, and at what rate."""

  total_iterations: int
  """``S``: the stage-local iteration at which every rate reaches 0."""
  human_freeze: int
  """``N``: the human actor (weights, std and observation statistics) is frozen
  for ``k < N``."""
  critic_warmup: int
  """The OSL actor's weights are frozen for ``k < critic_warmup`` (its observation
  statistics still update)."""
  critic_lr: float
  osl_actor_lr: float
  human_peak_lr: float
  human_ramp: int
  """``R``: iterations from the unfreeze to the human's peak rate."""
  blend: bool
  """Drive the OSL with ``(1 - beta) * teacher + beta * osl_actor``."""
  blend_ramp: int
  """Iterations over which ``beta`` rises from 0 to 1, from ``critic_warmup``."""

  def __post_init__(self) -> None:
    if self.total_iterations < 1:
      raise ValueError(f"total_iterations must be >= 1, got {self.total_iterations}")
    for name in ("human_freeze", "critic_warmup", "human_ramp", "blend_ramp"):
      if getattr(self, name) < 0:
        raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
    for name in ("critic_lr", "osl_actor_lr", "human_peak_lr"):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}")
    if self.blend:
      if self.blend_ramp < 1:
        raise ValueError("The blend needs blend_ramp >= 1.")
      if self.critic_warmup + self.blend_ramp > self.human_freeze:
        # The teacher reads the frozen human trunk's features; once the human
        # learns, they drift and the teacher no longer means anything.
        raise ValueError(
          f"The blend must reach the OSL actor (beta = 1) by the human unfreeze: "
          f"critic warm-up {self.critic_warmup} + blend ramp {self.blend_ramp} > "
          f"human freeze {self.human_freeze}."
        )

  def human_actor_frozen(self, k: int) -> bool:
    return k < self.human_freeze

  def osl_actor_frozen(self, k: int) -> bool:
    return k < self.critic_warmup

  def decay(self, k: int) -> float:
    """The stage's linear decay, from 1 at ``k = 0`` to 0 at ``S``."""
    return max(1.0 - k / self.total_iterations, 0.0)

  def lr(self, module: str, k: int) -> float:
    if module in ("human_critic", "osl_critic"):
      return self.critic_lr * self.decay(k)
    if module == "osl_actor":
      return 0.0 if self.osl_actor_frozen(k) else self.osl_actor_lr * self.decay(k)
    if module == "human_actor":
      if self.human_actor_frozen(k):
        return 0.0
      n, s = self.human_freeze, self.total_iterations
      ramp = min((k - n + 1) / max(self.human_ramp, 1), 1.0)
      decay = max(1.0 - (k - n) / max(s - n, 1), 0.0)
      return self.human_peak_lr * ramp * decay
    raise KeyError(f"Unknown module {module!r}; expected one of {MODULES}.")

  def lrs(self, k: int) -> dict[str, float]:
    return {module: self.lr(module, k) for module in MODULES}

  def beta(self, k: int) -> float:
    """The OSL actor's share of the commanded OSL action: 1 without the blend."""
    if not self.blend:
      return 1.0
    return min(max((k - self.critic_warmup) / self.blend_ramp, 0.0), 1.0)

  @classmethod
  def from_train_cfg(
    cls, train_cfg: Mapping[str, Any], total_iterations: int, reward_switch: bool
  ) -> SplitSchedule:
    """From the runner's ``asdict(SplitRunnerCfg)``.

    ``total_iterations`` is the stage length the runner settled on
    (``algorithm.schedule_iterations``, or the run's iteration count when 0). A
    reward switch starts both critics fresh, so their warm-up is the longer
    ``reward_switch_critic_warmup_iterations``.
    """
    algorithm = train_cfg["algorithm"]
    freeze = train_cfg["freeze"]
    blend = train_cfg.get("blend") or {}
    warmup = (
      freeze["reward_switch_critic_warmup_iterations"]
      if reward_switch
      else freeze["critic_warmup_iterations"]
    )
    return cls(
      total_iterations=int(total_iterations),
      human_freeze=int(freeze["human_actor_iterations"]),
      critic_warmup=int(warmup),
      critic_lr=float(algorithm["critic_lr"]),
      osl_actor_lr=float(algorithm["osl_actor_lr"]),
      human_peak_lr=float(algorithm["human_actor_peak_lr"]),
      human_ramp=int(algorithm["human_actor_ramp_iterations"]),
      blend=bool(blend.get("enabled", False)),
      blend_ramp=int(blend.get("ramp_iterations", 0)),
    )


Weights = dict[str, dict[str, float]]
"""``{channel: {group: weight}}``, the form ``split_mimic_reward`` takes."""


@dataclass(frozen=True)
class ChannelSchedule:
  """The reward channels' coordinate-group weights over the stage.

  Each channel holds its start weights until ``anneal_start`` and then moves
  linearly to its end weights over ``anneal_iterations``. The defaults start at the
  stage-1 equivalents, so both channels begin as the stage-1 reward, and hold them
  through the human freeze: the OSL actor first learns the reward the human was
  trained on.
  """

  start: Weights
  end: Weights
  anneal_start: int
  anneal_iterations: int

  def __post_init__(self) -> None:
    # Private copies: the runner's config dicts stay the caller's.
    object.__setattr__(self, "start", _copy_weights(self.start))
    object.__setattr__(self, "end", _copy_weights(self.end))
    if self.start.keys() != self.end.keys():
      raise ValueError(
        f"Start channels {sorted(self.start)} differ from end channels "
        f"{sorted(self.end)}."
      )
    for channel, groups in self.start.items():
      if groups.keys() != self.end[channel].keys():
        raise ValueError(
          f"Channel {channel!r}: start groups {sorted(groups)} differ from end "
          f"groups {sorted(self.end[channel])}."
        )
      for weights in (groups, self.end[channel]):
        if any(not math.isfinite(w) or w < 0.0 for w in weights.values()):
          raise ValueError(f"Channel {channel!r}: weights must be >= 0: {weights}")
        # Interpolated weights are convex combinations, so a positive sum at both
        # ends keeps every channel well defined in between.
        if sum(weights.values()) <= 0.0:
          raise ValueError(f"Channel {channel!r} has no positive weight: {weights}")
    if self.anneal_start < 0 or self.anneal_iterations < 0:
      raise ValueError(
        f"anneal_start ({self.anneal_start}) and anneal_iterations "
        f"({self.anneal_iterations}) must be >= 0."
      )

  def fraction(self, k: int) -> float:
    """How far the anneal has gone at ``k``: 0 before it starts, 1 once done."""
    if self.anneal_iterations == 0:
      return 1.0 if k >= self.anneal_start else 0.0
    return min(max((k - self.anneal_start) / self.anneal_iterations, 0.0), 1.0)

  def weights(self, k: int) -> Weights:
    f = self.fraction(k)
    # (1 - f) * a + f * b is exactly a at f = 0 and exactly b at f = 1.
    return {
      channel: {
        group: (1.0 - f) * w + f * self.end[channel][group]
        for group, w in groups.items()
      }
      for channel, groups in self.start.items()
    }

  @classmethod
  def from_train_cfg(cls, train_cfg: Mapping[str, Any]) -> ChannelSchedule:
    """From ``asdict(SplitRunnerCfg)``; an ``anneal_start_iteration`` of -1 means
    the human unfreeze (``freeze.human_actor_iterations``)."""
    channels = train_cfg["channels"]
    start = int(channels["anneal_start_iteration"])
    if start < 0:
      start = int(train_cfg["freeze"]["human_actor_iterations"])
    return cls(
      start={"human": channels["human_start"], "osl": channels["osl_start"]},
      end={"human": channels["human_end"], "osl": channels["osl_end"]},
      anneal_start=start,
      anneal_iterations=int(channels["anneal_iterations"]),
    )


def _copy_weights(weights: Mapping[str, Mapping[str, float]]) -> Weights:
  return {
    str(channel): {str(group): float(w) for group, w in groups.items()}
    for channel, groups in weights.items()
  }
