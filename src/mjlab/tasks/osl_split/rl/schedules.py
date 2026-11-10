"""Stage 2's schedules: the freeze curriculum, the per-module learning rates, the
OSL blend and the reward channels' group weights.

Every schedule is a function of the *stage-local* iteration ``k`` (0 at stage 2's
first update), never of the global iteration number stage 2 logs under and never of
an optimizer's own step count. Stage 2 starts every optimizer fresh: none of stage
1's optimizer state or update count is carried over.

With ``S`` the stage length, ``W`` the critic-only warm-up, ``N`` the human freeze,
``R`` the human's ramp and ``r`` the floor ``lr_final_ratio``, the linear decay is
``d(k) = r + (1 - r) max(1 - k / S, 0)``: from 1 at ``k = 0`` down to ``r`` at
``S``, held at ``r`` beyond it, so a run extended past its schedule keeps
learning. Then:

* each critic ramps linearly from ``1 / W`` of its rate at ``k = 0`` to its full
  rate at ``k = W - 1`` (its optimizer state is fresh, the warm critic's weights
  are not), times ``d(k)``: ``human_critic_lr`` and ``osl_critic_lr``; the
  human critic's decay is floored at ``human_peak_lr / human_critic_lr``, so it
  never learns slower than the human actor (1e-4 down to 3e-5 by default, where
  stage 1's critics ended);
* the OSL actor acts but does not learn during the warm-up (``k < W``), then
  follows the same decay, ``osl_actor_lr * d(k)``; the warm-up does not restart
  it;
* the human actor is frozen for ``k < N``, then ramps linearly over ``R``
  iterations to ``human_peak_lr``, times its own decay from the unfreeze to ``S``
  down to ``human_lr_final_ratio`` (1, the default, holds the peak). Muon's step
  length is set by the rate rather than by the gradient's size, so the rate is the
  only brake on how far the warm-started human moves.

Nothing requires ``N >= W``: a shorter human freeze lets the human learn from
critics that are still warming up.
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
  """``S``: the stage-local iteration at which the decaying rates reach their
  floor."""
  human_freeze: int
  """``N``: the human actor (weights, std and observation statistics) is frozen
  for ``k < N``."""
  critic_warmup: int
  """``W``: the OSL actor's weights are frozen for ``k < W`` (its observation
  statistics still update)."""
  human_critic_lr: float
  osl_critic_lr: float
  critic_ramp: int
  """Iterations over which both critics' rates ramp up from ``k = 0`` (the critic
  warm-up; 0 starts them at full rate)."""
  osl_actor_lr: float
  human_peak_lr: float
  human_ramp: int
  """``R``: iterations from the unfreeze to the human's peak rate."""
  lr_final_ratio: float
  """``r``: the critics' and the OSL actor's rates end at this fraction of their
  peak at ``S`` and hold it after."""
  human_lr_final_ratio: float
  """The human's rate decays from its peak to this fraction of it at ``S`` and
  holds it after; 1 holds the peak."""
  blend: bool
  """Drive the OSL with ``(1 - beta) * teacher + beta * osl_actor``."""
  blend_ramp: int
  """Iterations over which ``beta`` rises from 0 to 1, from ``critic_warmup``."""

  def __post_init__(self) -> None:
    if self.total_iterations < 1:
      raise ValueError(f"total_iterations must be >= 1, got {self.total_iterations}")
    for name in (
      "human_freeze",
      "critic_warmup",
      "critic_ramp",
      "human_ramp",
      "blend_ramp",
    ):
      if getattr(self, name) < 0:
        raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
    for name in ("human_critic_lr", "osl_critic_lr", "osl_actor_lr", "human_peak_lr"):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}")
    for name in ("lr_final_ratio", "human_lr_final_ratio"):
      value = getattr(self, name)
      if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    if self.blend:
      if self.blend_ramp < 1:
        raise ValueError("The blend needs blend_ramp >= 1.")
      if self.critic_warmup + self.blend_ramp > self.human_freeze:
        # The teacher reads the frozen human trunk's features; once the human
        # learns, they drift and the teacher no longer means anything.
        raise ValueError(
          f"The blend must reach the OSL actor (beta = 1) by the human unfreeze: "
          f"critic warm-up {self.critic_warmup} + blend ramp {self.blend_ramp} > "
          f"human freeze {self.human_freeze}. Lengthen the freeze "
          "(--agent.freeze.human-actor-iterations), shorten the ramp "
          "(--agent.blend.ramp-iterations) or turn the blend off "
          "(--agent.blend.enabled False)."
        )

  def human_actor_frozen(self, k: int) -> bool:
    return k < self.human_freeze

  def osl_actor_frozen(self, k: int) -> bool:
    return k < self.critic_warmup

  def decay(self, k: int) -> float:
    """The stage's linear decay: 1 at ``k = 0``, ``lr_final_ratio`` from ``S`` on."""
    r = self.lr_final_ratio
    return r + (1.0 - r) * max(1.0 - k / self.total_iterations, 0.0)

  def critic_ramp_factor(self, k: int) -> float:
    """The critics' ramp: ``(k + 1) / critic_ramp`` up to 1."""
    if self.critic_ramp <= 0:
      return 1.0
    return min(max((k + 1) / self.critic_ramp, 0.0), 1.0)

  def lr(self, module: str, k: int) -> float:
    if module == "human_critic":
      # Never slower than the human actor it serves: floored at its peak rate.
      floor = min(1.0, self.human_peak_lr / max(self.human_critic_lr, 1e-12))
      decay = max(self.decay(k), floor)
      return self.human_critic_lr * self.critic_ramp_factor(k) * decay
    if module == "osl_critic":
      return self.osl_critic_lr * self.critic_ramp_factor(k) * self.decay(k)
    if module == "osl_actor":
      return 0.0 if self.osl_actor_frozen(k) else self.osl_actor_lr * self.decay(k)
    if module == "human_actor":
      if self.human_actor_frozen(k):
        return 0.0
      n, s, r = self.human_freeze, self.total_iterations, self.human_lr_final_ratio
      ramp = min((k - n + 1) / max(self.human_ramp, 1), 1.0)
      decay = r + (1.0 - r) * max(1.0 - (k - n) / max(s - n, 1), 0.0)
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
    critic that starts fresh, both under a reward switch or the OSL critic under
    ``algorithm.osl_critic_init="fresh"``, gets the longer warm-up
    ``reward_switch_critic_warmup_iterations``. The critics ramp over the warm-up.
    """
    algorithm = train_cfg["algorithm"]
    freeze = train_cfg["freeze"]
    blend = train_cfg.get("blend") or {}
    fresh = reward_switch or algorithm.get("osl_critic_init", "stage1") == "fresh"
    warmup = int(
      freeze["reward_switch_critic_warmup_iterations"]
      if fresh
      else freeze["critic_warmup_iterations"]
    )
    return cls(
      total_iterations=int(total_iterations),
      human_freeze=int(freeze["human_actor_iterations"]),
      critic_warmup=warmup,
      human_critic_lr=float(algorithm["human_critic_lr"]),
      osl_critic_lr=float(algorithm["osl_critic_lr"]),
      critic_ramp=warmup,
      osl_actor_lr=float(algorithm["osl_actor_lr"]),
      human_peak_lr=float(algorithm["human_actor_peak_lr"]),
      human_ramp=int(algorithm["human_actor_ramp_iterations"]),
      lr_final_ratio=float(algorithm["lr_final_ratio"]),
      human_lr_final_ratio=float(algorithm["human_actor_lr_final_ratio"]),
      blend=bool(blend.get("enabled", False)),
      blend_ramp=int(blend.get("ramp_iterations", 0)),
    )

  @classmethod
  def from_saved(cls, saved: Mapping[str, Any]) -> SplitSchedule:
    """A schedule as checkpoints keep it (``asdict``).

    Checkpoints from before the separate critic rates, the critic ramp and the
    rate floors kept one ``critic_lr`` and decayed every rate to 0: they replay as
    they trained (both critics at ``critic_lr``, no ramp, floors 0).
    """
    values = dict(saved)
    if "critic_lr" in values:
      critic_lr = values.pop("critic_lr")
      values.setdefault("human_critic_lr", critic_lr)
      values.setdefault("osl_critic_lr", critic_lr)
    values.setdefault("critic_ramp", 0)
    values.setdefault("lr_final_ratio", 0.0)
    values.setdefault("human_lr_final_ratio", 0.0)
    return cls(**values)


Weights = dict[str, dict[str, float]]
"""``{channel: {group: weight}}``, the form ``split_mimic_reward`` takes."""


@dataclass(frozen=True)
class ChannelSchedule:
  """The reward channels' coordinate-group weights over the stage.

  Each channel holds its start weights until ``anneal_start`` and then moves
  linearly to its end weights over ``anneal_iterations``. The defaults start at the
  stage-1 equivalents, so both channels begin as the stage-1 reward, and the OSL
  channel anneals towards its knee/ankle joint group from iteration 10 (the default
  critic warm-up's end; a fixed iteration, which a longer warm-up does not move)
  over 150 iterations: the OSL actor gets the coordinates it alone controls while
  the human is still frozen, and the anneal is done before the human unfreezes.
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
