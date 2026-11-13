"""The classical OSL walking controller, batched over envs.

The finite-state impedance controller of the opensourceleg SDK
(``examples/fsm_walking_python_controller.py``) and of MyoSuite's port of it
(``MyoOSLController``, ``myosuite/envs/myo/assets/leg/myoosl_control.py``): four
states (early and late stance, early and late swing), each a stiffness, a damping and
an equilibrium angle per joint, so a torque

    torque = stiffness * (equilibrium - angle) - damping * velocity

osl_deploy's ``osl_fsm`` is the source of truth, and this module is its
``StateMachine`` over a batch: the same states, gains, thresholds and rule sets, at
most one transition per update with every condition read from the state before it,
and the torque computed from the new state on the same reading. A policy trained
beside this controller is deployed beside osl_deploy's, so the two must agree
transition for transition.

Two choices keep that agreement exact:

* the transition conditions compare in float64. osl_deploy reads the float32 sensor
  observation into Python floats and compares those with float64 thresholds; a
  float32 comparison would disagree for readings within a rounding of a threshold
  (an ankle at ``float32(radians(6))`` reads above 6 degrees in float64, say). The
  torques, which no transition depends on, stay float32;
* the body weight is a Python float, as osl_deploy's.

Every episode of osl_deploy starts in early stance, the SDK example's initial state.
Training starts its episodes at random clip frames, mid-swing as often as not, which
MyoSuite's environment handles by picking the start state from the start pose:
:meth:`BatchedFsm.guess` is that pose rule.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

STATES: tuple[str, str, str, str] = ("e_stance", "l_stance", "e_swing", "l_swing")
"""The four states; a state's code is its index here."""
E_STANCE, L_STANCE, E_SWING, L_SWING = range(4)
JOINTS: tuple[str, str] = ("knee", "ankle")
"""The two joints, in the column order of every (N, 2) tensor here."""

# Per state, knee then ankle: stiffness (N m/rad), damping (N m s/rad) and
# equilibrium angle (deg; knee flexion and ankle dorsiflexion positive).
# fsm_walking_python_controller.py:26-61; MyoOSLController's defaults are equal.
IMPEDANCE: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
  "e_stance": ((99.372, 3.180, 5.0), (19.874, 0.0, -2.0)),
  "l_stance": ((99.372, 1.272, 8.0), (79.498, 0.063, -20.0)),
  "e_swing": ((39.749, 0.063, 60.0), (7.949, 0.0, 25.0)),
  "l_swing": ((15.899, 3.816, 5.0), (7.949, 0.0, 15.0)),
}
# Transition thresholds of both sources: loads in body weights, angles in rad.
LOAD_LATE_STANCE, LOAD_EARLY_SWING, LOAD_EARLY_STANCE = 0.25, 0.15, 0.4
ANKLE_LATE_STANCE = math.radians(6.0)
KNEE_LATE_SWING, KNEE_EARLY_STANCE = math.radians(50.0), math.radians(30.0)

RuleName = Literal["sdk", "myosuite"]


@dataclass(frozen=True)
class Rules:
  """A rule set: how a state's conditions combine, and its velocity threshold."""

  name: str
  every: bool
  """A state leaves when every condition holds; otherwise when any one does."""
  knee_velocity: float
  """rad/s: early swing ends once the knee flexes slower than this."""
  swing_load_exit: bool = False
  """Early swing returns to early stance once the leg is loaded again."""


RULES: dict[str, Rules] = {
  "sdk": Rules("sdk", True, 3.0),
  "myosuite": Rules("myosuite", False, math.radians(3.0)),
}
"""The SDK example's transitions, where early stance and early swing leave when both
of their conditions hold and early swing compares the knee velocity with 3 rad/s; and
MyoSuite's, where any one condition suffices and the threshold is 3 deg/s. Late
swing leaves on either of its conditions under both."""


def rule_set(name: str, swing_load_exit: bool | None = None) -> Rules:
  """The named rules, as osl_deploy resolves them.

  The load-based swing exit is the sdk rules' option, on unless ``swing_load_exit``
  is False: without it, a step whose knee never flexes past the early-swing
  threshold keeps the leg in early swing through the next stance.

  Raises:
    ValueError: For unknown rules, or the swing exit asked of MyoSuite's.
  """
  if name not in RULES:
    raise ValueError(f"Unknown rules {name!r}; choose from {sorted(RULES)}")
  if name != "sdk":
    if swing_load_exit:
      raise ValueError("The load-based swing exit belongs to the sdk rules")
    return RULES[name]
  return dataclasses.replace(RULES[name], swing_load_exit=swing_load_exit is not False)


def impedance_settings() -> dict[str, dict[str, dict[str, float]]]:
  """:data:`IMPEDANCE` as plain data, per state and joint, units in the keys."""
  return {
    state: {
      joint: {
        "stiffness_nm_per_rad": stiffness,
        "damping_nms_per_rad": damping,
        "equilibrium_deg": angle,
      }
      for joint, (stiffness, damping, angle) in zip(JOINTS, gains, strict=True)
    }
    for state, gains in IMPEDANCE.items()
  }


def threshold_settings() -> dict[str, float]:
  """The transition thresholds as plain data (the knee velocity is the rules')."""
  return {
    "load_late_stance_bw": LOAD_LATE_STANCE,
    "load_early_swing_bw": LOAD_EARLY_SWING,
    "load_early_stance_bw": LOAD_EARLY_STANCE,
    "ankle_late_stance_rad": ANKLE_LATE_STANCE,
    "knee_late_swing_rad": KNEE_LATE_SWING,
    "knee_early_stance_rad": KNEE_EARLY_STANCE,
  }


class BatchedFsm:
  """The four-state walking controller over N envs: readings in SI units in, torques
  out.

  ``state`` is an (N,) long tensor of state codes (indices into :data:`STATES`),
  starting in early stance. Readings are (N,) tensors: the axial load in N
  (compression positive), the knee angle and velocity and the ankle angle in rad and
  rad/s. :meth:`update` changes ``state`` in place and returns it, so a caller that
  needs the state from before the update clones it first.
  """

  def __init__(
    self,
    rules: Rules,
    body_weight: float,
    num_envs: int,
    device: str | torch.device = "cpu",
  ) -> None:
    if not isinstance(rules, Rules):
      raise TypeError(f"rules must be a Rules (see rule_set), got {rules!r}")
    if not math.isfinite(body_weight) or body_weight <= 0:
      raise ValueError("Body weight must be positive")
    self.rules = rules
    self.body_weight = float(body_weight)
    self.state = torch.full((num_envs,), E_STANCE, dtype=torch.long, device=device)

    def table(column: int, convert: Any = float) -> torch.Tensor:
      rows = [[convert(gains[column]) for gains in IMPEDANCE[s]] for s in STATES]
      return torch.tensor(rows, dtype=torch.float32, device=device)

    # (4, 2) per state and joint; the angles converted in float64, then rounded.
    self.stiffness = table(0)
    self.damping = table(1)
    self.equilibrium = table(2, math.radians)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Put the envs in ``env_ids`` (every env when None) back in early stance."""
    self.state[slice(None) if env_ids is None else env_ids] = E_STANCE

  def guess(self, load: torch.Tensor, knee_velocity: torch.Tensor) -> torch.Tensor:
    """(N,) the state a start pose implies, from that pose's own reading.

    A leg carrying more than early swing's unloading threshold is in stance, and a
    stance starts in early stance, as every cycle does. An unloaded leg is in early
    swing while its knee still flexes, and in late swing once it extends (or holds
    still). The regular update then runs on the same reading, so a guess one state
    short of the truth catches up at once wherever the rules allow.
    """
    weights = load.double() / self.body_weight
    swing = torch.where(knee_velocity > 0.0, E_SWING, L_SWING)
    return torch.where(weights > LOAD_EARLY_SWING, E_STANCE, swing)

  def start_from_pose(
    self, mask: torch.Tensor, load: torch.Tensor, knee_velocity: torch.Tensor
  ) -> None:
    """Set the state of the envs in the (N,) bool ``mask`` from :meth:`guess`."""
    self.state.copy_(torch.where(mask, self.guess(load, knee_velocity), self.state))

  def update(
    self,
    load: torch.Tensor,
    knee: torch.Tensor,
    knee_velocity: torch.Tensor,
    ankle: torch.Tensor,
  ) -> torch.Tensor:
    """Take at most one transition per env on this reading; return ``state``.

    Every condition reads the state from before the update, so no env moves two
    states in one call, as in both sources.
    """
    rules, state = self.rules, self.state
    weights = load.double() / self.body_weight
    knee64 = knee.double()
    met = torch.logical_and if rules.every else torch.logical_or
    early_stance = (state == E_STANCE) & met(
      weights > LOAD_LATE_STANCE, ankle.double() > ANKLE_LATE_STANCE
    )
    late_stance = (state == L_STANCE) & (weights < LOAD_EARLY_SWING)
    early_swing = state == E_SWING
    # Loaded again: stance wins over the swing conditions.
    if rules.swing_load_exit:
      load_exit = early_swing & (weights > LOAD_EARLY_STANCE)
    else:
      load_exit = torch.zeros_like(early_swing)
    swing_exit = (
      early_swing
      & ~load_exit
      & met(knee64 > KNEE_LATE_SWING, knee_velocity.double() < rules.knee_velocity)
    )
    late_swing = (state == L_SWING) & (
      (weights > LOAD_EARLY_STANCE) | (knee64 < KNEE_EARLY_STANCE)
    )
    # masked_fill rather than boolean indexing: no host sync on the GPU.
    new = state.masked_fill(early_stance, L_STANCE)
    new.masked_fill_(late_stance, E_SWING)
    new.masked_fill_(load_exit | late_swing, E_STANCE)
    new.masked_fill_(swing_exit, L_SWING)
    state.copy_(new)
    return state

  def torque(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """(N, 2) knee and ankle torque in N m of the current state's spring-dampers,
    from (N, 2) angles and velocities (knee column 0, ankle column 1). Unclipped:
    the motors' limits are the caller's."""
    s = self.state
    return self.stiffness[s] * (self.equilibrium[s] - q) - self.damping[s] * qd
