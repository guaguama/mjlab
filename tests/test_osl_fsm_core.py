"""Tests for the batched OSL walking controller (``mjlab.tasks.osl_fsm.fsm``).

osl_deploy's ``osl_fsm`` is the source of truth. Its golden tests
(osl_deploy ``tests/test_osl_fsm.py``) are ported here at the same 800 N body
weight, one env per case or one case per env, and a vendored copy of its scalar
``StateMachine`` checks the batched one reading for reading. osl_deploy is no mjlab
dependency, so the copy lives below rather than an import. CPU only.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from mjlab.tasks.osl_fsm import fsm

WEIGHT = 800.0  # N
rad = math.radians

# ===== osl_deploy's scalar controller, vendored =================================
# osl_deploy/src/osl_deploy/osl_fsm.py:74-177 as of 2026-11-13: the code verbatim,
# re-indented, with its docstrings and some comments left out. The batched
# controller is accessed as ``fsm.<name>`` only, so these names never stand in for
# it.

STATES = ("e_stance", "l_stance", "e_swing", "l_swing")
IMPEDANCE = {
  "e_stance": ((99.372, 3.180, 5.0), (19.874, 0.0, -2.0)),
  "l_stance": ((99.372, 1.272, 8.0), (79.498, 0.063, -20.0)),
  "e_swing": ((39.749, 0.063, 60.0), (7.949, 0.0, 25.0)),
  "l_swing": ((15.899, 3.816, 5.0), (7.949, 0.0, 15.0)),
}
LOAD_LATE_STANCE, LOAD_EARLY_SWING, LOAD_EARLY_STANCE = 0.25, 0.15, 0.4
ANKLE_LATE_STANCE = math.radians(6.0)
KNEE_LATE_SWING, KNEE_EARLY_STANCE = math.radians(50.0), math.radians(30.0)


@dataclasses.dataclass(frozen=True)
class Rules:
  name: str
  every: bool  # A state leaves when every condition holds; otherwise when any does.
  knee_velocity: float  # rad/s: early swing ends once the knee flexes slower.
  swing_load_exit: bool = False  # Early swing returns to early stance under load.


RULES = {
  "sdk": Rules("sdk", True, 3.0),
  "myosuite": Rules("myosuite", False, math.radians(3.0)),
}


def rule_set(name, swing_load_exit=None):
  if name not in RULES:
    raise ValueError(f"Unknown rules {name!r}; choose from {sorted(RULES)}")
  if name != "sdk":
    if swing_load_exit:
      raise ValueError("The load-based swing exit belongs to the sdk rules")
    return RULES[name]
  return dataclasses.replace(RULES[name], swing_load_exit=swing_load_exit is not False)


class StateMachine:
  def __init__(self, rules, body_weight, initial="e_stance"):
    if not math.isfinite(body_weight) or body_weight <= 0:
      raise ValueError("Body weight must be positive")
    if initial not in STATES:
      raise ValueError(f"Unknown state {initial!r}; states are {STATES}")
    self.rules, self.body_weight, self.initial = rules, float(body_weight), initial
    self.state = initial

  def reset(self):
    self.state = self.initial

  def update(self, load, knee, knee_velocity, ankle):
    rules, weights = self.rules, load / self.body_weight
    met = all if rules.every else any
    if self.state == "e_stance":
      if met((weights > LOAD_LATE_STANCE, ankle > ANKLE_LATE_STANCE)):
        self.state = "l_stance"
    elif self.state == "l_stance":
      if weights < LOAD_EARLY_SWING:
        self.state = "e_swing"
    elif self.state == "e_swing":
      # Loaded again: stance wins over the swing conditions.
      if rules.swing_load_exit and weights > LOAD_EARLY_STANCE:
        self.state = "e_stance"
      elif met((knee > KNEE_LATE_SWING, knee_velocity < rules.knee_velocity)):
        self.state = "l_swing"
    elif weights > LOAD_EARLY_STANCE or knee < KNEE_EARLY_STANCE:
      self.state = "e_stance"
    return self.state

  def torque(self, knee, knee_velocity, ankle, ankle_velocity):
    readings = ((knee, knee_velocity), (ankle, ankle_velocity))
    return tuple(
      stiffness * (math.radians(angle) - q) - damping * v
      for (stiffness, damping, angle), (q, v) in zip(
        IMPEDANCE[self.state], readings, strict=True
      )
    )


# ===== Helpers ===================================================================


def machine(
  rules: str = "sdk",
  swing_load_exit: bool | None = None,
  state: str = "e_stance",
  num_envs: int = 1,
) -> fsm.BatchedFsm:
  controller = fsm.BatchedFsm(fsm.rule_set(rules, swing_load_exit), WEIGHT, num_envs)
  controller.state[:] = fsm.STATES.index(state)
  return controller


def _t(*values: float) -> torch.Tensor:
  return torch.tensor(values, dtype=torch.float32)


def after(controller: fsm.BatchedFsm, load, knee, knee_velocity, ankle) -> list[str]:
  """The states after one update on per-env readings (floats broadcast)."""
  n = controller.state.numel()

  def column(value) -> torch.Tensor:
    return _t(*value) if isinstance(value, (list, tuple)) else _t(*([value] * n))

  state = controller.update(
    column(load), column(knee), column(knee_velocity), column(ankle)
  )
  return [fsm.STATES[code] for code in state.tolist()]


# The example's constants, typed here on their own: knee then ankle stiffness,
# damping and equilibrium angle in deg (fsm_walking_python_controller.py:26-61).
EXAMPLE = {
  "e_stance": (99.372, 3.180, 5, 19.874, 0, -2),
  "l_stance": (99.372, 1.272, 8, 79.498, 0.063, -20),
  "e_swing": (39.749, 0.063, 60, 7.949, 0.0, 25),
  "l_swing": (15.899, 3.816, 5, 7.949, 0.0, 15),
}


# ===== osl_deploy's golden tests, ported =========================================


def test_constants_are_osl_deploys() -> None:
  assert fsm.STATES == STATES and fsm.IMPEDANCE == IMPEDANCE
  thresholds = (
    fsm.LOAD_LATE_STANCE,
    fsm.LOAD_EARLY_SWING,
    fsm.LOAD_EARLY_STANCE,
    fsm.ANKLE_LATE_STANCE,
    fsm.KNEE_LATE_SWING,
    fsm.KNEE_EARLY_STANCE,
  )
  assert thresholds == (
    LOAD_LATE_STANCE,
    LOAD_EARLY_SWING,
    LOAD_EARLY_STANCE,
    ANKLE_LATE_STANCE,
    KNEE_LATE_SWING,
    KNEE_EARLY_STANCE,
  )
  for name, rules in RULES.items():
    assert dataclasses.asdict(fsm.RULES[name]) == dataclasses.asdict(rules)
  assert (fsm.E_STANCE, fsm.L_STANCE, fsm.E_SWING, fsm.L_SWING) == (0, 1, 2, 3)


@pytest.mark.parametrize("rules", ["sdk", "myosuite"])
@pytest.mark.parametrize("state", list(EXAMPLE))
def test_each_state_commands_the_examples_spring_damper(rules: str, state: str) -> None:
  knee_k, knee_b, knee_angle, ankle_k, ankle_b, ankle_angle = EXAMPLE[state]
  controller = machine(rules, state=state, num_envs=2)
  torque = controller.torque(_t(0.2, 0.1).expand(2, -1), _t(1.5, -2.0).expand(2, -1))
  assert torque.shape == (2, 2) and torque.dtype == torch.float32
  knee = knee_k * (rad(knee_angle) - 0.2) - knee_b * 1.5
  ankle = ankle_k * (rad(ankle_angle) - 0.1) - ankle_b * -2.0
  for row in torque.tolist():
    assert row == pytest.approx([knee, ankle], rel=1e-6)


def test_early_stance_leaves_on_both_conditions_under_sdk_and_either_under_myosuite():
  # One case per env: (load, ankle) and the state each rule set leaves it in.
  loaded, dorsiflexed = 0.3 * WEIGHT, rad(7)
  cases = (
    (loaded, rad(5), "e_stance", "l_stance"),
    (0.2 * WEIGHT, dorsiflexed, "e_stance", "l_stance"),
    (loaded, dorsiflexed, "l_stance", "l_stance"),
    (0.2 * WEIGHT, rad(5), "e_stance", "e_stance"),
  )
  load, ankle, sdk, myosuite = (list(column) for column in zip(*cases, strict=True))
  assert after(machine("sdk", num_envs=4), load, 0.1, 0.0, ankle) == sdk
  assert after(machine("myosuite", num_envs=4), load, 0.1, 0.0, ankle) == myosuite


def test_early_swing_exit_and_the_velocity_threshold_of_each_source() -> None:
  def swing(rules: str, knee: float, velocity: float) -> str:
    exit_ = False if rules == "sdk" else None
    return after(machine(rules, exit_, "e_swing"), 0.0, knee, velocity, 0.0)[0]

  # A knee that never passes 50 deg: the example waits, MyoSuite leaves once it
  # stops flexing.
  assert swing("sdk", rad(40), 0.0) == "e_swing"
  assert swing("myosuite", rad(40), 0.0) == "l_swing"
  # Past 50 deg and still flexing fast: MyoSuite leaves on the angle alone.
  assert swing("sdk", rad(55), 3.5) == "e_swing"
  assert swing("myosuite", rad(55), 3.5) == "l_swing"
  # 3 rad/s for the example, 3 deg/s for MyoSuite.
  assert swing("sdk", rad(55), 2.9) == "l_swing"
  assert swing("myosuite", rad(40), 0.06) == "e_swing"
  assert swing("myosuite", rad(40), 0.05) == "l_swing"


def test_the_load_exit_returns_a_loaded_early_swing_to_early_stance() -> None:
  for exit_, state in ((True, "e_stance"), (False, "e_swing")):
    controller = machine("sdk", exit_, "e_swing")
    assert after(controller, 0.5 * WEIGHT, rad(20), 0.0, 0.0) == [state]
  # Stance wins even when the swing conditions hold too; below 0.4 body weights the
  # exit does nothing.
  controller = machine("sdk", True, "e_swing", num_envs=2)
  assert after(controller, [400.0, 300.0], [rad(55), rad(20)], 0.0, 0.0) == [
    "e_stance",
    "e_swing",
  ]
  assert fsm.rule_set("sdk").swing_load_exit
  assert not fsm.rule_set("sdk", False).swing_load_exit
  assert not fsm.rule_set("myosuite").swing_load_exit
  assert fsm.rule_set("myosuite", False) == fsm.RULES["myosuite"]
  with pytest.raises(ValueError, match="sdk rules"):
    fsm.rule_set("myosuite", True)
  with pytest.raises(ValueError, match="Unknown rules"):
    fsm.rule_set("hkic")


@pytest.mark.parametrize("rules", ["sdk", "myosuite"])
def test_late_stance_and_late_swing_exits_are_the_same_in_both(rules: str) -> None:
  cases = (  # (state, load, knee, expected)
    ("l_stance", 0.16 * WEIGHT, 0.1, "l_stance"),
    ("l_stance", 0.14 * WEIGHT, 0.1, "e_swing"),
    ("l_swing", 0.0, rad(31), "l_swing"),
    ("l_swing", 0.0, rad(29), "e_stance"),
    ("l_swing", 0.41 * WEIGHT, rad(60), "e_stance"),
  )
  controller = machine(rules, num_envs=len(cases))
  controller.state[:] = torch.tensor([fsm.STATES.index(c[0]) for c in cases])
  load, knee = [c[1] for c in cases], [c[2] for c in cases]
  assert after(controller, load, knee, 0.0, 0.0) == [c[3] for c in cases]


def test_unloaded_and_extended_myosuite_rules_cycle_while_the_examples_wait() -> None:
  """Before foot contact (no load, knee under 30 deg and extending, ankle past
  6 deg) every MyoSuite exit already holds: one state per update, all four."""
  reading = (0.0, rad(10), -1.0, rad(8))
  controller = machine("myosuite")
  assert [after(controller, *reading)[0] for _ in range(5)] == [
    "l_stance",
    "e_swing",
    "l_swing",
    "e_stance",
    "l_stance",
  ]
  controller = machine("sdk")
  assert {after(controller, *reading)[0] for _ in range(5)} == {"e_stance"}


def test_a_reset_returns_to_early_stance_and_bad_settings_are_refused() -> None:
  controller = machine(num_envs=4)
  after(controller, 0.3 * WEIGHT, 0.0, 0.0, rad(7))
  assert controller.state.tolist() == [fsm.L_STANCE] * 4
  controller.reset(torch.tensor([1, 3]))
  assert controller.state.tolist() == [1, 0, 1, 0]
  controller.reset(slice(0, 1))
  assert controller.state.tolist() == [0, 0, 1, 0]
  controller.reset()
  assert controller.state.tolist() == [0] * 4
  assert controller.state.dtype == torch.long
  for weight in (0.0, -800.0, math.nan, math.inf):
    with pytest.raises(ValueError, match="Body weight"):
      fsm.BatchedFsm(fsm.rule_set("sdk"), weight, 1)
  with pytest.raises(TypeError, match="rule_set"):
    fsm.BatchedFsm("sdk", WEIGHT, 1)  # type: ignore[arg-type]


# ===== Beyond osl_deploy's tests =================================================


def test_comparisons_are_osl_deploys_float64_ones() -> None:
  """osl_deploy compares the float32 observation as Python floats, so a reading at
  the float32 rounding of a threshold lands on the float64 side of it: radians(6)
  rounds up in float32, so an ankle there is past 6 degrees. A float32 comparison
  would call it equal and stay in early stance."""
  ankle = float(np.float32(rad(6)))
  assert ankle > rad(6)
  assert after(machine("myosuite"), 0.0, 0.0, 0.0, ankle) == ["l_stance"]
  assert StateMachine(rule_set("myosuite"), WEIGHT).update(0.0, 0.0, 0.0, ankle) == (
    "l_stance"
  )
  # A load of exactly 0.15 body weights is not above it (120 N is exact in float32).
  assert after(machine("sdk", state="l_stance"), 0.15 * WEIGHT, 0.0, 0.0, 0.0) == [
    "l_stance"
  ]


def _random_readings(
  generator: torch.Generator, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """(load, knee, knee velocity, ankle, ankle velocity), float32, swept across every
  threshold; one value in ten sits exactly at a threshold's float32 rounding."""

  def uniform(lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, generator=generator)

  def at_thresholds(
    values: torch.Tensor, thresholds: tuple[float, ...]
  ) -> torch.Tensor:
    pick = torch.randint(len(thresholds), (n,), generator=generator)
    exact = torch.tensor(thresholds, dtype=torch.float32)[pick]
    return torch.where(torch.rand(n, generator=generator) < 0.1, exact, values)

  slow = torch.rand(n, generator=generator) < 0.3  # Near MyoSuite's 3 deg/s.
  velocity = torch.where(slow, uniform(-0.2, 0.2), uniform(-6.0, 6.0))
  loads = tuple(w * WEIGHT for w in (0.15, 0.25, 0.4))
  return (
    at_thresholds(uniform(-0.2, 1.0) * WEIGHT, loads),
    at_thresholds(uniform(-0.1, 1.3), (rad(30), rad(50))),
    at_thresholds(velocity, (3.0, rad(3))),
    at_thresholds(uniform(-0.4, 0.4), (rad(6),)),
    uniform(-5.0, 5.0),
  )


@pytest.mark.parametrize(
  "rules, swing_load_exit", [("sdk", None), ("sdk", False), ("myosuite", None)]
)
def test_batched_matches_osl_deploys_state_machine(
  rules: str, swing_load_exit: bool | None
) -> None:
  """Reading for reading, state for state, over a long random sequence with
  episode resets: the batched controller is osl_deploy's (torques to float32)."""
  generator = torch.Generator().manual_seed(7)
  n, steps = 64, 400
  batched = fsm.BatchedFsm(fsm.rule_set(rules, swing_load_exit), WEIGHT, n)
  scalar = [StateMachine(rule_set(rules, swing_load_exit), WEIGHT) for _ in range(n)]
  transitions: set[tuple[str, str]] = set()
  for step in range(steps):
    if step % 40 == 39:
      reset = torch.randperm(n, generator=generator)[: n // 4]
      batched.reset(reset)
      for i in reset.tolist():
        scalar[i].reset()
    load, knee, knee_velocity, ankle, ankle_velocity = _random_readings(generator, n)
    state = batched.update(load, knee, knee_velocity, ankle).tolist()
    torque = batched.torque(
      torch.stack((knee, ankle), dim=1), torch.stack((knee_velocity, ankle_velocity), 1)
    )
    expected_torque = []
    for i, machine_i in enumerate(scalar):
      before = machine_i.state
      reading = (
        float(load[i]),
        float(knee[i]),
        float(knee_velocity[i]),
        float(ankle[i]),
      )
      now = machine_i.update(*reading)
      transitions.add((before, now))
      assert fsm.STATES[state[i]] == now, (step, i, before, reading)
      expected_torque.append(
        machine_i.torque(
          float(knee[i]),
          float(knee_velocity[i]),
          float(ankle[i]),
          float(ankle_velocity[i]),
        )
      )
    torch.testing.assert_close(
      torque, torch.tensor(expected_torque, dtype=torch.float32), rtol=1e-5, atol=1e-4
    )
  # The sequence took every transition its rules have.
  cycle = {
    ("e_stance", "l_stance"),
    ("l_stance", "e_swing"),
    ("e_swing", "l_swing"),
    ("l_swing", "e_stance"),
  }
  assert cycle <= transitions
  load_exit = ("e_swing", "e_stance") in transitions
  assert load_exit == (rules == "sdk" and swing_load_exit is not False)


def test_the_pose_guess_picks_the_state_from_the_start_reading() -> None:
  controller = machine("myosuite", num_envs=5)
  # Loaded; at exactly 0.15 W (not above it) and flexing; unloaded and flexing;
  # unloaded and extending; unloaded and still.
  load = _t(0.5 * WEIGHT, 0.15 * WEIGHT, 0.0, 0.0, 0.1 * WEIGHT)
  knee_velocity = _t(-1.0, 2.0, 2.0, -2.0, 0.0)
  guess = controller.guess(load, knee_velocity)
  assert guess.dtype == torch.long
  assert [fsm.STATES[s] for s in guess.tolist()] == [
    "e_stance",
    "e_swing",
    "e_swing",
    "l_swing",
    "l_swing",
  ]
  # Only the masked envs take their guess; the others keep their state.
  controller.state[:] = fsm.L_STANCE
  mask = torch.tensor([True, True, False, True, False])
  controller.start_from_pose(mask, load, knee_velocity)
  assert controller.state.tolist() == [0, 2, 1, 3, 1]
  # The regular update then runs on the same reading (the action term's order), one
  # transition per env under MyoSuite's rules.
  knee, ankle = _t(rad(10), rad(55), rad(40), rad(10), rad(10)), _t(*[rad(8)] * 5)
  controller.update(load, knee, knee_velocity, ankle)
  assert [fsm.STATES[s] for s in controller.state.tolist()] == [
    "l_stance",  # e_stance: 0.5 W > 0.25 W.
    "l_swing",  # e_swing: knee 55 deg > 50 deg.
    "e_swing",  # l_stance: 0 W < 0.15 W.
    "e_stance",  # l_swing: knee 10 deg < 30 deg.
    "e_swing",  # l_stance: 0.1 W < 0.15 W.
  ]


def test_settings_tables_are_plain_and_match_the_constants() -> None:
  table = fsm.impedance_settings()
  assert list(table) == list(fsm.STATES)
  for state, gains in fsm.IMPEDANCE.items():
    for joint, (stiffness, damping, angle) in zip(fsm.JOINTS, gains, strict=True):
      assert table[state][joint] == {
        "stiffness_nm_per_rad": stiffness,
        "damping_nms_per_rad": damping,
        "equilibrium_deg": angle,
      }
  assert fsm.threshold_settings() == {
    "load_late_stance_bw": 0.25,
    "load_early_swing_bw": 0.15,
    "load_early_stance_bw": 0.4,
    "ankle_late_stance_rad": rad(6),
    "knee_late_swing_rad": rad(50),
    "knee_early_stance_rad": rad(30),
  }
