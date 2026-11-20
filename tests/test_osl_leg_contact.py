"""The OSL/left-leg contact sensor and penalty (CPU).

* Every OSL imitation scene carries the sensor once; stage 2 keeps stage 1's.
* The penalty: the share of substeps over the threshold, times the coefficient,
  inside the penalties' -1 floor with the other regularizers, times their scale.
* Stage 2 charges it to the channels ``leg_contact_owner`` names.
* It is recorded, compared and rebuilt with the reward settings, may be staged, and
  every task family takes its flags on the command line.
"""

from __future__ import annotations

import copy
import shlex
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_osl_reward_clfrl import (
  NUM_ENVS,
  _advance,
  _Command,
  _env,
  _params,
  _staged_env,
  _term,
  _train_cfg,
)

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.osl_mimic.leg_contact import (
  LEG_CONTACT_DIAGNOSTICS,
  LEG_CONTACT_SENSOR,
  add_leg_contact_sensor,
  contact_fraction,
  drop_leg_contact_sensor,
  substep_forces,
)
from mjlab.tasks.osl_mimic.mdp.rewards import (
  LEG_CONTACT_COEFFICIENT,
  LEG_CONTACT_THRESHOLD,
  RegularizerCfg,
  clf_regularizers,
  mimic_reward,
)
from mjlab.tasks.osl_mimic.reward_settings import (
  apply_reward_settings,
  describe_reward_settings,
  env_reward_settings,
  reward_flags,
  reward_mismatches,
  reward_settings,
)
from mjlab.tasks.osl_mimic.rl.runner import stage_accessors
from mjlab.tasks.osl_split.mdp.rewards import CHANNELS, SplitRewardCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

H = 5
"""Substeps per control step (the OSL tasks' decimation)."""
FORCES = (
  (0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 5.0, 0.0, 0.0),  # 5 N mid-step only.
  (12.0, 0.0, 0.0, 0.0, 1.0),
  (30.0, 30.0, 30.0, 30.0, 30.0),
  (0.5, 8.0, 40.0, 8.0, 0.5),
  (0.0, 0.0, 0.0, 0.0, 0.0),
)
"""Per env, the net contact force magnitude in each substep (N), newest first."""
FAMILIES = {
  "unicycle stage 1, MSK": "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-MSK-CLF-Compliant",
  "unicycle stage 1, robot": "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT-CLF",
  "stairs stage 1": "Mjlab-Stairs-OSL-Fullbody-MSK-CLF-Compliant",
  "single clip": "Mjlab-Mimic-OSL-Fullbody-MSK-CLF",
  "unicycle stage 2, MSK": "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant",
  "unicycle stage 2, robot": "Mjlab-Split-Unicycle-OSL-Fullbody-ROBOT-CLF-Compliant",
  "stairs stage 2": "Mjlab-Split-Stairs-OSL-Fullbody-MSK-CLF-Compliant",
  "stage 1.5": "Mjlab-FSM-Unicycle-OSL-Fullbody-MSK-CLF-Compliant",
}
"""One CLF task of every family that has the new reward, both models."""
PARAMS = "--env.rewards.mimic.params.regularizers"


def _history(forces: Any = FORCES, primaries: int = 1) -> torch.Tensor:
  """(B, P, H, 3) force histories of the magnitudes ``forces`` (along one axis per
  substep, so magnitudes are exact), the last primaries at half."""
  magnitude = torch.tensor(forces, dtype=torch.float32)
  directions = torch.nn.functional.normalize(
    torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.6, 0.8], [1.0, 0.0, 0.0]]), dim=-1
  )
  per_substep = directions[torch.arange(H) % 3]  # (H, 3)
  history = magnitude[:, None, :, None] * per_substep[None, None]
  if primaries > 1:
    rest = 0.5 * history.expand(-1, primaries - 1, -1, -1)
    history = torch.cat([history, rest], dim=1)
  return history


def _sensor(history: torch.Tensor) -> SimpleNamespace:
  return SimpleNamespace(
    data=SimpleNamespace(force=history[:, :, 0], force_history=history)
  )


def _fraction(threshold: float) -> torch.Tensor:
  return (torch.tensor(FORCES) > threshold).float().mean(-1)


def _with_sensor(env: Any, history: torch.Tensor | None = None) -> SimpleNamespace:
  sensor = _sensor(_history() if history is None else history)
  env.scene = SimpleNamespace(sensors={LEG_CONTACT_SENSOR: sensor})
  return sensor


##
# The sensor and the substep counting.
##


def test_substep_forces_and_contact_fraction() -> None:
  """The force per substep is the largest over the primaries; a substep counts when
  its force exceeds the threshold, strictly; the fraction is over the substeps.
  Without a substep history the current force is the one substep."""
  history = _history(primaries=2)
  forces = substep_forces(SimpleNamespace(force_history=history))
  torch.testing.assert_close(forces, torch.tensor(FORCES))
  for threshold, want in (
    (0.0, [0.0, 0.2, 0.4, 1.0, 1.0, 0.0]),
    (1.0, [0.0, 0.2, 0.2, 1.0, 0.6, 0.0]),
    (5.0, [0.0, 0.0, 0.2, 1.0, 0.6, 0.0]),  # 5 N is not over 5 N.
    (10.0, [0.0, 0.0, 0.2, 1.0, 0.2, 0.0]),
    (25.0, [0.0, 0.0, 0.0, 1.0, 0.2, 0.0]),
    (40.0, [0.0] * 6),
  ):
    torch.testing.assert_close(contact_fraction(forces, threshold), torch.tensor(want))
  current = SimpleNamespace(force=history[:, :, 0], force_history=None)
  now = substep_forces(current)
  assert now.shape == (NUM_ENVS, 1)
  torch.testing.assert_close(now[:, 0], torch.tensor(FORCES)[:, 0])


def test_every_osl_scene_has_one_leg_contact_sensor() -> None:
  """Every OSL imitation task's training and play scene carries the sensor once,
  over one control step's substeps, with its metrics; the helpers add it once and
  drop it (the stage-1 exporters do)."""
  tasks = sorted(t for t in list_tasks() if "OSL" in t)
  checked = 0
  for task in tasks:
    for play in (False, True):
      cfg = load_env_cfg(task, play=play)
      if "mimic" not in cfg.rewards:
        continue  # The G1-style tracking tasks.
      sensors = [s for s in cfg.scene.sensors if s.name == LEG_CONTACT_SENSOR]
      assert len(sensors) == 1, task
      (sensor,) = sensors
      assert isinstance(sensor, ContactSensorCfg)
      assert sensor.history_length == cfg.decimation == H
      assert sensor.reduce == "netforce" and sensor.fields == ("force",)
      for name, reduce in zip(LEG_CONTACT_DIAGNOSTICS, ("mean", "max"), strict=True):
        assert cfg.metrics[name].reduce == reduce, (task, name)
      assert "reg_leg_contact" in cfg.metrics
      split = "channels" in cfg.rewards["mimic"].params
      for channel in CHANNELS:
        assert (f"split_reg_leg_contact_{channel}" in cfg.metrics) == split
      regularizers = cfg.rewards["mimic"].params["regularizers"]
      # On in every CLF task; MimicReward keeps musclemimic's reward.
      clf = cfg.rewards["mimic"].params["clf"] is not None
      assert regularizers.leg_contact == (LEG_CONTACT_COEFFICIENT if clf else 0.0)
      assert regularizers.leg_contact_threshold == LEG_CONTACT_THRESHOLD
      assert regularizers.leg_contact_owner == "both"
      checked += 1
  assert checked >= 80
  cfg = load_env_cfg(FAMILIES["unicycle stage 1, MSK"])
  before = cfg.scene.sensors
  add_leg_contact_sensor(cfg)
  assert cfg.scene.sensors == before
  drop_leg_contact_sensor(cfg)
  assert cfg.scene.sensors == () and before != ()
  add_leg_contact_sensor(cfg)
  assert cfg.scene.sensors == before


##
# The penalty.
##


def _pair(
  model: str = "msk", split: bool = False, **regularizers: Any
) -> tuple[Any, Any, dict, dict, Any, _Command]:
  """Two terms on one fake env with the leg-contact sensor: one with
  ``regularizers`` (the leg contact on), one with the same regularizers and the leg
  contact at 0, and their params."""
  cmd = _Command(seed=21)
  env = _env(cmd, model)
  _with_sensor(env)
  on = _params("clf", regularizers=RegularizerCfg(**regularizers))
  off = copy.deepcopy(on)
  off["regularizers"].leg_contact = 0.0
  return _term(env, on, split), _term(env, off, split), on, off, env, cmd


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_the_penalty_sits_inside_the_floor(model: str) -> None:
  """``coefficient x fraction`` joins the regularizers inside the -1 floor, times
  their scale; the CLF violation stays outside it, and the clip at 0 is unchanged.
  The diagnostics report the contact rate, the hardest strike and the cost."""
  on, off, p_on, p_off, env, cmd = _pair(
    model, leg_contact=0.3, leg_contact_threshold=5.0, scale=0.5
  )
  reward = torch.zeros(NUM_ENVS)
  for _ in range(3):
    _advance(env, cmd)
    reward = on(env, **p_on)
    off(env, **p_off)
  cost = 0.3 * _fraction(5.0)
  c_on, c_off = on.components, off.components
  torch.testing.assert_close(c_on["reg_leg_contact"], cost)
  torch.testing.assert_close(c_on["reg_total"], 0.5 * cost)
  torch.testing.assert_close(c_off["reg_leg_contact"], torch.zeros(NUM_ENVS))
  floored = (off._cache["penalty"] - 0.5 * cost).clamp(min=-1.0)
  torch.testing.assert_close(on._cache["penalty"], floored)
  violation = 0.5 * c_on["clf_violation"]  # pen_scale x pen_w.
  torch.testing.assert_close(c_on["penalty_total"], floored - violation)
  level = c_on["clf_level"]
  torch.testing.assert_close(c_on["reward_preclip"], level + floored - violation)
  torch.testing.assert_close(reward, (level + floored - violation).clamp(min=0.0))
  torch.testing.assert_close(
    c_on["leg_contact"], (torch.tensor(FORCES).amax(-1) > 0).float()
  )
  torch.testing.assert_close(c_on["leg_contact_force"], torch.tensor(FORCES).amax(-1))
  # A strike in every substep at a large coefficient fills the floor by itself.
  big, _, p_big, _, env, cmd = _pair(model, leg_contact=10.0, leg_contact_threshold=5.0)
  _advance(env, cmd)
  big(env, **p_big)
  assert float(big._cache["penalty"][3]) == -1.0  # 30 N in every substep.


def test_at_zero_the_reward_ignores_the_sensor() -> None:
  """At coefficient 0 (the old reward's flag) the reward, its pieces and its state
  are those of a term in a scene without the sensor, bit for bit: the sensor only
  adds diagnostics."""
  cmd = _Command(seed=22)
  env = _env(cmd)
  off = replace(clf_regularizers("msk"), leg_contact=0.0)
  bare_params = _params("clf", regularizers=off)
  bare = _term(env, bare_params)
  _with_sensor(env)
  params = copy.deepcopy(bare_params)
  sensed = _term(env, params)
  for _ in range(4):
    _advance(env, cmd)
    want, got = bare(env, **bare_params), sensed(env, **params)
    torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)
    for key, value in bare.components.items():
      torch.testing.assert_close(sensed.components[key], value, rtol=0.0, atol=0.0)
    assert set(sensed.components) - set(bare.components) == set(LEG_CONTACT_DIAGNOSTICS)
    assert "leg_contact" not in sensed._reg_cache


def test_the_penalty_needs_the_sensor() -> None:
  """On in a scene without the sensor: refused at build; staged on later: refused
  on the step it would be computed."""
  cmd = _Command(seed=23)
  env = _env(cmd)
  params = _params("clf", regularizers=RegularizerCfg(leg_contact=0.2))
  with pytest.raises(ValueError, match="no 'osl_leg_contact' contact sensor"):
    _term(env, params)
  params["regularizers"].leg_contact = 0.0
  term = _term(env, params)
  _advance(env, cmd)
  term(env, **params)
  params["regularizers"].leg_contact = 0.2
  with pytest.raises(RuntimeError, match="no 'osl_leg_contact' contact sensor"):
    term(env, **params)
  for bad in (
    RegularizerCfg(leg_contact=-0.1),
    RegularizerCfg(leg_contact_threshold=-1.0),
    RegularizerCfg(leg_contact_threshold=float("nan")),
    RegularizerCfg(leg_contact_owner="team"),  # type: ignore[arg-type]
  ):
    _with_sensor(env)
    with pytest.raises(ValueError, match="leg_contact"):
      _term(env, _params("clf", regularizers=bad))


@pytest.mark.parametrize("owner", ["both", "human", "osl"])
def test_stage2_charges_its_owners(owner: str) -> None:
  """Stage 2 charges the cost in full to each channel ``leg_contact_owner`` names,
  inside that channel's floor; the team reward (stage 1's) holds it once. Without
  owner penalties both channels pay the team's."""
  on, off, p_on, p_off, env, cmd = _pair(
    split=True, leg_contact=0.4, leg_contact_threshold=1.0, leg_contact_owner=owner
  )
  for _ in range(2):
    _advance(env, cmd)
    on(env, **p_on)
    off(env, **p_off)
  cost = 0.4 * _fraction(1.0)
  owners = CHANNELS if owner == "both" else (owner,)
  zeros = torch.zeros(NUM_ENVS)
  for channel in CHANNELS:
    pays = channel in owners
    got = on.components[f"split_reg_leg_contact_{channel}"]
    torch.testing.assert_close(got, cost if pays else zeros)
    violation = 0.5 * on.components[f"split_violation_{channel}"]
    before = off.components[f"split_penalty_{channel}"] + violation
    floored = (before - cost).clamp(min=-1.0) if pays else before
    torch.testing.assert_close(
      on.components[f"split_penalty_{channel}"], floored - violation
    )
  team = (off._cache["penalty"] - cost).clamp(min=-1.0)
  torch.testing.assert_close(on._cache["penalty"], team)
  torch.testing.assert_close(
    on.components["split_leg_contact"], on.components["leg_contact"]
  )
  # Without owner penalties, both channels pay the team's penalty.
  shared = copy.deepcopy(p_on)
  shared["channels"] = SplitRewardCfg(owner_penalties=False)
  term = _term(env, shared, split=True)
  term(env, **shared)
  for channel in CHANNELS:
    torch.testing.assert_close(
      term.components[f"split_reg_leg_contact_{channel}"], cost
    )


##
# Across launches: the settings, the flags, the stages.
##


def test_settings_record_compare_and_rebuild() -> None:
  """The settings record the coefficient and threshold (and in stage 2 the owner);
  a record without them (every earlier run) counts as no penalty, so it differs
  from the CLF tasks' default until the old flag turns it off; the flags they write
  parse back to them exactly; a warm start never compares stage 1's owner."""
  stage1 = load_env_cfg(FAMILIES["unicycle stage 1, MSK"])
  stage2 = load_env_cfg(FAMILIES["unicycle stage 2, MSK"])
  s1, s2 = env_reward_settings(stage1), env_reward_settings(stage2)
  assert s1 is not None and s2 is not None
  assert (s1["reg_leg_contact"], s1["reg_leg_contact_owner"]) == (0.25, None)
  assert (s2["reg_leg_contact"], s2["reg_leg_contact_owner"]) == (0.25, "both")
  assert s1["reg_leg_contact_threshold"] == LEG_CONTACT_THRESHOLD
  earlier = {k: v for k, v in s2.items() if not k.startswith("reg_leg_contact")}
  assert reward_mismatches(earlier, s2) == [
    "regularizer leg_contact (x scale) 0.0 vs 0.25"
  ]
  assert reward_mismatches(earlier, {**s2, "reg_leg_contact": 0.0}) == []
  on: dict[str, Any] = {**s2, "reg_leg_contact_threshold": 1.0 / 3.0}
  assert reward_mismatches(on, {**on, "reg_leg_contact_threshold": 5.0}) == [
    f"leg_contact_threshold {1.0 / 3.0!r} vs 5.0"
  ]
  assert reward_mismatches(on, {**on, "reg_leg_contact_owner": "osl"}) == [
    "leg_contact_owner 'both' vs 'osl'"
  ]
  # Stage 1 records no owner, so a warm start compares only what it computes.
  s1_on = {**s1, "reg_leg_contact": 0.25, "reg_leg_contact_threshold": 1.0 / 3.0}
  assert reward_mismatches(s1_on, {**on, "reg_leg_contact_owner": "osl"}) == []
  # Off: the threshold and owner are not compared.
  off = {**s2, "reg_leg_contact": 0.0}
  assert reward_mismatches(off, {**off, "reg_leg_contact_threshold": 3.0}) == []
  assert "leg_contact 0.25 (over 0.333 N, both channels)" in (
    describe_reward_settings(on)
  )
  # The flags rebuild the settings exactly, through the command line.
  task = FAMILIES["unicycle stage 2, MSK"]
  want = {**on, "reg_leg_contact_owner": "osl"}
  flags = reward_flags(want)
  assert f"{PARAMS}.leg-contact 0.25" in flags
  assert f"{PARAMS}.leg-contact-owner osl" in flags
  parsed = env_reward_settings(_train_cfg(task, *shlex.split(" ".join(flags))).env)
  assert parsed == want
  # A record applied to a config (an evaluation scores under its run's reward).
  apply_reward_settings(stage2, want)
  assert env_reward_settings(stage2) == want
  regularizers = stage2.rewards["mimic"].params["regularizers"]
  assert regularizers.leg_contact_owner == "osl"
  assert regularizers.leg_contact_threshold == 1.0 / 3.0
  assert reward_settings(None, None)["reg_leg_contact_threshold"] is None


def test_the_coefficient_is_staged_live() -> None:
  """``rewards.mimic.params.regularizers.leg_contact`` is an env-stage path the term
  reads on every step (it ramps with ``scale`` too); the threshold and owner are
  read at build and refused."""
  cmd = _Command(seed=24)
  env = _env(cmd)
  _with_sensor(env)
  params = _params("clf", regularizers=RegularizerCfg(leg_contact_threshold=1.0))
  term_cfg = RewardTermCfg(func=mimic_reward, weight=1.0, params=params)
  term = mimic_reward(term_cfg, env)
  staged = _staged_env(term_cfg)
  get, set_ = stage_accessors(staged, "rewards.mimic.params.regularizers.leg_contact")
  assert get() == 0.0
  _advance(env, cmd)
  term(env, **params)
  assert "leg_contact" not in term._reg_cache
  set_(0.5)
  _advance(env, cmd)
  term(env, **params)
  torch.testing.assert_close(term.components["reg_leg_contact"], 0.5 * _fraction(1.0))
  assert "regularizers.leg_contact" in mimic_reward.LIVE_PARAMS
  for name in ("leg_contact_threshold", "leg_contact_owner"):
    with pytest.raises(ValueError, match="once, when it is built"):
      stage_accessors(staged, f"rewards.mimic.params.regularizers.{name}")


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_takes_the_flags(family: str) -> None:
  """The flags reach the reward through tyro, as ``train.py`` builds the config, in
  every family of the new reward (the owner too, in stage 2)."""
  task = FAMILIES[family]
  split = "Split" in task or "FSM" in task
  argv = [f"{PARAMS}.leg-contact", "0.5", f"{PARAMS}.leg-contact-threshold", "5.0"]
  if split:
    argv += [f"{PARAMS}.leg-contact-owner", "osl"]
  regularizers = _train_cfg(task, *argv).env.rewards["mimic"].params["regularizers"]
  assert regularizers.leg_contact == 0.5
  assert regularizers.leg_contact_threshold == 5.0
  assert regularizers.leg_contact_owner == ("osl" if split else "both")
  default = _train_cfg(task).env.rewards["mimic"].params["regularizers"]
  assert default.leg_contact == LEG_CONTACT_COEFFICIENT == 0.25
  assert default.leg_contact_threshold == 10.0 and default.leg_contact_owner == "both"
  old = _train_cfg(task, f"{PARAMS}.leg-contact", "0")  # The old reward's flag.
  assert old.env.rewards["mimic"].params["regularizers"].leg_contact == 0.0
