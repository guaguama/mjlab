"""Tests for stage 2's gait-symmetry penalty and metric
(``tasks/osl_split/mdp/symmetry.py``, the ``gait_symmetry`` env setting in
``config/env_cfgs.py`` and the runner's validation metric).

CPU only and without an mjlab env, except the slow test:

* the term on a fake gait-library command: zero for perfect tracking, the expected
  d for one-sided errors, the half-stride lag through the ring buffer, and the
  masks (episode start, reset, clip change, blend, idle clip);
* the configs as ``train`` parses them: off by default and then absent, on with the
  flags, refused on the stairs' single clip and for bad values;
* the runner, on the fakes of ``test_osl_split_runner``: both agents earn the
  penalty, evaluation envs carry it and the validation env the metric, the
  validation pass reports the asymmetry with the penalty off, and a resumed launch
  takes an added penalty and can validate its checkpoint first.

The slow test builds the real stage-2 env with the penalty on
(``MJLAB_OSL_UNICYCLE_LIBRARY``).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import tyro
from tensordict import TensorDict
from test_osl_split_runner import (
  HUMAN_DIM,
  ITERATIONS,
  STAGE_START,
  TWIST_SPLIT,
  _FakeEnv,
  _quick_cfg,
  _Runner,
  _scalars,
  _write_warm_start,
)

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommand, GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic.mdp import mimic_component
from mjlab.tasks.osl_mimic.rl import runner as mimic_runner
from mjlab.tasks.osl_split.config.env_cfgs import (
  GaitSymmetryCfg,
  OslSplitEnvCfg,
  add_gait_symmetry_metric,
  osl_split_env_cfg,
  sync_gait_symmetry,
)
from mjlab.tasks.osl_split.mdp import (
  SYMMETRY_DIAGNOSTICS,
  SYMMETRY_METRIC,
  SYMMETRY_PAIRS,
  gait_symmetry,
  stride_periods,
)
from mjlab.tasks.osl_split.rl.runner import AgentRewards, SplitOnPolicyRunner
from mjlab.tasks.osl_split_stairs.config.env_cfgs import osl_split_stairs_env_cfg

SIGMA = 0.15
JOINTS = (
  "lumbar_extension",
  *(left for _, _, left in SYMMETRY_PAIRS),
  *(right for _, right, _ in reversed(SYMMETRY_PAIRS)),
)
"""The fake robot's joints: one the term ignores, then the pairs out of order."""
FRAMES = (8, 9, 14)
"""Two walking clips (half strides 4 and 5: 9 frames round half up) and the idle
clip, whose 14 frames would make a longer half stride than any walking clip's."""
IDLE = 2
BLEND = 3
PENALTY = 0.1
SPLIT_TASK = "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
FSM_TASK = "Mjlab-FSM-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
STAIRS_TASK = "Mjlab-Split-Stairs-OSL-Fullbody-MSK-CLF-Compliant"
FLAGS = "--env.gait-symmetry.weight 0.1 --env.gait-symmetry.sigma 0.2"


##
# The term on a fake command.
##


class _Command:
  """What :class:`gait_symmetry` reads of the gait library's command: the robot's
  joints, ``reference(1)``, each env's clip, the clips' lengths, the idle clip and
  the blend."""

  def __init__(self, num_envs: int) -> None:
    self.robot = SimpleNamespace(
      joint_names=list(JOINTS),
      data=SimpleNamespace(joint_pos=torch.zeros(num_envs, len(JOINTS))),
    )
    self.ref = torch.zeros(num_envs, len(JOINTS))
    self.clip = torch.zeros(num_envs, dtype=torch.long)
    self.num_frames = torch.tensor(FRAMES)
    self.idle_clip = IDLE
    self.blend_steps = BLEND
    self.blend_count = torch.full((num_envs,), BLEND, dtype=torch.long)

  def reference(self, offset: int) -> SimpleNamespace:
    assert offset == 1  # The reward's own comparison.
    return SimpleNamespace(joint_pos=self.ref)

  def track(self, errors: Mapping[str, torch.Tensor | float] | None = None) -> None:
    """The robot at the reference plus ``{joint: error}``."""
    self.robot.data.joint_pos = self.ref.clone()
    for joint, error in (errors or {}).items():
      self.robot.data.joint_pos[:, JOINTS.index(joint)] += error


def _term(num_envs: int) -> tuple[gait_symmetry, _Command, Any]:
  cmd = _Command(num_envs)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    command_manager=SimpleNamespace(get_term=lambda name: cmd),
  )
  cfg = RewardTermCfg(
    func=gait_symmetry,
    weight=-PENALTY,
    params={"command_name": "motion", "sigma": SIGMA},
  )
  return gait_symmetry(cfg, cast(Any, env)), cmd, env


def _pair(name: str) -> tuple[int, str, str]:
  (index,) = [i for i, pair in enumerate(SYMMETRY_PAIRS) if pair[0] == name]
  return index, SYMMETRY_PAIRS[index][1], SYMMETRY_PAIRS[index][2]


def test_perfect_tracking_scores_zero() -> None:
  """A robot on its reference adds no asymmetry, whatever the reference (here
  random, so not symmetric at all): d, s and the value are 0 once d counts."""
  term, cmd, env = _term(3)
  generator = torch.Generator().manual_seed(0)
  for step in range(4 * FRAMES[0]):
    cmd.ref = torch.randn(3, len(JOINTS), generator=generator)
    cmd.track()
    value = term(env, "motion", SIGMA)
    assert bool((term.valid == (step >= 4)).all()), step
    assert bool((term.d == 0.0).all()) and bool((term.s == 0.0).all())
    assert bool((value == 0.0).all())
    assert set(term.components) == set(SYMMETRY_DIAGNOSTICS)


def test_one_sided_errors_give_the_expected_d() -> None:
  """d is the OSL side's error minus the intact side's: an OSL knee error alone,
  an intact hip-rotation error alone, and s and the value they make."""
  term, cmd, env = _term(2)
  knee, osl_knee, _ = _pair("knee")
  rotation, _, intact_rotation = _pair("hip_rotation")
  cmd.ref = torch.randn(2, len(JOINTS))
  cmd.track({osl_knee: 0.1, intact_rotation: -0.06})
  for _ in range(4):
    term(env, "motion", SIGMA)
  value = term(env, "motion", SIGMA)
  assert bool(term.valid.all())
  expected = torch.zeros(2, len(SYMMETRY_PAIRS))
  expected[:, knee] = 0.1
  expected[:, rotation] = 0.06
  torch.testing.assert_close(term.d, expected)
  group_sq = torch.tensor([0.1**2, 0.0, 0.06**2 / 3]).expand(2, 3)
  torch.testing.assert_close(term.group_sq, group_sq)
  s = group_sq.mean(-1)
  torch.testing.assert_close(term.s, s)
  torch.testing.assert_close(value, 1.0 - torch.exp(-s / SIGMA**2))
  assert bool(((value > 0.0) & (value < 1.0)).all())


def test_the_intact_error_arrives_half_a_stride_later() -> None:
  """The ring buffer hands back the intact side's error exactly half a stride
  (``num_frames`` / 2, rounded half up) after it was made, through many wraps of
  the buffer, which holds the longest walking half stride plus the current step
  (the idle clip's capped)."""
  term, cmd, env = _term(2)
  assert term.half_periods.tolist() == [4, 5, 5]
  assert term.history_length == 6
  cmd.clip[:] = torch.tensor([0, 1])
  knee, _, intact_knee = _pair("knee")
  half = torch.tensor([4, 5])
  for t in range(40):
    # A different intact error every step: 0.01 t.
    cmd.track({intact_knee: 0.01 * t})
    term(env, "motion", SIGMA)
    counts = t >= half
    assert term.valid.tolist() == counts.tolist(), t
    lagged = 0.01 * (t - half).clamp(min=0).float()
    expected = torch.where(counts, -lagged, torch.zeros(2))
    torch.testing.assert_close(term.d[:, knee], expected)


def test_masks() -> None:
  """d counts only half a stride into the env's history: after a reset, after a
  clip change (at the new clip's half stride), after a blend has run out (the
  history restarts on every blended step), and never on the idle clip. A reset
  leaves the step's outputs alone."""
  term, cmd, env = _term(4)
  cmd.clip[:] = torch.tensor([0, 0, 0, IDLE])
  _, osl_ankle, _ = _pair("ankle")
  cmd.track({osl_ankle: 0.2})

  def steps(count: int) -> list[list[bool]]:
    valid = []
    for _ in range(count):
      term(env, "motion", SIGMA)
      valid.append(term.valid.tolist())
    return valid

  assert steps(5)[-1] == [True, True, True, False]
  d = term.d.clone()
  # Reset env 0, switch env 1 to clip 1, and run a blend on env 2.
  term.reset(torch.tensor([0]))
  torch.testing.assert_close(term.d, d)
  assert term.valid.tolist() == [True, True, True, False]
  cmd.clip[1] = 1
  cmd.blend_count[2] = 1
  first = steps(1)[0]
  assert first == [False, False, False, False]
  cmd.blend_count[2] = 2
  assert steps(1)[0] == [False, False, False, False]
  cmd.blend_count[2] = BLEND  # The blend is over: the history restarted last step.
  valid = steps(6)
  # Env 0 counts again on its 5th step since the reset, env 1 on its 6th since the
  # switch, env 2 on its 4th after the last blended step (that step, whose
  # reference(1) is the new clip's alone, is its history's first), env 3 never.
  assert [row[0] for row in valid] == [False, False, True, True, True, True]
  assert [row[1] for row in valid] == [False, False, False, True, True, True]
  assert [row[2] for row in valid] == [False, False, False, True, True, True]
  assert not any(row[3] for row in valid)
  assert bool((term.s[3] == 0.0) & (term.d[3] == 0.0).all())


def test_the_term_needs_a_stride_period() -> None:
  """A command with one non-looping clip (the stairs') has no stride period."""
  stairs = SimpleNamespace(num_frames=731, robot=None)
  with pytest.raises(ValueError, match="stride period"):
    stride_periods(stairs)
  env = SimpleNamespace(
    num_envs=1, device="cpu", command_manager=SimpleNamespace(get_term=lambda n: stairs)
  )
  cfg = RewardTermCfg(func=gait_symmetry, weight=-1.0, params={"command_name": "m"})
  with pytest.raises(ValueError, match="stride period"):
    gait_symmetry(cfg, cast(Any, env))


##
# The env setting.
##


def _parse(task: str, flags: str = "") -> TrainConfig:
  """The flags parsed as ``train`` parses them."""
  return tyro.cli(
    TrainConfig,
    args=flags.split(),
    default=TrainConfig.from_task(task),
    config=mjlab.TYRO_FLAGS,
  )


def _has_symmetry(cfg: Any) -> bool:
  diagnostics = set(SYMMETRY_DIAGNOSTICS) & set(cfg.metrics)
  return SYMMETRY_METRIC in cfg.rewards or bool(diagnostics)


@pytest.mark.parametrize("task", [SPLIT_TASK, FSM_TASK, STAIRS_TASK])
def test_off_by_default_the_penalty_is_absent(task: str) -> None:
  """Without the flags (or at weight 0) no task has the term or its metrics: the
  config is the one without the setting (``test_env_cfg_contract`` pins the
  default rewards and metrics term for term)."""
  default = _parse(task).env
  assert isinstance(default, OslSplitEnvCfg)
  assert default.gait_symmetry == GaitSymmetryCfg()
  assert not _has_symmetry(default)
  explicit = _parse(task, "--env.gait-symmetry.weight 0.0").env
  assert asdict(explicit) == asdict(default)


@pytest.mark.parametrize("task", [SPLIT_TASK, FSM_TASK])
def test_the_flags_add_the_penalty(task: str) -> None:
  """``--env.gait-symmetry.weight w --env.gait-symmetry.sigma s`` add the reward
  term at weight -w and its metrics; everything else stays. Turning it off on a
  built config takes both out again."""
  default = _parse(task).env
  cfg = _parse(task, FLAGS).env
  assert list(cfg.rewards) == [*default.rewards, SYMMETRY_METRIC]
  term = cfg.rewards[SYMMETRY_METRIC]
  assert term.func is gait_symmetry and term.weight == -0.1
  assert term.params == {"command_name": "motion", "sigma": 0.2}
  for name in SYMMETRY_DIAGNOSTICS:
    metric = cfg.metrics[name]
    assert metric.func is mimic_component
    assert metric.params == {"name": name, "reward_name": SYMMETRY_METRIC}
  for name, reward in default.rewards.items():
    assert cfg.rewards[name] == reward
  assert {k: v for k, v in cfg.metrics.items() if k not in SYMMETRY_DIAGNOSTICS} == (
    default.metrics
  )
  assert isinstance(cfg, OslSplitEnvCfg)
  cfg.gait_symmetry = GaitSymmetryCfg()
  sync_gait_symmetry(cfg)
  assert cfg.rewards == default.rewards and cfg.metrics == default.metrics


def test_refusals() -> None:
  """The stairs' single clip has no stride period: the flags are refused there,
  as are a negative weight and a non-positive sigma."""
  with pytest.raises(SystemExit):
    _parse(STAIRS_TASK, FLAGS)
  stairs = osl_split_stairs_env_cfg()
  stairs.gait_symmetry = GaitSymmetryCfg(weight=0.1)
  with pytest.raises(ValueError, match="no stride period"):
    sync_gait_symmetry(stairs)
  cfg = osl_split_env_cfg()
  for settings, match in (
    (GaitSymmetryCfg(weight=-0.1), "weight"),
    (GaitSymmetryCfg(weight=0.1, sigma=0.0), "sigma"),
    (GaitSymmetryCfg(weight=math.inf), "weight"),
  ):
    cfg.gait_symmetry = settings
    with pytest.raises(ValueError, match=match):
      sync_gait_symmetry(cfg)


def test_the_validation_metric_needs_a_stride_period() -> None:
  """:func:`add_gait_symmetry_metric` gives a gait-library config the metrics term
  (at its sigma) and leaves the stairs' config alone."""
  cfg = osl_split_env_cfg("msk", "compliant", "clf", "validation")
  cfg.gait_symmetry = GaitSymmetryCfg(sigma=0.3)
  assert add_gait_symmetry_metric(cfg)
  metric = cfg.metrics[SYMMETRY_METRIC]
  assert isinstance(metric, MetricsTermCfg) and metric.func is gait_symmetry
  assert metric.params == {"command_name": "motion", "sigma": 0.3}
  assert SYMMETRY_METRIC not in cfg.rewards
  stairs = osl_split_stairs_env_cfg(mode="validation")
  before = dict(stairs.metrics)
  assert not add_gait_symmetry_metric(stairs)
  assert stairs.metrics == before


##
# The runner.
##


class _PenaltyEnv(_FakeEnv):
  """The runner tests' fake env with the gait-symmetry penalty as a third reward
  term: a random value in [0, 1) at weight -``PENALTY`` every step, which both
  agents earn in full."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    manager = self.reward_manager
    manager.cfgs[SYMMETRY_METRIC] = SimpleNamespace(
      func=SimpleNamespace(components={}), weight=-PENALTY
    )
    manager._step_reward = torch.zeros(self.num_envs, 3)

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    obs, reward, dones, extras = super().step(actions)
    penalty = -PENALTY * self._rand(self.num_envs)
    manager = self.reward_manager
    manager._step_reward = torch.cat([manager._step_reward, penalty[:, None]], -1)
    self.expected_agent_rewards[-1] = self.expected_agent_rewards[-1] + penalty[:, None]
    return obs, reward + penalty, dones, extras


def test_both_agents_earn_the_penalty() -> None:
  """The penalty is an additive term: both agents earn its step value in full
  unless ``additive_term_weights`` lists it."""
  env = _PenaltyEnv()
  rewards = AgentRewards(cast(Any, env), {"twist": TWIST_SPLIT})
  assert rewards.additive == {"twist": TWIST_SPLIT, SYMMETRY_METRIC: (1.0, 1.0)}
  env.step(torch.zeros(env.num_envs, env.num_actions))
  torch.testing.assert_close(rewards(), env.expected_agent_rewards[-1])


def test_evaluation_envs_carry_the_penalty_and_validation_the_metric(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Validation and video envs, rebuilt from the task, take the training env's
  penalty (stress plays a copy of the training config); the validation env alone
  also gets the metrics term, penalty on or off. The env is not built: its
  constructor records the config."""
  built: list[Any] = []

  def build(cfg: Any, device: str, **kwargs: Any) -> _FakeEnv:  # ManagerBasedRlEnv.
    del device, kwargs
    built.append(cfg)
    return _FakeEnv(num_envs=cfg.scene.num_envs, seed=1)

  monkeypatch.setattr(mimic_runner, "ManagerBasedRlEnv", build)
  monkeypatch.setattr(mimic_runner, "RslRlVecEnvWrapper", lambda env: env)
  env = _FakeEnv()
  runner = SplitOnPolicyRunner(cast(Any, env), _quick_cfg(None), None, "cpu")

  env.cfg = osl_split_env_cfg("msk", "compliant", "clf", "train")
  runner._make_validation_env()
  validation = built[-1]
  assert SYMMETRY_METRIC not in validation.rewards
  metric = validation.metrics[SYMMETRY_METRIC]
  assert metric.func is gait_symmetry and metric.params["sigma"] == SIGMA

  train = osl_split_env_cfg("msk", "compliant", "clf", "train")
  train.gait_symmetry = GaitSymmetryCfg(weight=0.1, sigma=0.2)
  sync_gait_symmetry(train)
  env.cfg = train
  runner._make_validation_env()
  validation = built[-1]
  for cfg in (validation, runner._eval_env_cfg("video")):
    assert isinstance(cfg, OslSplitEnvCfg)
    term = cfg.rewards[SYMMETRY_METRIC]
    assert term.weight == -0.1 and term.params["sigma"] == 0.2
    assert cfg.gait_symmetry == GaitSymmetryCfg(weight=0.1, sigma=0.2)
  assert validation.metrics[SYMMETRY_METRIC].params["sigma"] == 0.2
  assert SYMMETRY_METRIC not in runner._eval_env_cfg("video").metrics
  stress = runner._eval_env_cfg("stress")
  assert stress.rewards[SYMMETRY_METRIC] == train.rewards[SYMMETRY_METRIC]
  assert SYMMETRY_METRIC not in stress.metrics


def test_validation_reports_the_asymmetry_with_the_penalty_off() -> None:
  """With no penalty in training, the validation pass still reports the gait
  asymmetry from its env's metrics term: the RMS of each group's d in degrees
  over the steps it counts, overall and per clip family (none for the idle
  clip), and the share of env-steps counted. Env 3's episode ends early, which
  restarts its history."""
  n, num_steps = 4, 12
  term, cmd, metric_env = _term(n)
  cmd.clip[:] = torch.tensor([0, 1, IDLE, 0])
  _, osl_knee, _ = _pair("knee")
  _, _, intact_ankle = _pair("ankle")
  _, osl_flexion, _ = _pair("hip_flexion")
  a, b, c = 0.10, 0.05, 0.20
  errors = {
    osl_knee: torch.tensor([a, 0.0, a, 0.0]),
    intact_ankle: torch.tensor([0.0, b, 0.0, 0.0]),
    osl_flexion: torch.tensor([0.0, 0.0, 0.0, c]),
  }
  ends = {6: 3}  # Step index -> the env whose episode ends there.
  runner = SplitOnPolicyRunner.__new__(SplitOnPolicyRunner)
  runner.validation_cfg = {"num_steps": num_steps}
  runner.device = "cpu"
  masks = {
    "fwd": torch.tensor([True, False, False, True]),
    "turn": torch.tensor([False, True, False, False]),
    "idle": torch.tensor([False, False, True, False]),
  }
  runner._validation_groups = lambda env: masks
  split = SimpleNamespace(components={})
  step_count = 0

  class Rewards:
    term = split

    def __call__(self) -> torch.Tensor:
      return torch.zeros(n, 2)

  def obs() -> TensorDict:
    return TensorDict({"actor": torch.zeros(n, 1)}, batch_size=[n])

  def reset() -> tuple[TensorDict, dict]:
    term.reset(None)
    return obs(), {}

  def step(actions: torch.Tensor):
    nonlocal step_count
    cmd.ref = torch.randn(n, len(JOINTS))
    cmd.track(errors)
    term(metric_env, "motion", SIGMA)  # The metrics manager, at reward time.
    dones = torch.zeros(n)
    if step_count in ends:
      dones[ends[step_count]] = 1.0
      term.reset(torch.tensor([ends[step_count]]))  # The env's auto-reset.
    step_count += 1
    return obs(), torch.zeros(n), dones, {"time_outs": torch.zeros(n)}

  mimic = SimpleNamespace(func=split)
  env = SimpleNamespace(
    num_envs=n,
    device="cpu",
    reward_manager=SimpleNamespace(
      active_terms=["mimic"], get_term_cfg=lambda name: mimic
    ),
    metrics_manager=SimpleNamespace(cfg={SYMMETRY_METRIC: SimpleNamespace(func=term)}),
  )
  vec = SimpleNamespace(unwrapped=env, device="cpu", reset=reset, step=step)

  def policy(observation: TensorDict) -> torch.Tensor:
    return torch.zeros(n, HUMAN_DIM + 2)

  metrics = runner._validation_pass(
    cast(Any, vec), cast(Any, policy), cast(Any, Rewards())
  )
  # Counted steps: env 0 from its 5th step (8), env 1 from its 6th (7), env 3 its
  # 5th to 7th, then the 5th after the end (4); never the idle env 2.
  counted = {0: 8, 1: 7, 3: 4}
  sq = {  # Each env's summed d² per group: knee, ankle, hip.
    0: (counted[0] * a**2, 0.0, 0.0),
    1: (0.0, counted[1] * b**2, 0.0),
    3: (0.0, 0.0, counted[3] * c**2 / 3),
  }

  def expected(envs: tuple[int, ...]) -> dict[str, float]:
    total = sum(counted[e] for e in envs)
    ms = [sum(sq[e][g] for e in envs) / total for g in range(3)]
    out = {
      f"gait_symmetry_{name}_deg": math.degrees(math.sqrt(value))
      for name, value in zip(("knee", "ankle", "hip"), ms, strict=True)
    }
    out["gait_symmetry_deg"] = math.degrees(math.sqrt(sum(ms) / 3))
    return out

  for key, value in expected((0, 1, 3)).items():
    assert metrics[key] == pytest.approx(value), key
  for group, envs in (("fwd", (0, 3)), ("turn", (1,))):
    for key, value in expected(envs).items():
      assert metrics[f"{key}/{group}"] == pytest.approx(value), (key, group)
  assert not any(key.endswith("/idle") and "symmetry" in key for key in metrics)
  assert metrics["gait_symmetry_valid"] == pytest.approx(19 / (n * num_steps))


def _resume_with_the_penalty(
  checkpoint: Path, log_dir: Path, at_resume: bool, monkeypatch: pytest.MonkeyPatch
) -> list[int]:
  """Resume ``checkpoint`` for one iteration on the fake env with the penalty,
  checking that both agents earn it; returns the iterations validated."""
  cfg = _quick_cfg(None)
  cfg["validation"].update(interval=100, at_resume=at_resume)
  env = _PenaltyEnv(seed=1)
  resumed = _Runner(cast(Any, env), cfg, str(log_dir), "cpu")
  resumed.load(str(checkpoint))
  assert resumed._rewards.additive[SYMMETRY_METRIC] == (1.0, 1.0)
  recorded: list[torch.Tensor] = []
  process_env_step = resumed.alg.process_env_step

  def spy(obs: TensorDict, rewards: torch.Tensor, *args: Any) -> None:
    recorded.append(rewards.clone())
    process_env_step(obs, rewards, *args)

  monkeypatch.setattr(resumed.alg, "process_env_step", spy)
  resumed.learn(1)
  for got, want in zip(recorded, env.expected_agent_rewards, strict=True):
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)
  tag = "Validation/err_site_abs"
  return sorted(_scalars(log_dir, tag)[tag])


def test_a_resumed_launch_takes_the_penalty_and_validates_first(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A stage-2 checkpoint trained without the penalty resumes with it (both agents
  earn it from the first step), and ``validation.at_resume`` validates the
  checkpoint's own weights before the first update, at its iteration. Off, the
  resumed launch validates at its last iteration only."""
  ws_path = tmp_path / "ws.pt"
  _write_warm_start(ws_path)
  runner = _Runner(cast(Any, _FakeEnv()), _quick_cfg(ws_path), str(tmp_path / "run"))
  runner.learn(ITERATIONS)
  last = STAGE_START + ITERATIONS - 1
  checkpoint = tmp_path / "run" / f"model_{last}.pt"
  off = _resume_with_the_penalty(checkpoint, tmp_path / "off", False, monkeypatch)
  assert off == [last + 1]
  on = _resume_with_the_penalty(checkpoint, tmp_path / "on", True, monkeypatch)
  assert on == [last, last + 1]


##
# The real env (slow).
##

UNICYCLE_LIBRARY = os.environ.get("MJLAB_OSL_UNICYCLE_LIBRARY", "")


@pytest.mark.slow
@pytest.mark.skipif(
  not UNICYCLE_LIBRARY, reason="set MJLAB_OSL_UNICYCLE_LIBRARY to a library dir"
)
def test_split_env_with_the_penalty() -> None:
  """The stage-2 env on the real library with the penalty on: the term's half
  strides are the clips' frame counts halved (rounded half up), one env step
  advances an env's reference one frame (outside resets and switches), and the
  penalty is finite, within [0, 1) and 0 wherever d does not count."""
  cfg = osl_split_env_cfg("robot", "compliant", "clf", "train")
  cfg.gait_symmetry = GaitSymmetryCfg(weight=PENALTY)
  sync_gait_symmetry(cfg)
  cfg.scene.num_envs = 8
  motion = cfg.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  motion.library_dir = UNICYCLE_LIBRARY
  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  cmd = env.command_manager.get_term("motion")
  assert isinstance(cmd, GaitLibraryCommand)
  term = env.reward_manager.get_term_cfg(SYMMETRY_METRIC).func
  assert isinstance(term, gait_symmetry)
  frames = cmd.num_frames
  assert (
    term.half_periods.tolist()
    == ((frames + 1) // 2).clamp(max=term.history_length - 1).tolist()
  )
  env.reset()
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
  for _ in range(40):
    frame, clip = cmd.frame.clone(), cmd.clip.clone()
    _, _, terminated, truncated, _ = env.step(zero)
    kept = ~(terminated | truncated).bool() & (cmd.clip == clip)
    advanced = (frame + 1) % frames[clip]
    assert bool((cmd.frame[kept] == advanced[kept]).all())
    value = term.components["gait_symmetry_penalty"]
    assert bool(torch.isfinite(value).all())
    assert bool(((value >= 0.0) & (value < 1.0)).all())
    assert bool((value[~term.valid] == 0.0).all())
  env.close()
