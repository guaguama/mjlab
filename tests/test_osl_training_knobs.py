"""Stage-1 tuning knobs of the musclemimic-style PPO stack (CPU, fake envs).

The action-std cap and floor, the resumable linear rate schedule, bfloat16
Newton-Schulz, the per-term policy statistics, the noise-averaged validation action,
the stress-validation bookkeeping and its env config.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from mjlab.envs.mdp.actions import TendonEffortAction
from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg
from mjlab.tasks.osl_gait_library.config.rl_cfg import osl_gait_library_runner_cfg
from mjlab.tasks.osl_mimic.config.env_cfgs import stress_env_cfg
from mjlab.tasks.osl_mimic.config.rl_cfg import INIT_STD, MuscleMimicPpoCfg
from mjlab.tasks.osl_mimic.rl.expected import (
  action_bounds,
  action_term_slices,
  expected_clipped_gaussian,
)
from mjlab.tasks.osl_mimic.rl.models import gaussian_std
from mjlab.tasks.osl_mimic.rl.muon import MuonAdam, orthogonalize
from mjlab.tasks.osl_mimic.rl.ppo import MuscleMimicPPO, geometric_schedule
from mjlab.tasks.osl_mimic.rl.runner import OslMimicOnPolicyRunner, inference_modes

NUM_ENVS = 8
NUM_STEPS = 4
OBS_DIM = 10
MUSCLES = 5
OSL = 2


class _FakeMuscles(TendonEffortAction):
  """A muscle action term without an env: only its type and size matter."""

  def __init__(self, dim: int) -> None:
    self._action_dim = dim


class _FakeActionManager:
  def __init__(self, terms: dict[str, Any]) -> None:
    self._terms = terms

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms)

  def get_term(self, name: str) -> Any:
    return self._terms[name]


def _action_manager() -> Any:
  """The MSK stage-1 order: ``osl_motors`` (2), then ``muscles``."""
  return _FakeActionManager(
    {"osl_motors": SimpleNamespace(action_dim=OSL), "muscles": _FakeMuscles(MUSCLES)}
  )


def _obs(n: int = NUM_ENVS) -> TensorDict:
  return TensorDict(
    {"actor": torch.randn(n, OBS_DIM), "critic": torch.randn(n, OBS_DIM)},
    batch_size=[n],
  )


# The stage-1 recipe's PPO knobs, which _algorithm resets to their neutral defaults.
RECIPE_KNOBS = (
  "init_std_terms",
  "std_cap_start",
  "std_cap_end",
  "std_cap_begin_update",
  "std_cap_end_update",
  "lr_schedule_updates",
  "lr_final_ratio",
  "muon_ns_dtype",
)


def _algorithm(**knobs: Any) -> MuscleMimicPPO:
  """A real MuscleMimicPPO from the unicycle MSK runner config, shrunk, with the
  recipe's knobs at MuscleMimicPpoCfg's defaults (and musclemimic's muscle init std
  3.0), so each test sets the ones it checks."""
  cfg = asdict(
    osl_gait_library_runner_cfg("msk", "compliant", "clf", "test", "unicycle")
  )
  neutral = asdict(MuscleMimicPpoCfg())
  cfg["algorithm"].update({key: neutral[key] for key in RECIPE_KNOBS})
  cfg["actor"]["distribution_cfg"]["init_std"] = INIT_STD["msk"]
  for key in ("actor", "critic"):
    cfg[key]["hidden_dims"] = (16, 16)
    # As MjlabOnPolicyRunner strips them before building the models.
    for opt in ("cnn_cfg", "distribution_cfg"):
      if cfg[key].get(opt) is None:
        cfg[key].pop(opt, None)
    if cfg[key].get("rnn_type") is None:
      for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        cfg[key].pop(opt, None)
  cfg["num_steps_per_env"] = NUM_STEPS
  cfg["algorithm"].update(num_mini_batches=2, **knobs)
  cfg["multi_gpu"] = None
  env = SimpleNamespace(
    num_envs=NUM_ENVS,
    num_actions=OSL + MUSCLES,
    unwrapped=SimpleNamespace(action_manager=_action_manager()),
  )
  alg = MuscleMimicPPO.construct_algorithm(_obs(), cast(Any, env), cfg, "cpu")
  assert isinstance(alg, MuscleMimicPPO)
  return alg


def _rollout_and_update(alg: MuscleMimicPPO) -> dict[str, float]:
  alg.train_mode()
  obs = _obs()
  with torch.inference_mode():
    for _ in range(NUM_STEPS):
      alg.act(obs)
      obs = _obs()
      alg.process_env_step(obs, torch.rand(NUM_ENVS), torch.zeros(NUM_ENVS), {})
    alg.compute_returns(obs)
  return alg.update()


##
# Schedules.
##


def test_geometric_schedule() -> None:
  args = (3.0, 0.15, 300, 2100)
  assert geometric_schedule(0, *args) == 3.0
  assert geometric_schedule(300, *args) == 3.0
  assert geometric_schedule(1200, *args) == pytest.approx(math.sqrt(3.0 * 0.15))
  assert geometric_schedule(2100, *args) == pytest.approx(0.15)
  assert geometric_schedule(9000, *args) == pytest.approx(0.15)
  # finish <= begin: a step right after begin.
  assert geometric_schedule(300, 3.0, 0.15, 300, 300) == 3.0
  assert geometric_schedule(301, 3.0, 0.15, 300, 300) == 0.15


def test_std_cap_projects_learnable_std_and_resumes() -> None:
  alg = _algorithm(
    std_cap_start=2.0, std_cap_end=0.5, std_cap_begin_update=0, std_cap_end_update=2
  )
  # init_std 3.0 is already under update 0's cap (2.0) after construction.
  assert torch.allclose(gaussian_std(alg.actor), torch.full((OSL + MUSCLES,), 2.0))
  _rollout_and_update(alg)
  assert alg.num_updates_done == 1
  # After update 0, the next rollout already samples under update 1's cap (1.0), so
  # update 1's steps (which hold that cap) score it with the same std.
  std = gaussian_std(alg.actor).detach()
  assert alg.std_cap() == pytest.approx(1.0)
  assert float(std.max()) == pytest.approx(1.0)
  # One dim lowered by hand stays learnable below the cap.
  with torch.no_grad():
    alg.actor.distribution.log_std_param[0] = math.log(0.1)  # type: ignore[union-attr]
  _rollout_and_update(alg)
  std = gaussian_std(alg.actor).detach()
  assert alg.policy_stats["std_cap"] == pytest.approx(1.0)  # Update 1's.
  assert float(std.max()) <= 0.5 + 1e-6 and float(std[0]) < 0.2  # Under update 2's.
  # A resumed run (update count restored by load) is projected under its cap before
  # its first rollout, and keeps this run's Muon settings, not the checkpoint's.
  state = alg.save()
  fresh = _algorithm(
    std_cap_start=2.0, std_cap_end=0.5, std_cap_begin_update=0, std_cap_end_update=2
  )
  state["muscle_mimic_state"]["num_updates_done"] = 5
  state["actor_state_dict"]["distribution.log_std_param"].fill_(math.log(3.0))
  for group in state["optimizer_state_dict"]["param_groups"]:
    group["beta"] = 0.5
  fresh.load(state, None, True)
  assert torch.allclose(gaussian_std(fresh.actor), torch.full((OSL + MUSCLES,), 0.5))
  assert all(group["beta"] == 0.95 for group in fresh.optimizer.param_groups)


def test_std_floor_and_invalid_settings() -> None:
  alg = _algorithm(std_floor=0.5)
  with torch.no_grad():
    alg.actor.distribution.log_std_param.fill_(math.log(0.01))  # type: ignore[union-attr]
  alg.apply_std_bounds()
  assert torch.allclose(gaussian_std(alg.actor), torch.full((OSL + MUSCLES,), 0.5))
  with pytest.raises(ValueError, match="exceeds the final std cap"):
    _algorithm(std_cap_start=1.0, std_cap_end=0.2, std_floor=0.3)
  with pytest.raises(ValueError, match="needs std_cap_start"):
    _algorithm(std_cap_end=0.2)
  with pytest.raises(ValueError, match="lr_final_ratio"):
    _algorithm(lr_final_ratio=1.5)


def test_policy_stats_per_action_term() -> None:
  alg = _algorithm()
  assert alg.action_slices == {"osl_motors": slice(0, 2), "muscles": slice(2, 7)}
  mean = torch.zeros(4, OSL + MUSCLES)
  mean[:, 2] = -3.0  # A muscle commanded far below its [0, 1] range.
  alg._record_policy_stats(mean)
  stats = alg.policy_stats
  assert stats["std/muscles"] == pytest.approx(3.0)
  assert "std_cap" not in stats
  # Muscles at mean 0 with std 3 get E[clip] = 3 psi(0) - 3 psi(-1/3) > 0; the
  # mean -3 one less, but not 0. The OSL motors at 0 are symmetric: no gap.
  assert stats["noise_gap/osl_motors"] == pytest.approx(0.0, abs=1e-6)
  assert stats["noise_gap/muscles"] > 0.2
  assert stats["mean_outside/muscles"] == pytest.approx(1 / MUSCLES)
  assert stats["mean_outside/osl_motors"] == 0.0


##
# Optimizer.
##


def test_bfloat16_newton_schulz_matches_float32() -> None:
  g = torch.randn(64, 32)
  exact = orthogonalize(g)
  fast = orthogonalize(g, dtype=torch.bfloat16)
  assert fast.dtype == torch.float32 and fast.shape == g.shape
  cosine = (exact * fast).sum() / (exact.norm() * fast.norm())
  assert float(cosine) > 0.99
  param = nn.Parameter(torch.randn(8, 4))
  param.grad = torch.randn(8, 4)
  before = param.detach().clone()
  MuonAdam([param], lr=1e-2, ns_dtype="bfloat16").step()
  assert not torch.equal(param.detach(), before)
  with pytest.raises(ValueError, match="ns_dtype"):
    MuonAdam([param], lr=1e-2, ns_dtype="float16")


##
# Validation helpers.
##


def test_action_bounds_and_slices() -> None:
  manager = _action_manager()
  lo, hi = action_bounds(manager)
  assert lo.tolist() == [-1.0] * OSL + [0.0] * MUSCLES
  assert hi.tolist() == [1.0] * (OSL + MUSCLES)
  assert action_term_slices(manager) == {
    "osl_motors": slice(0, 2),
    "muscles": slice(2, 7),
  }
  assert inference_modes(None, ("mean",)) == ("mean",)
  assert inference_modes("expected", ("mean",)) == ("expected",)
  with pytest.raises(ValueError, match="Unknown inference"):
    inference_modes(("mean", "median"), ("mean",))


class _FakePolicy(nn.Module):
  """Mean = the first action_dim observation columns; a fixed log std."""

  def __init__(self, dim: int, std: float) -> None:
    super().__init__()
    self.obs_normalizer = nn.Linear(1, 1)  # Stands in for the stats snapshot.
    self.distribution = SimpleNamespace()
    self.dim = dim
    self.std = std
    self.modes: list[str] = []

  def set_norm_mode(self, mode: str) -> None:
    self.modes.append(mode)

  def forward(self, obs: TensorDict, stochastic_output: bool = False) -> torch.Tensor:
    actor = obs["actor"]
    assert isinstance(actor, torch.Tensor)
    return actor[:, : self.dim]


def _fake_runner(policy: _FakePolicy, cfg: dict[str, Any]) -> OslMimicOnPolicyRunner:
  runner = OslMimicOnPolicyRunner.__new__(OslMimicOnPolicyRunner)
  runner.alg = cast(Any, SimpleNamespace(get_policy=lambda: policy))
  runner.device = "cpu"
  runner.validation_cfg = cfg
  return runner


def test_expected_inference_is_the_noise_averaged_action(monkeypatch) -> None:
  policy = _FakePolicy(OSL + MUSCLES, std=2.0)
  runner = _fake_runner(policy, {})
  monkeypatch.setattr(
    "mjlab.tasks.osl_mimic.rl.runner.gaussian_std",
    lambda model: torch.full((OSL + MUSCLES,), model.std),
  )
  env = SimpleNamespace(action_manager=_action_manager())
  obs = TensorDict({"actor": torch.zeros(3, OBS_DIM)}, batch_size=[3])
  act = runner._inference_policy("expected", cast(Any, env))(obs)
  lo, hi = action_bounds(env.action_manager)
  expected = expected_clipped_gaussian(torch.zeros(3, OSL + MUSCLES), 2.0, lo, hi)
  torch.testing.assert_close(act, expected)
  assert float(act[0, 0]) == pytest.approx(0.0, abs=1e-6)  # Symmetric OSL motor.
  assert float(act[0, 2]) > 0.3  # Muscle at mean 0: noise-driven excitation.
  mean = runner._inference_policy("mean", cast(Any, env))(obs)
  assert torch.equal(mean, torch.zeros(3, OSL + MUSCLES))


def test_stress_pass_counts_each_env_first_episode() -> None:
  """Env 1 terminates at step 2 and env 2 is truncated at step 3 (both then
  auto-reset and keep running); envs 0 and 3 survive the whole window."""
  policy = _FakePolicy(OSL + MUSCLES, std=1.0)
  runner = _fake_runner(policy, {"stress_num_steps": 5})
  n = 4
  step_count = 0
  components = {"err_site_abs": torch.zeros(n)}

  def reset() -> tuple[TensorDict, dict]:
    return TensorDict({"actor": torch.zeros(n, OBS_DIM)}, batch_size=[n]), {}

  def step(actions: torch.Tensor):
    nonlocal step_count
    step_count += 1
    dones = torch.zeros(n)
    time_outs = torch.zeros(n)
    if step_count == 2:
      dones[1] = 1.0
    if step_count == 3:
      dones[2] = 1.0
      time_outs[2] = 1.0
    if step_count == 4:
      # Second episodes failing count for nothing: env 1 failed already, and env 2
      # survived its first episode (truncated at step 3).
      dones[1] = 1.0
      dones[2] = 1.0
    # Each env's error: env i reports i at every step.
    components["err_site_abs"] = torch.arange(n, dtype=torch.float32)
    obs = TensorDict({"actor": torch.zeros(n, OBS_DIM)}, batch_size=[n])
    return obs, torch.zeros(n), dones, {"time_outs": time_outs}

  env = SimpleNamespace(
    num_envs=n, device="cpu", step_dt=0.01, action_manager=_action_manager()
  )
  vec = SimpleNamespace(unwrapped=env, device="cpu", reset=reset, step=step)
  term = SimpleNamespace(components=components)
  setattr(runner, "_diagnostic_terms", lambda env: [term])  # noqa: B010
  metrics = runner._stress_pass(cast(Any, vec), "mean")
  assert metrics["survival_rate"] == pytest.approx(3 / 4)
  # First-episode lengths 5, 2, 3, 5 steps.
  assert metrics["mean_first_episode_s"] == pytest.approx(15 / 4 * 0.01)
  # Steps 1-2 average all four envs (1.5), step 3 envs {0, 2, 3} (5/3), steps 4-5
  # envs {0, 3} (1.5).
  assert metrics["err_site_abs"] == pytest.approx((1.5 * 2 + 5 / 3 + 1.5 * 2) / 5)
  assert policy.modes == ["update", "update"]


def test_stress_env_cfg_is_the_training_distribution_with_the_site_check() -> None:
  cfg = stress_env_cfg(
    osl_gait_library_env_cfg("msk", "compliant", "clf", "train", "unicycle"),
    num_steps=1000,
    num_envs=512,
  )
  assert set(cfg.terminations) == {"time_out", "site_deviation"}
  assert cfg.terminations["site_deviation"].params["threshold"] == 0.5
  assert cfg.episode_length_s == pytest.approx(10.0)
  assert cfg.scene.num_envs == 512
  # Disturbances and random draws stay: pushes, physical draws, random delays.
  assert {"push_robot", "foot_friction", "muscle_strength"} <= set(cfg.events)
  actuators = cfg.scene.entities["robot"].articulation.actuators  # type: ignore[union-attr]
  assert {(a.delay_min_lag, a.delay_max_lag) for a in actuators} == {(0, 10), (10, 20)}
  motion = cfg.commands["motion"]
  assert motion.resampling_time_range == (3.0, 8.0)
  # The training config itself is untouched by building another.
  train = osl_gait_library_env_cfg("msk", "compliant", "clf", "train", "unicycle")
  assert train.terminations["site_deviation"].params["threshold"] == 1.0


def test_video_inference_auto_picks_expected_for_muscles() -> None:
  runner = OslMimicOnPolicyRunner.__new__(OslMimicOnPolicyRunner)
  runner.video_cfg = {"inference": "auto"}
  muscles = SimpleNamespace(action_manager=_action_manager())
  servos = SimpleNamespace(
    action_manager=_FakeActionManager({"servos": SimpleNamespace(action_dim=3)})
  )
  assert runner._video_inference(cast(Any, muscles)) == "expected"
  assert runner._video_inference(cast(Any, servos)) == "mean"
  runner.video_cfg = {"inference": "mean"}
  assert runner._video_inference(cast(Any, muscles)) == "mean"


def test_init_std_per_action_term() -> None:
  alg = _algorithm(init_std_terms={"osl_motors": 0.3})
  std = gaussian_std(alg.actor).detach()
  torch.testing.assert_close(std[:OSL], torch.full((OSL,), 0.3))
  torch.testing.assert_close(std[OSL:], torch.full((MUSCLES,), 3.0))
  # Under a cap the per-term std stays where it is below it.
  alg = _algorithm(init_std_terms={"osl_motors": 0.3}, std_cap_start=1.0)
  std = gaussian_std(alg.actor).detach()
  torch.testing.assert_close(std, torch.tensor([0.3] * OSL + [1.0] * MUSCLES))
  with pytest.raises(ValueError, match="unknown action terms"):
    _algorithm(init_std_terms={"servos": 0.3})


def test_eval_envs_inherit_the_runs_overrides(capsys) -> None:
  """Stress validation plays the training config itself; validation and video keep
  their own mode but take the run's simulation and reward settings."""
  from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
  from mjlab.tasks.osl_gait_library.rl.runner import GaitLibraryOnPolicyRunner

  train = osl_gait_library_env_cfg("msk", "compliant", "clf", "train", "unicycle")
  train.sim.mujoco.iterations = 12
  train.rewards["mimic"].params["clf"].pen_w = 0.0
  train.rewards["twist"].weight = 0.1
  train.commands["motion"].library_dir = "/some/library"  # type: ignore[attr-defined]
  train.events["push_robot"].params["velocity_range"]["x"] = (-0.5, 0.5)
  train.observations["actor"].terms["touch"].delay_max_lag = 2
  train.observations["actor"].enable_corruption = True
  runner = GaitLibraryOnPolicyRunner.__new__(GaitLibraryOnPolicyRunner)
  runner.env = cast(Any, SimpleNamespace(unwrapped=SimpleNamespace(cfg=train)))
  runner.validation_cfg = {
    "model": "msk",
    "socket": "compliant",
    "reward": "clf",
    "variant": "unicycle",
  }

  stress = runner._eval_env_cfg("stress")
  assert stress is not train
  assert stress.events["push_robot"].params["velocity_range"]["x"] == (-0.5, 0.5)
  assert stress.sim.mujoco.iterations == 12
  stress.sim.mujoco.iterations = 99  # A copy: the training config is untouched.
  assert train.sim.mujoco.iterations == 12

  for purpose in ("validation", "video"):
    cfg = runner._eval_env_cfg(purpose)
    motion = cfg.commands["motion"]
    assert isinstance(motion, GaitLibraryCommandCfg)
    assert motion.library_dir == "/some/library"
    assert motion.fixed_clips == (purpose == "validation")
    assert cfg.sim.mujoco.iterations == 12
    assert cfg.rewards["mimic"].params["clf"].pen_w == 0.0
    assert cfg.rewards["twist"].weight == 0.1
    assert "push_robot" not in cfg.events  # Validation and video run nominal plants.
    # The policy's input as the run built it, without its noise.
    actor = cfg.observations["actor"]
    assert actor.terms["touch"].delay_max_lag == 2 and not actor.enable_corruption

  def broken(it: int) -> None:
    raise RuntimeError("renderer lost")

  runner._guarded(broken, 7)  # Reported, never fatal.
  assert "[WARN] broken at it=7 failed" in capsys.readouterr().out


##
# Env stages.
##


def test_env_stages_hold_until_their_update_and_survive_a_resume() -> None:
  twist = SimpleNamespace(weight=0.2)
  root = SimpleNamespace(params={"command_name": "motion", "threshold": 2.0})
  env = SimpleNamespace(
    reward_manager=SimpleNamespace(get_term_cfg=lambda n: {"twist": twist}[n]),
    termination_manager=SimpleNamespace(get_term_cfg=lambda n: {"root": root}[n]),
  )
  stages = (
    (1500, "terminations.root.threshold", 1.0),
    (3000, "rewards.twist.weight", 0.06),
    (500, "terminations.root.threshold", 0.5),  # Overlaps; ends first, so holds.
  )

  def runner(update: int) -> Any:
    return SimpleNamespace(
      env=SimpleNamespace(unwrapped=env),
      alg=SimpleNamespace(num_updates_done=update),
      env_stages=stages,
      _stage_defaults={},
    )

  apply = OslMimicOnPolicyRunner.apply_env_stages
  r = runner(0)
  assert apply(r) == {"terminations.root.threshold": 0.5, "rewards.twist.weight": 0.06}
  assert root.params["threshold"] == 0.5 and twist.weight == 0.06
  r.alg.num_updates_done = 500
  apply(r)
  assert root.params["threshold"] == 1.0 and twist.weight == 0.06
  r.alg.num_updates_done = 3000
  apply(r)  # Every stage over: the config's own values return.
  assert root.params["threshold"] == 2.0 and twist.weight == 0.2
  # A run resumed at update 2000 starts from the config's values, in the right stage.
  assert apply(runner(2000)) == {
    "terminations.root.threshold": 2.0,
    "rewards.twist.weight": 0.06,
  }
  bad = runner(0)
  bad.env_stages = ((9, "rewards.twist", 1.0),)
  with pytest.raises(ValueError, match="Unsupported env stage path"):
    apply(bad)
  bad.env_stages = ((9, "terminations.root.x", 1.0),)
  with pytest.raises(KeyError, match="no param"):
    apply(bad)
