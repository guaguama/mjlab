"""Contracts for stage 2's two-agent PPO (``tasks/osl_split/rl``).

CPU only: tiny musclemimic MLPs on fake observations and a fake env that exposes
just an action manager. The schedules, GAE and the minibatches are checked against
independent references, the noise-averaged action against Monte Carlo and its
limits, and the algorithm on both models' action layouts: the scatter, the
bit-exact freeze, per-module optimizer isolation, the OSL std floor, the blend,
checkpoints (whole, and actors only for play) and the warm start (with and without
a reward switch).
"""

from __future__ import annotations

import copy
import io
import math
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model
from mjlab.envs.mdp.actions import TendonEffortAction
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel, RunningMeanStd
from mjlab.tasks.osl_mimic.rl.ppo import MuscleMimicPPO
from mjlab.tasks.osl_split.config.rl_cfg import (
  OSL_MIN_STD,
  SplitPpoCfg,
  osl_split_runner_cfg,
)
from mjlab.tasks.osl_split.rl.expected import action_bounds, expected_clipped_gaussian
from mjlab.tasks.osl_split.rl.layout import ActionLayout
from mjlab.tasks.osl_split.rl.policy import (
  distribution_std,
  gaussian_distribution,
  std_parameter,
)
from mjlab.tasks.osl_split.rl.ppo import STAGE1_ONLY_KNOBS, TwoAgentPPO
from mjlab.tasks.osl_split.rl.schedules import MODULES, ChannelSchedule, SplitSchedule
from mjlab.tasks.osl_split.rl.storage import TwoAgentRolloutStorage, obs_group

NUM_ENVS = 6
NUM_STEPS = 4
ACTOR_DIM = 12
OSL_OBS_DIM = 6
HUMAN_DIM = 5
HIDDEN = (16, 16)

##
# Fakes and helpers.
##


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


def _action_manager(model: Model) -> Any:
  """The stage-1 term order: MSK ``osl_motors, muscles``; robot ``servos,
  osl_motors``."""
  if model == "msk":
    terms = {
      "osl_motors": SimpleNamespace(action_dim=2),
      "muscles": _FakeMuscles(HUMAN_DIM),
    }
  else:
    terms = {
      "servos": SimpleNamespace(action_dim=HUMAN_DIM),
      "osl_motors": SimpleNamespace(action_dim=2),
    }
  return _FakeActionManager(terms)


def _env(model: Model = "msk") -> SimpleNamespace:
  return SimpleNamespace(
    num_envs=NUM_ENVS,
    num_actions=HUMAN_DIM + 2,
    unwrapped=SimpleNamespace(action_manager=_action_manager(model)),
  )


def _obs(n: int = NUM_ENVS) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(n, ACTOR_DIM),
      "osl": 3.0 * torch.randn(n, OSL_OBS_DIM) + 1.0,
    },
    batch_size=[n],
  )


def _train_cfg(model: Model = "msk", **algorithm: Any) -> dict[str, Any]:
  """The real runner config, shrunk."""
  cfg = asdict(osl_split_runner_cfg(model, "compliant", "clf", "test"))
  for key in ("actor", "critic", "osl_critic"):
    cfg[key]["hidden_dims"] = HIDDEN
  cfg["osl_actor"]["hidden_dims"] = (8, 8)
  cfg["num_steps_per_env"] = NUM_STEPS
  cfg["algorithm"]["num_mini_batches"] = 2
  cfg["algorithm"].update(algorithm)
  cfg["multi_gpu"] = None
  return cfg


def _alg(model: Model = "msk", **algorithm: Any) -> TwoAgentPPO:
  return TwoAgentPPO.construct_algorithm(
    _obs(), _env(model), _train_cfg(model, **algorithm), "cpu"
  )


def _schedule(
  human_freeze: int = 10,
  critic_warmup: int = 2,
  blend: bool = False,
  blend_ramp: int = 4,
) -> SplitSchedule:
  return SplitSchedule(
    total_iterations=100,
    human_freeze=human_freeze,
    critic_warmup=critic_warmup,
    human_critic_lr=1e-4,
    osl_critic_lr=3e-4,
    critic_ramp=critic_warmup,
    osl_actor_lr=4e-4,
    human_peak_lr=3e-5,
    human_ramp=5,
    lr_final_ratio=0.1,
    human_lr_final_ratio=1.0,
    blend=blend,
    blend_ramp=blend_ramp,
  )


def _step(alg: TwoAgentPPO) -> torch.Tensor:
  """One env step with random rewards, dones and time-outs, as the runner does it
  (inside inference mode)."""
  with torch.inference_mode():
    obs = _obs()
    actions = alg.act(obs)
    alg.process_env_step(
      obs,
      torch.rand(NUM_ENVS, 2),
      (torch.rand(NUM_ENVS) < 0.25).long(),
      {"time_outs": torch.rand(NUM_ENVS) < 0.1},
    )
  return actions


def _rollout(alg: TwoAgentPPO) -> None:
  while not alg.storage.is_full:
    _step(alg)
  with torch.inference_mode():
    alg.compute_returns(_obs())


def _output(model: MuscleMimicMLPModel) -> nn.Linear:
  layer = model.mlp[-1]
  assert isinstance(layer, nn.Linear)
  return layer


def _normalizer(model: MuscleMimicMLPModel) -> RunningMeanStd:
  normalizer = model.obs_normalizer
  assert isinstance(normalizer, RunningMeanStd)
  return normalizer


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
  return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _assert_same(a: Any, b: Any) -> None:
  """Bit-identical nested state (tensors, dicts, lists, scalars)."""
  if isinstance(a, torch.Tensor):
    assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
  elif isinstance(a, dict):
    assert isinstance(b, dict) and a.keys() == b.keys()
    for key in a:
      _assert_same(a[key], b[key])
  elif isinstance(a, (list, tuple)):
    assert isinstance(b, (list, tuple)) and len(a) == len(b)
    for x, y in zip(a, b, strict=True):
      _assert_same(x, y)
  else:
    assert a == b


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
  return any(not torch.equal(before[k], after[k]) for k in before)


def _stage1(
  alg: TwoAgentPPO,
) -> tuple[SimpleNamespace, MuscleMimicMLPModel, MuscleMimicMLPModel]:
  """A synthetic warm start in the converter's format, with the unified actor and
  the critic it was cut from (random weights and observation statistics)."""
  obs = _obs()
  layout = alg.layout
  unified = MuscleMimicMLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    layout.total_dim,
    hidden_dims=HIDDEN,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 3.0,
      "std_type": "log",
    },
  )
  critic = MuscleMimicMLPModel(
    obs, {"critic": ["actor"]}, "critic", 1, hidden_dims=HIDDEN
  )
  with torch.no_grad():
    for model in (unified, critic):
      for p in model.parameters():
        p.normal_(0.0, 0.3)
      norm = _normalizer(model)
      norm.mean.normal_()
      norm.var.uniform_(0.5, 2.0)
      norm.count.fill_(2.0e8)
  unified.set_norm_mode("frozen")
  critic.set_norm_mode("frozen")
  last = len(unified.mlp) - 1
  sliced = {f"mlp.{last}.weight", f"mlp.{last}.bias", "distribution.log_std_param"}
  human = layout.index("human")
  osl = layout.index("osl")
  state = _state(unified)
  ws = SimpleNamespace(
    human_actor={
      k: v.index_select(0, human) if k in sliced else v for k, v in state.items()
    },
    human_critic=_state(critic),
    teacher_head={
      "weight": state[f"mlp.{last}.weight"].index_select(0, osl),
      "bias": state[f"mlp.{last}.bias"].index_select(0, osl),
      "log_std": state["distribution.log_std_param"].index_select(0, osl),
    },
    return_stats={
      "mean": torch.tensor(0.3),
      "var": torch.tensor(2.5),
      "count": torch.tensor(1.0e9),
    },
    lineage={},
  )
  return ws, unified, critic


def _warm_start(alg: TwoAgentPPO, ws: SimpleNamespace, reward_switch: bool) -> None:
  alg.apply_warm_start(
    cast(Any, ws),
    reward_switch=reward_switch,
    num_envs=NUM_ENVS,
    num_steps=NUM_STEPS,
  )


##
# Schedules.
##


def test_split_schedule_freeze_and_rates() -> None:
  s = SplitSchedule(
    total_iterations=1000,
    human_freeze=200,
    critic_warmup=10,
    human_critic_lr=1e-4,
    osl_critic_lr=3e-4,
    critic_ramp=10,
    osl_actor_lr=4e-4,
    human_peak_lr=3e-5,
    human_ramp=50,
    lr_final_ratio=0.1,
    human_lr_final_ratio=1.0,
    blend=False,
    blend_ramp=100,
  )
  assert s.osl_actor_frozen(9) and not s.osl_actor_frozen(10)
  assert s.human_actor_frozen(199) and not s.human_actor_frozen(200)
  # Critics: a ramp over the warm-up, then the decay to 10% at the stage end, held.
  assert s.lr("human_critic", 0) == pytest.approx(1e-4 * 0.1 * (1.0 - 0.9 * 0.0))
  assert s.lr("osl_critic", 4) == pytest.approx(3e-4 * 0.5 * (1.0 - 0.9 * 0.004))
  assert s.lr("osl_critic", 9) == pytest.approx(3e-4 * (1.0 - 0.9 * 0.009))
  assert s.lr("osl_critic", 500) == pytest.approx(3e-4 * 0.55)
  assert s.lr("human_critic", 500) == pytest.approx(1e-4 * 0.55)  # Above its floor.
  assert s.lr("human_critic", 900) == pytest.approx(3e-5)  # 1e-4 * 0.19, floored.
  for k in (1000, 1500, 10_000):  # The floor, beyond the schedule too.
    # The human critic's floor is the human actor's peak (3e-5), not 10% (1e-5).
    assert s.lr("human_critic", k) == pytest.approx(3e-5)
    assert s.lr("osl_critic", k) == pytest.approx(3e-5)
    assert s.lr("osl_actor", k) == pytest.approx(4e-5)
  # The OSL actor: 0 in the warm-up, then the critics' decay, not a restart.
  assert s.lr("osl_actor", 9) == 0.0
  assert s.lr("osl_actor", 10) == pytest.approx(4e-4 * (1.0 - 0.9 * 0.01))
  assert s.lr("osl_actor", 500) == pytest.approx(4e-4 * 0.55)
  # The human: 0 while frozen, a ramp from the unfreeze, then its peak held.
  assert s.lr("human_actor", 199) == 0.0
  assert s.lr("human_actor", 200) == pytest.approx(3e-5 / 50)
  assert s.lr("human_actor", 249) == pytest.approx(3e-5)
  assert s.lr("human_actor", 600) == s.lr("human_actor", 5000) == pytest.approx(3e-5)
  assert max(s.lr("human_actor", k) for k in range(1001)) <= 3e-5
  # Below 1 the human decays from the unfreeze to its own floor at the stage end.
  decaying = replace(s, human_lr_final_ratio=0.5)
  assert decaying.lr("human_actor", 600) == pytest.approx(3e-5 * 0.75)
  assert decaying.lr("human_actor", 2000) == pytest.approx(1.5e-5)
  assert s.lrs(300) == {m: s.lr(m, 300) for m in MODULES}
  assert s.beta(0) == 1.0  # No blend.
  # Without a ramp the critics start at full rate.
  assert replace(s, critic_ramp=0).lr("human_critic", 0) == pytest.approx(1e-4)
  with pytest.raises(KeyError):
    s.lr("teacher_head", 0)
  with pytest.raises(ValueError, match="lr_final_ratio"):
    replace(s, lr_final_ratio=1.5)
  with pytest.raises(ValueError, match="human_lr_final_ratio"):
    replace(s, human_lr_final_ratio=-0.1)


def test_split_schedule_from_saved_replays_old_checkpoints() -> None:
  """A checkpoint from before the separate critic rates, the critic ramp and the
  floors kept one critic rate and decayed every rate to 0: it replays that way."""
  old = {
    "total_iterations": 100,
    "human_freeze": 20,
    "critic_warmup": 5,
    "critic_lr": 4e-4,
    "osl_actor_lr": 4e-4,
    "human_peak_lr": 1e-4,
    "human_ramp": 10,
    "blend": False,
    "blend_ramp": 0,
  }
  s = SplitSchedule.from_saved(old)
  assert (s.human_critic_lr, s.osl_critic_lr, s.critic_ramp) == (4e-4, 4e-4, 0)
  assert (s.lr_final_ratio, s.human_lr_final_ratio) == (0.0, 0.0)
  assert s.lr("human_critic", 0) == pytest.approx(4e-4)
  assert s.lr("osl_critic", 100) == 0.0 and s.lr("osl_actor", 100) == 0.0
  assert s.lr("human_actor", 100) == 0.0
  assert s.lr("human_actor", 29) == pytest.approx(1e-4 * (1.0 - 9 / 80))
  new = SplitSchedule.from_saved(asdict(_schedule()))  # A new one round-trips.
  assert new == _schedule()


def test_split_schedule_from_train_cfg_and_blend() -> None:
  cfg = asdict(osl_split_runner_cfg("msk", "compliant", "clf", "test"))
  s = SplitSchedule.from_train_cfg(cfg, total_iterations=1000, reward_switch=False)
  assert (s.human_freeze, s.critic_warmup, s.human_ramp) == (200, 10, 50)
  assert (s.human_critic_lr, s.osl_critic_lr) == (1e-4, 3e-4)
  assert (s.osl_actor_lr, s.human_peak_lr) == (4e-4, 3e-5)
  assert s.critic_ramp == s.critic_warmup  # The critics ramp over the warm-up.
  assert (s.lr_final_ratio, s.human_lr_final_ratio) == (0.1, 1.0)
  assert s.total_iterations == 1000
  # The blend is on by default: warm-up 10 + ramp 100 <= freeze 200.
  assert s.blend and s.blend_ramp == 100
  assert [s.beta(k) for k in (0, 10, 60, 110, 500)] == [0.0, 0.0, 0.5, 1.0, 1.0]
  # A reward switch starts the critics fresh: the longer warm-up and ramp; so does
  # a fresh OSL critic.
  switched = SplitSchedule.from_train_cfg(cfg, 1000, reward_switch=True)
  assert switched.critic_warmup == switched.critic_ramp == 50
  assert switched.osl_actor_frozen(49) and switched.beta(100) == 0.5
  fresh_cfg = copy.deepcopy(cfg)
  fresh_cfg["algorithm"]["osl_critic_init"] = "fresh"
  fresh = SplitSchedule.from_train_cfg(fresh_cfg, 1000, reward_switch=False)
  assert fresh == switched

  cfg["freeze"]["human_actor_iterations"] = 100  # beta would be < 1 at the unfreeze.
  with pytest.raises(ValueError, match="human unfreeze"):
    SplitSchedule.from_train_cfg(cfg, 1000, reward_switch=False)
  cfg["blend"]["enabled"] = False
  assert not SplitSchedule.from_train_cfg(cfg, 1000, reward_switch=False).blend
  with pytest.raises(ValueError, match="total_iterations"):
    _ = SplitSchedule.from_train_cfg(cfg, 0, reward_switch=False)


def test_channel_schedule() -> None:
  cfg = asdict(osl_split_runner_cfg("msk", "compliant", "clf", "test"))
  ch = ChannelSchedule.from_train_cfg(cfg)
  # From the critic warm-up's end, done before the human unfreezes.
  assert (ch.anneal_start, ch.anneal_iterations) == (10, 150)
  stage1 = {"human": 1.0, "osl_sites": 1.0, "osl_joints": 0.0}
  assert ch.weights(0) == {"human": stage1, "osl": stage1}
  assert ch.weights(10) == {"human": stage1, "osl": stage1}
  assert ch.weights(85)["osl"] == pytest.approx(
    {"human": 0.75, "osl_sites": 1.0, "osl_joints": 5.0}
  )
  end = {"human": 0.5, "osl_sites": 1.0, "osl_joints": 10.0}
  assert ch.weights(160) == {"human": stage1, "osl": end}
  assert ch.weights(10_000)["osl"] == end
  cfg["channels"]["osl_end"]["human"] = 7.0  # The schedule holds its own copies.
  assert ch.weights(400)["osl"]["human"] == 0.5
  cfg["channels"]["anneal_start_iteration"] = -1  # -1: the human unfreeze.
  assert ChannelSchedule.from_train_cfg(cfg).anneal_start == 200

  with pytest.raises(ValueError, match="no positive weight"):
    ChannelSchedule({"osl": {"a": 0.0}}, {"osl": {"a": 0.0}}, 0, 1)
  with pytest.raises(ValueError, match="groups"):
    ChannelSchedule({"osl": {"a": 1.0}}, {"osl": {"b": 1.0}}, 0, 1)
  jump = ChannelSchedule({"osl": {"a": 1.0}}, {"osl": {"a": 3.0}}, 5, 0)
  assert jump.weights(4)["osl"]["a"] == 1.0 and jump.weights(5)["osl"]["a"] == 3.0


##
# Storage.
##


def test_gae_matches_reference() -> None:
  t_len, n, gamma, lam = 7, 3, 0.9, 0.8
  g = torch.Generator().manual_seed(0)
  obs = TensorDict({"actor": torch.zeros(n, 2)}, batch_size=[n])
  storage = TwoAgentRolloutStorage(n, t_len, obs, ["actor"], {"human": 3, "osl": 2})
  rewards = torch.randn(t_len, n, 2, generator=g)
  values = torch.randn(t_len, n, 2, generator=g)
  dones = torch.rand(t_len, n, generator=g) < 0.3
  last = torch.randn(n, 2, generator=g)
  actions = {"human": torch.zeros(n, 3), "osl": torch.zeros(n, 2)}
  for t in range(t_len):
    storage.add(obs, actions, torch.zeros(n, 2), values[t], rewards[t], dones[t])
  storage.compute_returns(last, gamma, lam)

  # A_t = sum over s >= t of prod_{t <= j < s} (gamma lam (1 - d_j)) delta_s.
  not_done = (~dones).float().unsqueeze(-1)
  next_values = torch.cat((values[1:], last[None]))
  delta = rewards + gamma * not_done * next_values - values
  expected = torch.zeros(t_len, n, 2)
  for t in range(t_len):
    weight = torch.ones(n, 1)
    for s in range(t, t_len):
      expected[t] += weight * delta[s]
      weight = weight * gamma * lam * not_done[s]
  torch.testing.assert_close(storage.advantages, expected)
  torch.testing.assert_close(storage.returns, expected + values)


def test_minibatches_cover_every_sample() -> None:
  t_len, n = 6, 4
  obs = TensorDict({"actor": torch.zeros(n, 3), "osl": torch.zeros(n, 2)}, [n])
  storage = TwoAgentRolloutStorage(
    n, t_len, obs, ["actor", "osl", "actor"], {"human": 3, "osl": 2}
  )
  assert storage.groups == ("actor", "osl")  # Each group is stored once.
  for t in range(t_len):
    ids = torch.arange(n, dtype=torch.float32) + t * n
    column = ids.unsqueeze(-1)
    pair = torch.stack((ids, ids + 0.5), dim=-1)
    step_obs = TensorDict(
      {"actor": column.repeat(1, 3), "osl": -column.repeat(1, 2)}, batch_size=[n]
    )
    acts = {"human": column.repeat(1, 3), "osl": column.repeat(1, 2)}
    storage.add(step_obs, acts, pair, pair, pair, torch.zeros(n))
  storage.compute_returns(torch.zeros(n, 2), 0.99, 0.95)

  seen = []
  for batch in storage.mini_batch_generator(num_mini_batches=3, num_epochs=2):
    ids = obs_group(batch.observations, "actor")[:, 0]
    assert batch.observations.batch_size == torch.Size([8])
    assert torch.equal(obs_group(batch.observations, "osl")[:, 1], -ids)
    assert torch.equal(batch.actions["human"][:, 2], ids)
    assert torch.equal(batch.actions["osl"][:, 0], ids)
    assert torch.equal(batch.values, torch.stack((ids, ids + 0.5), dim=-1))
    assert torch.equal(batch.old_log_probs[:, 1], ids + 0.5)
    seen.append(ids)
  for epoch in torch.cat(seen).reshape(2, -1):
    assert torch.equal(epoch.sort().values, torch.arange(24.0))


##
# The noise-averaged action.
##


@pytest.mark.parametrize(("lo", "hi"), [(0.0, 1.0), (-1.0, 1.0)])
def test_expected_clipped_gaussian_matches_monte_carlo(lo: float, hi: float) -> None:
  g = torch.Generator().manual_seed(0)
  mu = torch.tensor([-3.0, -1.0, 0.0, 0.5, 1.0, 3.0]).repeat_interleave(3)
  sigma = torch.tensor([0.1, 1.0, 2.45]).repeat(6)
  noise = torch.randn(400_000, mu.numel(), generator=g, dtype=torch.float64)
  monte_carlo = (mu.double() + sigma.double() * noise).clamp(lo, hi).mean(0)
  got = expected_clipped_gaussian(mu, sigma, lo, hi)
  assert got.dtype == torch.float32
  torch.testing.assert_close(got.double(), monte_carlo, atol=4e-3, rtol=0.0)


def test_expected_clipped_gaussian_limits() -> None:
  mu = torch.tensor([-5.0, -0.3, 0.0, 0.4, 0.99, 7.0])
  # sigma -> 0: the clipped mean.
  torch.testing.assert_close(
    expected_clipped_gaussian(mu, 0.0, 0.0, 1.0), mu.clamp(0, 1)
  )
  torch.testing.assert_close(
    expected_clipped_gaussian(mu, 1e-9, -1.0, 1.0), mu.clamp(-1, 1)
  )
  # Far outside the range: the bound.
  far = torch.tensor([-1e6, 1e6])
  torch.testing.assert_close(
    expected_clipped_gaussian(far, 2.45, 0.0, 1.0), torch.tensor([0.0, 1.0])
  )
  # sigma -> inf: half the mass at each bound.
  torch.testing.assert_close(
    expected_clipped_gaussian(mu, 1e6, 0.0, 1.0),
    torch.full_like(mu, 0.5),
    atol=1e-5,
    rtol=0.0,
  )
  # Infinite bounds clip nothing (and make no NaN).
  torch.testing.assert_close(
    expected_clipped_gaussian(mu, 2.0, -math.inf, math.inf), mu
  )
  # Per-row tensor bounds; E[clip(Z, 0, 1)] = phi(0) - phi(1) + 1 - Phi(1).
  out = expected_clipped_gaussian(
    torch.zeros(3, 2), torch.ones(2), torch.tensor([0.0, -1.0]), torch.ones(2)
  )
  pdf0, pdf1 = (math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) for z in (0.0, 1.0))
  half = pdf0 - pdf1 + 0.5 * math.erfc(1.0 / math.sqrt(2.0))
  torch.testing.assert_close(out, torch.tensor([[half, 0.0]] * 3))


def test_action_bounds() -> None:
  msk = ActionLayout.for_model("msk", HUMAN_DIM + 2)
  lo, hi = action_bounds(_action_manager("msk"), msk, "human")
  assert lo.tolist() == [0.0] * HUMAN_DIM and hi.tolist() == [1.0] * HUMAN_DIM
  lo, hi = action_bounds(_action_manager("msk"), msk, "osl")
  assert lo.tolist() == [-1.0, -1.0] and hi.tolist() == [1.0, 1.0]
  robot = ActionLayout.for_model("robot", HUMAN_DIM + 2)
  lo, _ = action_bounds(_action_manager("robot"), robot, "human")
  assert lo.tolist() == [-1.0] * HUMAN_DIM
  with pytest.raises(ValueError, match="dims"):
    action_bounds(_action_manager("msk"), ActionLayout.for_model("msk", 9), "human")


def test_agent_terms() -> None:
  """Each agent's action terms in its own columns, in action-manager order."""
  msk = ActionLayout.for_model("msk", HUMAN_DIM + 2)
  assert msk.agent_terms("human") == [("muscles", slice(0, HUMAN_DIM))]
  assert msk.agent_terms("osl") == [("osl_motors", slice(0, 2))]
  mixed = ActionLayout((("servos", 3), ("osl_motors", 2), ("extra", 4)))
  assert mixed.agent_terms("human") == [("servos", slice(0, 3)), ("extra", slice(3, 7))]


##
# The algorithm.
##


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_act_scatters_by_layout(model: Model) -> None:
  torch.manual_seed(0)
  alg = _alg(model)
  alg.set_schedule(_schedule())
  rows_h, rows_o = alg.layout.rows("human"), alg.layout.rows("osl")
  assert rows_o == ([0, 1] if model == "msk" else [HUMAN_DIM, HUMAN_DIM + 1])
  with torch.no_grad():
    _output(alg.human_actor).bias.fill_(3.0)
    _output(alg.osl_actor).bias.fill_(-3.0)
    std_parameter(alg.human_actor).fill_(math.log(1e-3))
  actions = _step(alg)
  assert actions.shape == (NUM_ENVS, HUMAN_DIM + 2)
  # Each agent's own sample, in its own rows (no blend).
  assert torch.equal(actions[:, rows_h], alg.storage.actions["human"][0])
  assert torch.equal(actions[:, rows_o], alg.storage.actions["osl"][0])
  assert (actions[:, rows_h] - 3.0).abs().max() < 0.1
  assert actions[:, rows_o].max() < -1.5  # Mean -3, std 0.2.
  values = alg.storage.values[0]
  assert values.shape == (NUM_ENVS, 2) and not torch.equal(values[:, 0], values[:, 1])


def test_frozen_modules_are_bit_exact() -> None:
  torch.manual_seed(0)
  alg = _alg()
  # Train everything once, so the human's optimizer has moments to keep.
  alg.set_schedule(_schedule(human_freeze=0, critic_warmup=0))
  _rollout(alg)
  assert {"human/surrogate", "osl/surrogate"} <= set(alg.update())
  assert alg.optimizers["human_actor"].state

  # Then the human frozen (k = 1 < 10) and the OSL actor in a critic warm-up.
  alg.set_schedule(_schedule(human_freeze=10, critic_warmup=5))
  alg.set_stage_iteration(1)
  human = _state(alg.human_actor)
  human_opt = copy.deepcopy(alg.optimizers["human_actor"].state_dict())
  osl = _state(alg.osl_actor)
  osl_opt = copy.deepcopy(alg.optimizers["osl_actor"].state_dict())
  critics = {name: _state(alg.models[name]) for name in ("human_critic", "osl_critic")}
  _rollout(alg)
  losses = alg.update()

  assert set(losses) == {"human/value", "osl/value"}
  # Weights, std and observation statistics, and the optimizer's moments.
  _assert_same(_state(alg.human_actor), human)
  _assert_same(alg.optimizers["human_actor"].state_dict(), human_opt)
  # The OSL actor's weights hold; its observation statistics keep updating.
  osl_after = _state(alg.osl_actor)
  for key, value in osl.items():
    if key.startswith("obs_normalizer."):
      assert not torch.equal(osl_after[key], value), key
    else:
      assert torch.equal(osl_after[key], value), key
  _assert_same(alg.optimizers["osl_actor"].state_dict(), osl_opt)
  for name, before in critics.items():
    assert _changed(before, _state(alg.models[name])), name
  assert alg.lrs["human_actor"] == 0.0 and alg.lrs["osl_actor"] == 0.0


def test_non_finite_osl_loss_does_not_block_the_others() -> None:
  torch.manual_seed(0)
  alg = _alg()
  alg.set_schedule(_schedule(human_freeze=0, critic_warmup=0))
  _rollout(alg)
  alg.storage.log_probs[..., 1] = float("nan")  # Only the OSL's ratio.
  before = {name: _state(model) for name, model in alg.models.items()}
  losses = alg.update()

  assert math.isnan(losses["osl/surrogate"]) and math.isfinite(losses["osl/value"])
  assert alg.skipped_steps == {
    "human_actor": 0,
    "human_critic": 0,
    "osl_actor": 2,  # Every minibatch.
    "osl_critic": 0,
  }
  _assert_same(_state(alg.osl_actor), before["osl_actor"])
  for name in ("human_actor", "human_critic", "osl_critic"):
    assert _changed(before[name], _state(alg.models[name])), name


def test_osl_std_floor_is_not_absorbing() -> None:
  """rsl_rl's std floor is a clamp, which passes no gradient at or below it: once
  momentum had carried the OSL std parameter past the floor, the std could never
  rise again, however much the advantages rewarded exploration."""
  torch.manual_seed(0)
  alg = _alg()
  alg.set_schedule(_schedule(human_freeze=10, critic_warmup=0))  # The OSL learns.
  floor, _ = gaussian_distribution(alg.osl_actor).log_std_range
  with torch.no_grad():
    std_parameter(alg.osl_actor).fill_(floor - 1e-3)  # Where momentum leaves it.
  for _ in range(3):
    _rollout(alg)
    # Advantages that reward exploration: -log p grows with the sample's |eps|.
    alg.storage.advantages[..., 1] = -alg.storage.log_probs[..., 1]
    alg.update()
  # Risen by far more than the projection's own 1e-6: the gradient raised it.
  assert float(alg.osl_std.min()) > OSL_MIN_STD * math.exp(1e-3)


def test_reward_normalization_and_bootstrap_per_agent() -> None:
  """Each channel is normalized exactly as stage 1 normalizes its one reward, and
  time-outs are bootstrapped after it with that agent's value."""
  torch.manual_seed(0)
  alg = _alg()
  alg.set_schedule(_schedule())
  stage1 = [
    SimpleNamespace(
      gamma=alg.gamma,
      _return=torch.zeros(NUM_ENVS),
      _return_mean=torch.zeros(()),
      _return_var=torch.ones(()),
      _return_count=torch.tensor(1e-4),
    )
    for _ in range(2)
  ]
  with torch.inference_mode():
    for t in range(NUM_STEPS):
      obs = _obs()
      alg.act(obs)
      rewards = torch.rand(NUM_ENVS, 2)
      dones = (torch.rand(NUM_ENVS) < 0.3).long()
      time_outs = torch.rand(NUM_ENVS) < 0.3
      alg.process_env_step(obs, rewards, dones, {"time_outs": time_outs})
      for i, ref in enumerate(stage1):
        normalized = MuscleMimicPPO._normalize_rewards(
          cast(MuscleMimicPPO, ref), rewards[:, i], dones
        )
        bootstrap = alg.gamma * alg.storage.values[t, :, i] * time_outs.float()
        torch.testing.assert_close(alg.storage.rewards[t, :, i], normalized + bootstrap)


def test_save_load_round_trip() -> None:
  torch.manual_seed(0)
  alg = _alg()
  alg.set_schedule(_schedule(human_freeze=0, critic_warmup=0))
  alg.set_stage_iteration(3)
  _rollout(alg)
  alg.update()
  buffer = io.BytesIO()
  torch.save(alg.save(), buffer)
  buffer.seek(0)
  loaded = torch.load(buffer, weights_only=False)

  other = _alg()
  other.load(loaded)
  assert other.stage_iteration == 3
  for name in MODULES:
    _assert_same(other.models[name].state_dict(), alg.models[name].state_dict())
    _assert_same(other.optimizers[name].state_dict(), alg.optimizers[name].state_dict())
  _assert_same(other.teacher_head.state_dict(), alg.teacher_head.state_dict())
  _assert_same(other.save()["reward_normalization"], alg.save()["reward_normalization"])
  # A resumed algorithm with a schedule re-applies it at the loaded iteration.
  other.set_schedule(_schedule(human_freeze=5))
  assert other.human_actor_frozen and other.lrs["human_actor"] == 0.0

  with pytest.raises(ValueError, match="warm-start"):
    other.load({"actor_state_dict": {}, "critic_state_dict": {}, "iter": 610})
  with pytest.raises(TypeError, match="strict"):  # rsl_rl's (loaded, load_cfg, ...).
    other.load(loaded, cast(Any, {"actor": True}))


def test_actors_only_load_plays_a_wider_critic() -> None:
  """play.py builds the algorithm from the registered config, whose OSL critic
  reads only the full observation. A run trained with osl_critic_obs="full+osl"
  must still play: its actors load without its critics, optimizers or statistics."""
  torch.manual_seed(0)
  trained = _alg(osl_critic_obs="full+osl")
  ws, _, _ = _stage1(trained)
  _warm_start(trained, ws, reward_switch=False)  # A non-zero teacher head.
  trained.set_schedule(_schedule(human_freeze=0, critic_warmup=0))
  trained.set_stage_iteration(3)
  _rollout(trained)
  trained.update()
  buffer = io.BytesIO()
  torch.save(trained.save(), buffer)
  buffer.seek(0)
  loaded = torch.load(buffer, weights_only=False)

  with pytest.raises(ValueError, match="osl-critic-obs") as err:
    _alg().load(loaded)  # A full restore needs the run's critics.
  assert "size mismatch" in str(err.value)
  played = _alg()  # The registered default, osl_critic_obs="full".
  played.set_schedule(_schedule())
  critics = {
    name: _state(played.models[name]) for name in ("human_critic", "osl_critic")
  }
  optimizers = copy.deepcopy({k: o.state_dict() for k, o in played.optimizers.items()})
  stats = played.save()["reward_normalization"]
  played.load(loaded, actors_only=True)

  for name in ("human_actor", "osl_actor"):
    _assert_same(played.models[name].state_dict(), loaded[f"{name}_state_dict"])
  _assert_same(played.teacher_head.state_dict(), loaded["teacher_head_state_dict"])
  for name, state in critics.items():
    _assert_same(_state(played.models[name]), state)
  _assert_same({k: o.state_dict() for k, o in played.optimizers.items()}, optimizers)
  _assert_same(played.save()["reward_normalization"], stats)
  assert played.stage_iteration == 0 and played.human_actor_frozen
  obs = _obs()
  with torch.inference_mode():
    torch.testing.assert_close(played.get_policy()(obs), trained.get_policy()(obs))
  with pytest.raises(ValueError, match="warm-start"):
    played.load({"actor_state_dict": {}}, actors_only=True)


@pytest.mark.parametrize("osl_critic_obs", ["full", "full+osl"])
def test_warm_start(osl_critic_obs: str) -> None:
  torch.manual_seed(0)
  alg = _alg(osl_critic_obs=osl_critic_obs)
  ws, unified, critic = _stage1(alg)
  _warm_start(alg, ws, reward_switch=False)

  rows_h, rows_o = alg.layout.rows("human"), alg.layout.rows("osl")
  _assert_same(alg.human_actor.state_dict(), ws.human_actor)
  for model in (alg.human_actor, alg.human_critic, alg.osl_critic):
    model.set_norm_mode("frozen")
  obs = _obs()
  with torch.no_grad():
    full = unified(obs)
    torch.testing.assert_close(alg.human_actor(obs), full[:, rows_h])
    features = alg.human_actor.mlp[:-1](alg.human_actor.get_latent(obs))
    torch.testing.assert_close(alg.teacher_head(features), full[:, rows_o])
    torch.testing.assert_close(alg.human_std, distribution_std(unified)[rows_h])
    _assert_same(alg.human_critic.state_dict(), ws.human_critic)
    if osl_critic_obs == "full":
      _assert_same(alg.osl_critic.state_dict(), ws.human_critic)
    else:
      # Zero weights on the appended OSL inputs: the stage-1 critic's values.
      assert alg.osl_critic.obs_dim == ACTOR_DIM + OSL_OBS_DIM
      torch.testing.assert_close(alg.osl_critic(obs), critic(obs))
      norm = _normalizer(alg.osl_critic)
      assert norm.count.item() == NUM_ENVS * NUM_STEPS
      assert torch.equal(norm.mean[ACTOR_DIM:], torch.zeros(OSL_OBS_DIM))
      assert torch.equal(norm.var[ACTOR_DIM:], torch.ones(OSL_OBS_DIM))
  stats = alg.save()["reward_normalization"]
  torch.testing.assert_close(stats["mean"], torch.full((2,), 0.3))
  torch.testing.assert_close(stats["var"], torch.full((2,), 2.5))
  torch.testing.assert_close(
    stats["count"], torch.full((2,), 3.0 * NUM_ENVS * NUM_STEPS)
  )


def test_warm_start_with_reward_switch() -> None:
  torch.manual_seed(0)
  alg = _alg(osl_critic_obs="full+osl")
  fresh = {name: _state(alg.models[name]) for name in ("human_critic", "osl_critic")}
  ws, _, _ = _stage1(alg)
  _warm_start(alg, ws, reward_switch=True)

  _assert_same(alg.human_actor.state_dict(), ws.human_actor)  # Still warm.
  for name, before in fresh.items():
    after = _state(alg.models[name])
    for key, value in before.items():
      if not key.startswith("obs_normalizer."):
        assert torch.equal(after[key], value), key  # Fresh weights.
    d = ACTOR_DIM
    assert torch.equal(
      after["obs_normalizer.mean"][:d], ws.human_critic["obs_normalizer.mean"]
    )
    assert torch.equal(
      after["obs_normalizer.var"][:d], ws.human_critic["obs_normalizer.var"]
    )
    assert after["obs_normalizer.count"].item() == NUM_ENVS * NUM_STEPS
  stats = alg.save()["reward_normalization"]
  assert stats["mean"].tolist() == [0.0, 0.0] and stats["var"].tolist() == [1.0, 1.0]
  torch.testing.assert_close(stats["count"], torch.full((2,), 1e-4))


def test_blend_drives_the_teacher() -> None:
  torch.manual_seed(0)
  alg = _alg()
  ws, unified, _ = _stage1(alg)
  _warm_start(alg, ws, reward_switch=False)
  alg.set_schedule(
    _schedule(human_freeze=10, critic_warmup=2, blend=True, blend_ramp=4)
  )
  rows_o = alg.layout.rows("osl")

  for k, beta in ((0, 0.0), (4, 0.5)):
    alg.set_stage_iteration(k)
    assert alg.blend_beta == beta
    torch.manual_seed(k)
    obs = _obs()
    with torch.inference_mode():
      actions = alg.act(obs)
      alg.process_env_step(obs, torch.rand(NUM_ENVS, 2), torch.zeros(NUM_ENVS), {})
      teacher = unified(obs)[:, rows_o]  # The unified policy's OSL mean.
    own = alg.storage.actions["osl"][alg.storage.step - 1]
    torch.testing.assert_close(actions[:, rows_o], (1 - beta) * teacher + beta * own)
    assert not torch.allclose(own, teacher)  # PPO keeps the OSL actor's sample.
  _rollout(alg)
  assert "osl/surrogate" in alg.update()  # k = 4: past the warm-up.
  assert alg.last_teacher_rms is not None and alg.last_teacher_rms > 0.0


def test_inference_policy() -> None:
  torch.manual_seed(0)
  alg = _alg()
  alg.set_schedule(_schedule())
  _rollout(alg)  # The OSL actor's statistics move away from their defaults.
  osl_stats = _state(_normalizer(alg.osl_actor))
  obs = _obs()
  manager = _env().unwrapped.action_manager
  with torch.inference_mode():
    mean = alg.get_policy()(obs)
    expected = alg.inference_policy("expected", manager)(obs)
    mu = alg.human_actor(obs)  # Frozen by the policies.
    osl = alg.osl_actor(obs)
  _assert_same(_state(_normalizer(alg.osl_actor)), osl_stats)  # Never updated.
  rows_h, rows_o = alg.layout.rows("human"), alg.layout.rows("osl")
  torch.testing.assert_close(mean[:, rows_h], mu)
  torch.testing.assert_close(mean[:, rows_o], osl)
  torch.testing.assert_close(expected[:, rows_o], osl)
  # MSK human rows are muscles: E[clip(mu + sigma eps, 0, 1)].
  torch.testing.assert_close(
    expected[:, rows_h], expected_clipped_gaussian(mu, alg.human_std, 0.0, 1.0)
  )
  with pytest.raises(ValueError, match="action_manager"):
    alg.inference_policy("expected")
  with pytest.raises(ValueError, match="layout"):
    alg.inference_policy("mean", _action_manager("robot"))


def test_construct_algorithm_contract() -> None:
  alg = _alg(osl_critic_obs="full+osl")
  sets = {name: list(model.obs_groups) for name, model in alg.models.items()}
  assert sets == {
    "human_actor": ["actor"],
    "human_critic": ["actor"],
    "osl_actor": ["osl"],
    "osl_critic": ["actor", "osl"],
  }
  assert alg.storage.groups == ("actor", "osl")
  assert distribution_std(alg.osl_actor).tolist() == pytest.approx([0.2, 0.2])
  assert not std_parameter(alg.human_actor).requires_grad  # human_learn_std=False.
  # Until a schedule is set nothing learns.
  assert alg.human_actor_frozen and alg.osl_actor_frozen
  assert set(alg.lrs.values()) == {0.0}
  with pytest.raises(RuntimeError, match="schedule"):
    alg.update()
  with pytest.raises(ValueError, match="Unknown TwoAgentPPO settings"):
    _alg(not_a_setting=1)
  cfg = _train_cfg()
  cfg["multi_gpu"] = {"world_size": 2}
  with pytest.raises(NotImplementedError):
    TwoAgentPPO.construct_algorithm(_obs(), _env(), cfg, "cpu")


def test_stage1_knobs_are_neutral_or_refused() -> None:
  """Stage-1 settings stage 2 does not use keep neutral values in the split config,
  and anything else fails loudly instead of doing nothing."""
  cfg = osl_split_runner_cfg("msk", "compliant", "clf", "test")
  algorithm = asdict(cfg.algorithm)
  for key, (neutral, _) in STAGE1_ONLY_KNOBS.items():
    assert algorithm[key] == neutral, key
  # The recipe's own values never reach the split config.
  assert algorithm["std_cap_start"] == 0.0 and algorithm["init_std_terms"] == {}
  assert algorithm["lr_final_ratio"] == 0.1  # Stage 2's schedule floor.
  # Newton-Schulz runs at the stage-1 recipe's precision: bfloat16 for MSK, the
  # robot's float32.
  assert algorithm["muon_ns_dtype"] == "bfloat16"
  robot = osl_split_runner_cfg("robot", "compliant", "clf", "test").algorithm
  assert robot.muon_ns_dtype == "float32"
  _alg()  # The defaults build.
  # On the command line (tyro builds the config) before the env is built...
  with pytest.raises(ValueError, match="std_floor = 0.03"):
    SplitPpoCfg(std_floor=0.03)
  with pytest.raises(ValueError, match="std_cap_end = 0.05"):
    replace(cfg.algorithm, std_cap_end=0.05)
  # ...and again on the runner's dict.
  overrides: dict[str, Any] = {
    "std_cap_end": 0.05,
    "std_floor": 0.03,
    "init_std_terms": {"osl_motors": 0.3},
    "learning_rate": 1e-4,
    "lr_schedule_updates": 4000,
    "lr_schedule_offset": 10,
    "actor_bias_init": "qpos0",
    "desired_kl": 0.02,
  }
  for key, value in overrides.items():
    with pytest.raises(ValueError, match=f"{key} = ") as refused:
      _alg(**{key: value})
    assert "use " in str(refused.value)


def test_muon_settings_pass_through_and_survive_resume() -> None:
  alg = _alg()
  assert {opt.ns_dtype for opt in alg.optimizers.values()} == {torch.bfloat16}
  assert {o.ns_dtype for o in _alg(muon_ns_dtype="float32").optimizers.values()} == {
    None
  }
  with pytest.raises(ValueError, match="muon_ns_dtype"):
    _alg(muon_ns_dtype="float16")

  # A checkpoint trained with other Muon settings resumes on this run's.
  torch.manual_seed(0)
  trained = _alg(muon_beta=0.8, muon_ns_steps=3, muon_ns_dtype="float32")
  trained.set_schedule(_schedule(human_freeze=0, critic_warmup=0))
  _rollout(trained)
  trained.update()
  buffer = io.BytesIO()
  torch.save(trained.save(), buffer)
  buffer.seek(0)
  resumed = _alg()
  resumed.load(torch.load(buffer, weights_only=False))
  for optimizer in resumed.optimizers.values():
    assert optimizer.ns_dtype is torch.bfloat16
    for group in optimizer.param_groups:
      assert (group["beta"], group["nesterov"], group["ns_steps"]) == (0.95, True, 5)
  # The momenta themselves come from the checkpoint.
  _assert_same(
    resumed.optimizers["osl_critic"].state_dict()["state"],
    trained.optimizers["osl_critic"].state_dict()["state"],
  )


def test_policy_and_critic_stats() -> None:
  torch.manual_seed(0)
  alg = _alg()
  alg.set_schedule(_schedule(human_freeze=10, critic_warmup=0))  # Human frozen.
  with torch.no_grad():
    std_parameter(alg.human_actor).fill_(math.log(0.07))
    _output(alg.osl_actor).bias.copy_(torch.tensor([3.0, 0.0]))  # Knee saturated.
  _rollout(alg)
  returns = alg.storage.returns.flatten(0, 1).clone()
  values = alg.storage.values.flatten(0, 1).clone()
  osl_std = float(alg.osl_std.mean())  # Recorded before the update's first step.
  alg.update()

  stats = alg.policy_stats
  assert set(stats) == {
    f"{kind}/{term}"
    for kind in ("std", "noise_gap", "mean_outside")
    for term in ("muscles", "osl_motors")
  }
  assert stats["std/muscles"] == pytest.approx(0.07, rel=1e-5)
  assert stats["std/osl_motors"] == pytest.approx(osl_std, rel=1e-5)
  assert stats["mean_outside/osl_motors"] == pytest.approx(0.5)  # Knee, not ankle.
  assert 0.0 < stats["noise_gap/osl_motors"]
  assert 0.0 <= stats["mean_outside/muscles"] <= 1.0
  for i, agent in enumerate(("human", "osl")):
    residual = (returns[:, i] - values[:, i]).var() / returns[:, i].var()
    expected = 1.0 - float(residual)
    got = alg.critic_stats[f"{agent}/explained_variance"]
    assert got == pytest.approx(expected, abs=1e-5)  # float32 variances.


def test_inference_policy_osl_modes() -> None:
  """The teacher family is the unified stage-1 policy (human trunk, its OSL head),
  raw or noise-averaged on every row as stage 1 averages it; the sampled family
  samples the OSL actor, reproducibly under a seed."""
  torch.manual_seed(0)
  alg = _alg()
  ws, unified, _ = _stage1(alg)
  _warm_start(alg, ws, reward_switch=False)
  alg.set_schedule(_schedule())
  obs = _obs()
  rows_h, rows_o = alg.layout.rows("human"), alg.layout.rows("osl")
  manager = _env().unwrapped.action_manager
  # Stage 1's bounds in action order: muscles [0, 1], OSL motors [-1, 1].
  lo = torch.zeros(alg.layout.total_dim)
  lo[rows_o] = -1.0
  with torch.inference_mode():
    teacher = alg.inference_policy("mean", osl_mode="teacher")(obs)
    torch.testing.assert_close(teacher, unified(obs))  # Every row, in place.
    teacher_expected = alg.inference_policy("expected", manager, "teacher")(obs)
    stage1_expected = expected_clipped_gaussian(
      unified(obs), distribution_std(unified), lo, torch.ones_like(lo)
    )
    torch.testing.assert_close(teacher_expected, stage1_expected)
    assert not torch.allclose(teacher_expected[:, rows_o], teacher[:, rows_o])
    mean = alg.get_policy()(obs)
    sampled = alg.inference_policy("mean", osl_mode="sampled")
    torch.manual_seed(3)
    first = sampled(obs)
    torch.manual_seed(3)
    again = sampled(obs)
  assert torch.equal(first, again)
  torch.testing.assert_close(first[:, rows_h], mean[:, rows_h])
  noise = (first[:, rows_o] - mean[:, rows_o]) / alg.osl_std
  assert 0.2 < float(noise.std()) < 5.0  # Samples at the OSL actor's std.
  with pytest.raises(ValueError, match="osl_mode"):
    alg.inference_policy("mean", osl_mode=cast(Any, "teacher_sampled"))


def test_fresh_osl_critic() -> None:
  """osl_critic_init="fresh" keeps the OSL critic's own weights (it may be built
  smaller) on the stage-1 critic's observation statistics; the human critic and the
  return statistics still come from stage 1."""
  torch.manual_seed(0)
  alg = _alg(osl_critic_init="fresh")
  fresh = _state(alg.osl_critic)
  ws, _, _ = _stage1(alg)
  _warm_start(alg, ws, reward_switch=False)
  _assert_same(alg.human_critic.state_dict(), ws.human_critic)
  after = _state(alg.osl_critic)
  for key, value in fresh.items():
    if not key.startswith("obs_normalizer."):
      assert torch.equal(after[key], value), key
  assert torch.equal(
    after["obs_normalizer.mean"], ws.human_critic["obs_normalizer.mean"]
  )
  assert after["obs_normalizer.count"].item() == NUM_ENVS * NUM_STEPS
  stats = alg.save()["reward_normalization"]
  torch.testing.assert_close(stats["mean"], torch.full((2,), 0.3))
  with pytest.raises(ValueError, match="osl_critic_init"):
    _alg(osl_critic_init="copied")

  # A resume must build the OSL critic as its run did: a smaller fresh one does
  # not load into the default width, and the error names the flags to repeat.
  cfg = _train_cfg(osl_critic_init="fresh")
  cfg["osl_critic"]["hidden_dims"] = (8,)
  small = TwoAgentPPO.construct_algorithm(_obs(), _env(), cfg, "cpu")
  buffer = io.BytesIO()
  torch.save(small.save(), buffer)
  buffer.seek(0)
  with pytest.raises(ValueError, match="--agent.osl-critic.hidden-dims") as err:
    alg.load(torch.load(buffer, weights_only=False))
  assert "--agent.algorithm.osl-critic-init" in str(err.value)
