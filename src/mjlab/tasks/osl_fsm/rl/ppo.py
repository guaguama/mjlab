"""PPO for the human alone: stage 2's human agent, with the FSM driving the OSL.

In stage 1.5 a classical finite-state impedance controller drives the OSL from the
prosthesis's own sensors, as it runs on the device (:mod:`mjlab.tasks.osl_fsm.mdp.
actions`), and only the human learns. Of
:class:`~mjlab.tasks.osl_split.rl.ppo.TwoAgentPPO`'s four networks the human's two
remain: its actor, the unified stage-1 actor minus its OSL rows (warm-started
exactly), and its critic, the stage-1 critic, each with its own
:class:`~mjlab.tasks.osl_mimic.rl.muon.MuonAdam` on stage 2's schedule
(:class:`~mjlab.tasks.osl_split.rl.schedules.SplitSchedule`: the human actor's ramp
to its held rate, the critic's ramp and floored decay).

:class:`HumanOnlyPPO` is TwoAgentPPO's human path step for step, through the same
helpers: the normalizer modes, the sample and its log-probability, the reward
divided by its running return std with time-outs bootstrapped after it, GAE, the
clipped surrogate and value losses, a frozen actor taking no part, one optimizer step
per module with this run's Muon settings, also on a resumed run. The tests hold the
two to the same bits, so whatever separates a stage-1.5 run from a stage-2 one is
what drives the OSL, never how the human learns. (From 8 envs on, torch sums a lone
column in another order than a column of a pair, so the return mean and each
minibatch's advantage mean may differ from stage 2's in the last bit, which Muon's
bfloat16 Newton-Schulz then carries into the weights.)

Only what has no counterpart differs. The env action's OSL rows stay 0 (the FSM
action term ignores them); the runner's paired agent rewards are cut to the human's
column (the OSL channel remains a diagnostic of the team reward); the observation
sets ``osl_actor`` and ``osl_critic`` build nothing, so the ``osl`` group, which only
the exporter's deployment contract needs, is never stored; and stage 2's OSL
settings are refused unless they keep their defaults, since here they would do
nothing. Checkpoints carry ``"algorithm": "human_only"``, so neither stage's loader
takes the other's.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import torch
from rsl_rl.utils import resolve_obs_groups
from tensordict import TensorDict

from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_mimic.rl.muon import NS_DTYPES, MuonAdam
from mjlab.tasks.osl_split.config.rl_cfg import SplitPpoCfg
from mjlab.tasks.osl_split.rl.expected import action_bounds, expected_clipped_gaussian
from mjlab.tasks.osl_split.rl.layout import AGENTS, ActionLayout
from mjlab.tasks.osl_split.rl.policy import (
  HumanMode,
  OslMode,
  distribution_std,
  gaussian_distribution,
  std_parameter,
)
from mjlab.tasks.osl_split.rl.ppo import (
  IGNORED_KEYS,
  NETWORK_FLAGS,
  OBS_SETS,
  RETURN_COUNT_FRESH,
  SCHEDULE_KEYS,
  _build_model,
  _fresh_critic,
  _load_into,
  _output_layer,
  _policy_loss,
  _Transition,
  check_stage1_knobs,
)
from mjlab.tasks.osl_split.rl.schedules import Module, SplitSchedule
from mjlab.tasks.osl_split.rl.storage import MiniBatch, TwoAgentRolloutStorage

from .policy import HumanOnlyInferencePolicy

if TYPE_CHECKING:
  from mjlab.managers.action_manager import ActionManager
  from mjlab.tasks.osl_split.rl.warmstart import WarmStart

ALGORITHM = "human_only"
"""Stored under ``"algorithm"`` in every stage-1.5 checkpoint; stage 2's carry
none."""
HUMAN_MODULES: tuple[Module, Module] = ("human_actor", "human_critic")
"""The two networks, each with its own MuonAdam: the keys of
:attr:`HumanOnlyPPO.models` and of a checkpoint's optimizer states."""
HUMAN_OBS_SETS = [OBS_SETS[name] for name in HUMAN_MODULES]
"""The rsl_rl observation sets the networks read, which are also their model-config
keys: ``actor`` and ``critic``."""
HUMAN = AGENTS.index("human")
"""The human's column of the runner's paired ``(B, 2)`` agent rewards."""
OSL_ONLY_KNOBS: dict[str, str] = {
  "osl_critic_lr": "the OSL critic's rate",
  "osl_actor_lr": "the OSL actor's rate",
  "osl_clip_param": "the OSL actor's PPO clip",
  "osl_entropy_coef": "the OSL actor's entropy bonus",
  "osl_value_loss_coef": "the OSL critic's loss weight",
  "osl_critic_obs": "the OSL critic's input",
  "osl_critic_init": (
    "the OSL critic's start (and 'fresh' would lengthen the human critic's ramp to "
    "freeze.reward_switch_critic_warmup_iterations)"
  ),
}
"""Stage 2's OSL settings, with what each sets there. The stage-1.5 config inherits
them from ``SplitPpoCfg``; with no OSL actor or critic they would silently do
nothing, so :meth:`HumanOnlyPPO.construct_algorithm` refuses any value but
``SplitPpoCfg``'s default (:data:`OSL_KNOB_DEFAULTS`)."""
OSL_KNOB_DEFAULTS: dict[str, Any] = {
  f.name: f.default for f in fields(SplitPpoCfg) if f.name in OSL_ONLY_KNOBS
}
"""``SplitPpoCfg``'s value of each of :data:`OSL_ONLY_KNOBS`, read from the class so
that retuning stage 2's defaults never breaks stage 1.5."""


class HumanOnlyPPO:
  """PPO for the human actor and its critic; see the module docstring.

  Construct it with :meth:`construct_algorithm` (rsl_rl's runner does), then
  :meth:`apply_warm_start`, :meth:`set_schedule`, and :meth:`set_stage_iteration`
  before every rollout. Until a schedule is set the actor counts as frozen and both
  learning rates are 0, and :meth:`update` refuses to run.

  The stored log-probabilities, values, rewards, returns and the return statistics
  keep stage 2's trailing agent dimension with the human's column alone (size 1).
  The attributes stage 2's runner reads of an OSL actor hold its absence:
  :attr:`osl_actor_frozen` is always True, :attr:`blend_beta` 1, :attr:`osl_std`
  empty and :attr:`last_teacher_rms` None.
  """

  is_multi_gpu: bool = False
  last_teacher_rms: float | None = None
  """Always None: there is no teacher blend (stage 2 logs it while its human trunk
  is frozen)."""

  def __init__(
    self,
    human_actor: MuscleMimicMLPModel,
    human_critic: MuscleMimicMLPModel,
    storage: TwoAgentRolloutStorage,
    layout: ActionLayout,
    *,
    device: str = "cpu",
    num_learning_epochs: int = 1,
    num_mini_batches: int = 128,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_param: float = 0.2,
    entropy_coef: float = 0.0,
    value_loss_coef: float = 0.25,
    use_clipped_value_loss: bool = True,
    normalize_advantage_per_mini_batch: bool = True,
    max_grad_norm: float = 1.0,
    muon_beta: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_ns_dtype: str = "bfloat16",
    reward_normalization: bool = True,
    bootstrap_time_outs: bool = True,
    human_learn_std: bool = False,
    return_count_iterations: float = 3.0,
  ) -> None:
    if muon_ns_dtype not in NS_DTYPES:
      raise ValueError(
        f"Unknown muon_ns_dtype {muon_ns_dtype!r}; valid: {sorted(NS_DTYPES)}"
      )
    if tuple(storage.agents) != ("human",):
      # The paired buffers would broadcast the human's column over every agent.
      raise ValueError(
        f"The storage holds the agents {storage.agents}; HumanOnlyPPO stores the "
        "human alone."
      )
    self.device = device
    self.layout = layout
    self.storage = storage
    self.human_actor = human_actor.to(device)
    self.human_critic = human_critic.to(device)
    self.models: dict[str, MuscleMimicMLPModel] = {
      "human_actor": self.human_actor,
      "human_critic": self.human_critic,
    }
    out_dims = {"human_actor": layout.dim("human"), "human_critic": 1}
    for name, model in self.models.items():
      out = _output_layer(model).out_features
      if out != out_dims[name]:
        raise ValueError(f"The {name} outputs {out} values, expected {out_dims[name]}.")
    gaussian_distribution(self.human_actor)

    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.gamma = gamma
    self.lam = lam
    self.clip_param = clip_param
    self.entropy_coef = entropy_coef
    self.value_loss_coef = value_loss_coef
    self.use_clipped_value_loss = use_clipped_value_loss
    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
    self.max_grad_norm = max_grad_norm
    self.reward_normalization = reward_normalization
    self.bootstrap_time_outs = bootstrap_time_outs
    self.human_learn_std = human_learn_std
    self.return_count_iterations = return_count_iterations

    # Every parameter (the std included, whatever human_learn_std says), so
    # checkpoints keep one structure; a parameter without a gradient is skipped by
    # MuonAdam and gets no state.
    self.optimizers: dict[str, MuonAdam] = {
      name: MuonAdam(
        model.parameters(),
        lr=0.0,
        max_grad_norm=max_grad_norm,
        beta=muon_beta,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
        ns_dtype=muon_ns_dtype,
      )
      for name, model in self.models.items()
    }
    # Loading an optimizer state restores its groups' settings too; a resumed run
    # keeps this run's (_load_training_state), as stage 1's and stage 2's do.
    self._muon_settings = {
      "beta": muon_beta,
      "nesterov": muon_nesterov,
      "ns_steps": muon_ns_steps,
    }
    self._ns_dtype = NS_DTYPES[muon_ns_dtype]

    # The human's action terms in its own columns, for the policy statistics, and
    # the range each column is effectively clipped to (set by construct_algorithm,
    # which sees the env's action terms).
    self._stat_terms: list[tuple[str, slice]] = layout.agent_terms("human")
    self._bounds: tuple[torch.Tensor, torch.Tensor] | None = None
    self.policy_stats: dict[str, float] = {}
    """Per human action term, from the last update's first minibatch, under stage 1's
    names: ``std/<term>``, ``noise_gap/<term>`` (``|E[clip(mu + sigma eps)] -
    clip(mu)|``) and ``mean_outside/<term>`` (the share of means outside the
    effective range)."""
    self.critic_stats: dict[str, float] = {}
    """``human/explained_variance`` of the last rollout: ``1 - Var(R - V) / Var(R)``
    over the normalized returns and the critic's values."""

    self._human_index = layout.index("human", device)
    self._return = torch.zeros(storage.num_envs, 1, device=device)
    self._set_return_stats(0.0, 1.0, RETURN_COUNT_FRESH)

    self.stage_iteration = 0
    self.schedule: SplitSchedule | None = None
    self._transition: _Transition | None = None
    self._apply_freeze(human_frozen=True)

  ##
  # Construction.
  ##

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: Any, cfg: dict, device: str
  ) -> HumanOnlyPPO:
    """rsl_rl's hook: build the two models, the storage and the algorithm.

    ``cfg`` is the runner's ``asdict`` of a ``SplitRunnerCfg`` (the stage-1.5 config
    derives from it). Its ``obs_groups`` are resolved (and written back, as rsl_rl
    does) for the sets ``actor`` and ``critic``; any ``osl_actor`` and
    ``osl_critic`` sets are checked by rsl_rl but build nothing. The model configs
    are the ``actor`` and ``critic`` entries; each model class must be a
    :class:`MuscleMimicMLPModel`. The env keeps stage 1's action layout, the FSM's
    ``osl_motors`` term included.

    Raises:
      ValueError: On a stage-1-only or OSL-only setting changed from its neutral
        value (each would do nothing), an unknown setting, or an env whose action
        count does not match its terms.
    """
    if cfg.get("multi_gpu") is not None:
      raise NotImplementedError("HumanOnlyPPO trains on a single GPU.")
    manager = env.unwrapped.action_manager
    layout = ActionLayout.from_action_manager(manager)
    num_actions = getattr(env, "num_actions", None)
    if num_actions is not None and int(num_actions) != layout.total_dim:
      raise ValueError(f"The env has {num_actions} actions; the layout {layout.terms}.")

    check_stage1_knobs(cfg["algorithm"])
    check_osl_knobs(cfg["algorithm"])
    algorithm = {
      k: v
      for k, v in cfg["algorithm"].items()
      if k not in IGNORED_KEYS and k not in SCHEDULE_KEYS and k not in OSL_ONLY_KNOBS
    }
    accepted = {
      name
      for name, p in inspect.signature(HumanOnlyPPO.__init__).parameters.items()
      if p.kind is inspect.Parameter.KEYWORD_ONLY and name != "device"
    }
    unknown = sorted(set(algorithm) - accepted)
    if unknown:
      raise ValueError(f"Unknown HumanOnlyPPO settings: {unknown}")

    obs_groups = resolve_obs_groups(
      obs,
      {name: list(groups) for name, groups in cfg["obs_groups"].items()},
      HUMAN_OBS_SETS,
    )
    cfg["obs_groups"] = obs_groups

    output_dims: dict[Module, int] = {
      "human_actor": layout.dim("human"),
      "human_critic": 1,
    }
    models: dict[Module, MuscleMimicMLPModel] = {
      name: _build_model(
        obs, obs_groups, OBS_SETS[name], output_dims[name], cfg[OBS_SETS[name]], device
      )
      for name in HUMAN_MODULES
    }
    for name, model in models.items():
      params = sum(p.numel() for p in model.parameters())
      print(
        f"[HumanOnlyPPO] {name}: {list(model.obs_groups)} ({model.obs_dim}) -> "
        f"{output_dims[name]}, {params / 1e6:.2f}M parameters"
      )
    stored = [g for name in HUMAN_MODULES for g in obs_groups[OBS_SETS[name]]]
    storage = TwoAgentRolloutStorage(
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      stored,
      {"human": layout.dim("human")},
      device,
    )
    alg = HumanOnlyPPO(
      models["human_actor"],
      models["human_critic"],
      storage,
      layout,
      device=device,
      **algorithm,
    )
    alg._bounds = action_bounds(manager, layout, "human", device)
    return alg

  ##
  # Schedule.
  ##

  def set_schedule(self, schedule: SplitSchedule) -> None:
    """Install the stage's schedule and apply it at the current stage iteration.

    Raises:
      ValueError: For a schedule with the teacher blend, which would mix the OSL
        actor stage 1.5 does not have into commands the FSM ignores.
    """
    if schedule.blend:
      raise ValueError(
        "Stage 1.5 has no OSL actor to blend the teacher into (the FSM drives the "
        "OSL): turn the blend off (--agent.blend.enabled False)."
      )
    self.schedule = schedule
    self.set_stage_iteration(self.stage_iteration)

  def set_stage_iteration(self, k: int) -> None:
    """Apply stage-local iteration ``k``'s freeze flag and learning rates.

    Call it before each rollout: ``act`` reads the flag (the actor's normalizer
    mode) and ``update`` trains the actor only when it leaves it unfrozen.
    """
    if self.schedule is None:
      raise RuntimeError("Call set_schedule() before set_stage_iteration().")
    self.stage_iteration = int(k)
    self._apply_freeze(self.schedule.human_actor_frozen(k))
    for name, optimizer in self.optimizers.items():
      lr = self.schedule.lr(name, k)
      for group in optimizer.param_groups:
        group["lr"] = lr

  def _apply_freeze(self, human_frozen: bool) -> None:
    self.human_actor.requires_grad_(not human_frozen)
    std_parameter(self.human_actor).requires_grad_(
      not human_frozen and self.human_learn_std
    )
    self.human_critic.requires_grad_(True)

  @property
  def human_actor_frozen(self) -> bool:
    if self.schedule is None:
      return True
    return self.schedule.human_actor_frozen(self.stage_iteration)

  @property
  def osl_actor_frozen(self) -> bool:
    """Always True: the FSM drives the OSL, and no OSL network learns."""
    return True

  @property
  def blend_beta(self) -> float:
    """Always 1: no teacher is blended in (:meth:`set_schedule` refuses one)."""
    return 1.0

  @property
  def lrs(self) -> dict[str, float]:
    return {
      name: float(optimizer.param_groups[0]["lr"])
      for name, optimizer in self.optimizers.items()
    }

  @property
  def learning_rate(self) -> float:
    """The human actor's rate: what rsl_rl's Logger writes as
    ``Loss/learning_rate``."""
    return self.lrs["human_actor"]

  @property
  def human_std(self) -> torch.Tensor:
    """(A_h,) the std the human samples with."""
    return distribution_std(self.human_actor).detach()

  @property
  def osl_std(self) -> torch.Tensor:
    """Empty: there is no OSL actor, so no OSL std."""
    return torch.empty(0, device=self.device)

  @property
  def skipped_steps(self) -> dict[str, int]:
    """Optimizer steps each module skipped on a non-finite gradient, so far."""
    return {name: opt.skipped_steps for name, opt in self.optimizers.items()}

  ##
  # Warm start.
  ##

  def apply_warm_start(
    self, ws: WarmStart, *, reward_switch: bool, num_envs: int, num_steps: int
  ) -> None:
    """Start from stage 1's networks and return statistics, as stage 2's human.

    The actor (observation statistics included) loads exactly. Without a reward
    switch the critic loads the stage-1 critic and the return statistics take its
    mean and variance, with the count reset to ``return_count_iterations``
    iterations' samples so they can follow the human channel (stage 1's ~2e9 would
    freeze them). Under a reward switch the critic keeps its own weights on the
    stage-1 critic's observation statistics (count reset to one iteration), and the
    return statistics start fresh. The warm start's teacher head, the unified
    policy's OSL rows, has no use here: the FSM drives the OSL.
    """
    one_iteration = float(num_envs * num_steps)
    _load_into(self.human_actor, ws.human_actor, "human actor")
    if reward_switch:
      _fresh_critic(self.human_critic, ws.human_critic, one_iteration)
      self._set_return_stats(0.0, 1.0, RETURN_COUNT_FRESH)
    else:
      _load_into(self.human_critic, ws.human_critic, "human critic")
      stats = ws.return_stats
      self._set_return_stats(
        float(stats["mean"]),
        float(stats["var"]),
        self.return_count_iterations * one_iteration,
      )
    self._return = torch.zeros_like(self._return)

  def _set_return_stats(self, mean: float, var: float, count: float) -> None:
    # Reassigned, never updated in place: inside the runner's inference mode they
    # become inference tensors, which cannot be modified outside it.
    def full(value: float) -> torch.Tensor:
      return torch.full((1,), value, dtype=torch.float32, device=self.device)

    self._return_mean = full(mean)
    self._return_var = full(var)
    self._return_count = full(count)

  ##
  # Modes.
  ##

  def train_mode(self) -> None:
    for model in self.models.values():
      model.train()

  def eval_mode(self) -> None:
    for model in self.models.values():
      model.eval()
      model.set_norm_mode("frozen")

  ##
  # Rollout.
  ##

  @torch.no_grad()
  def act(self, obs: TensorDict) -> torch.Tensor:
    """Sample the human's action; returns the env action ``(B, A_total)``, whose
    OSL rows stay 0 for the FSM action term to ignore."""
    human_frozen = self.human_actor_frozen
    self.human_actor.set_norm_mode("frozen" if human_frozen else "update")
    self.human_critic.set_norm_mode("update")

    dist = gaussian_distribution(self.human_actor)
    dist.update(self.human_actor.mlp(self.human_actor.get_latent(obs)))
    human = dist.sample()
    log_prob = dist.log_prob(human)
    values = self.human_critic(obs)
    actions = human.new_zeros(human.shape[0], self.layout.total_dim)
    actions.index_copy_(1, self._human_index, human)
    self._transition = _Transition(
      obs=obs,
      actions={"human": human},
      log_probs=log_prob.unsqueeze(-1),
      values=values,
    )
    return actions

  def _normalize_rewards(
    self, rewards: torch.Tensor, dones: torch.Tensor
  ) -> torch.Tensor:
    """Stage 1's ``NormalizeVecReward``: ``(N, 1)`` rewards divided by the running
    discounted-return std."""
    ret = self._return * self.gamma * (1.0 - dones.float()).unsqueeze(-1) + rewards
    n = ret.shape[0]
    batch_mean = ret.mean(0)
    batch_var = ret.var(0, unbiased=False)
    total = self._return_count + n
    delta = batch_mean - self._return_mean
    self._return_mean = self._return_mean + delta * n / total
    m2 = (
      self._return_var * self._return_count
      + batch_var * n
      + delta.square() * self._return_count * n / total
    )
    self._return_var = m2 / total
    self._return_count = total
    self._return = ret
    return rewards / torch.sqrt(self._return_var + 1e-8)

  @torch.no_grad()
  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: Mapping[str, Any],
  ) -> None:
    """Complete and store the step ``act`` started.

    ``rewards`` is the runner's ``(B, 2)`` agent rewards, of which the human's
    column is used, or the human's alone, ``(B, 1)``; never the env's team reward,
    which also scores the FSM. Normalized first, then time-outs bootstrapped with
    the stored values, which predict normalized returns (stage 1's order).
    """
    del obs
    transition = self._transition
    if transition is None:
      raise RuntimeError("process_env_step() without a preceding act().")
    rewards = rewards.to(device=self.device, dtype=torch.float32)
    num_envs = transition.values.shape[0]
    if rewards.shape == (num_envs, len(AGENTS)):
      rewards = rewards[:, HUMAN : HUMAN + 1]
    elif rewards.shape != (num_envs, 1):
      raise ValueError(
        f"Expected the agents' ({num_envs}, {len(AGENTS)}) rewards or the human's "
        f"({num_envs}, 1), got {tuple(rewards.shape)}."
      )
    dones = dones.to(self.device)
    if self.reward_normalization:
      rewards = self._normalize_rewards(rewards, dones)
    if self.bootstrap_time_outs and "time_outs" in extras:
      time_outs = extras["time_outs"].to(self.device).float().unsqueeze(-1)
      rewards = rewards + self.gamma * transition.values * time_outs
    self.storage.add(
      transition.obs,
      transition.actions,
      transition.log_probs,
      transition.values,
      rewards,
      dones,
    )
    self._transition = None

  @torch.no_grad()
  def compute_returns(self, obs: TensorDict) -> None:
    self.human_critic.set_norm_mode("transient")
    self.storage.compute_returns(self.human_critic(obs), self.gamma, self.lam)
    if not self.normalize_advantage_per_mini_batch:
      advantages = self.storage.advantages
      flat = advantages.flatten(0, 1)
      advantages.copy_((advantages - flat.mean(0)) / (flat.std(0) + 1e-8))

  ##
  # Update.
  ##

  def update(self) -> dict[str, float]:
    """One PPO update of the critic, and of the actor unless the schedule freezes
    it.

    Returns the minibatch means of ``"human/value"`` (always) and of
    ``"human/surrogate"``, ``"human/entropy"`` and ``"human/clip_fraction"`` when
    the actor trained. Records :attr:`critic_stats` from the rollout and
    :attr:`policy_stats` from the first minibatch, before its step.
    """
    if self.schedule is None:
      raise RuntimeError("HumanOnlyPPO.update() needs a schedule: set_schedule().")
    if not self.storage.is_full:
      raise RuntimeError("update() needs a full rollout.")
    train_actor = not self.human_actor_frozen
    training: list[Module] = ["human_critic"]
    if train_actor:
      training.append("human_actor")
    for name, model in self.models.items():
      model.set_norm_mode("transient" if name in training else "frozen")
    self._record_critic_stats()
    for optimizer in self.optimizers.values():
      optimizer.zero_grad(set_to_none=True)

    sums: dict[str, torch.Tensor] = {}
    count = 0
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      if count == 0:
        self._record_policy_stats(batch.observations)
      advantages = batch.advantages
      if self.normalize_advantage_per_mini_batch:
        advantages = (advantages - advantages.mean(0)) / (advantages.std(0) + 1e-8)
      stats: dict[str, torch.Tensor] = {}
      loss = torch.zeros((), device=self.device)
      if train_actor:  # A frozen actor takes no part at all.
        surrogate, entropy, clip_fraction = _policy_loss(
          self.human_actor,
          batch.observations,
          batch.actions["human"],
          batch.old_log_probs[:, 0],
          advantages[:, 0],
          self.clip_param,
        )
        loss = loss + surrogate - self.entropy_coef * entropy
        stats["human/surrogate"] = surrogate
        stats["human/entropy"] = entropy
        stats["human/clip_fraction"] = clip_fraction
      value = self._value_loss(batch)
      loss = loss + self.value_loss_coef * value
      stats["human/value"] = value

      for name in training:
        self.optimizers[name].zero_grad(set_to_none=True)
      loss.backward()
      for name in training:
        self.optimizers[name].step()
      for key, value in stats.items():
        sums[key] = sums[key] + value.detach() if key in sums else value.detach()
      count += 1

    for optimizer in self.optimizers.values():
      optimizer.zero_grad(set_to_none=True)  # Frees the gradients until next time.
    self.storage.clear()
    return {key: float(value) / max(count, 1) for key, value in sums.items()}

  def _value_loss(self, batch: MiniBatch) -> torch.Tensor:
    values = self.human_critic(batch.observations).squeeze(-1)
    old, returns = batch.values[:, 0], batch.returns[:, 0]
    if self.use_clipped_value_loss:
      clip = self.clip_param
      clipped = old + (values - old).clamp(-clip, clip)
      return torch.max((values - returns).square(), (clipped - returns).square()).mean()
    return (values - returns).square().mean()

  @torch.no_grad()
  def _record_policy_stats(self, observations: TensorDict) -> None:
    """:attr:`policy_stats` from one minibatch: the actor's mean, in the
    normalizer mode the update set (``"transient"`` or ``"frozen"``)."""
    mean = self.human_actor(observations)  # The distribution mean.
    std = distribution_std(self.human_actor).detach()
    gap = outside = None
    if self._bounds is not None:
      lo, hi = self._bounds
      expected = expected_clipped_gaussian(mean, std, lo, hi)
      gap = (expected - torch.minimum(torch.maximum(mean, lo), hi)).abs()
      outside = ((mean < lo) | (mean > hi)).float()
    stats: dict[str, float] = {}
    for term, cols in self._stat_terms:
      stats[f"std/{term}"] = float(std[cols].mean())
      if gap is not None and outside is not None:
        stats[f"noise_gap/{term}"] = float(gap[:, cols].mean())
        stats[f"mean_outside/{term}"] = float(outside[:, cols].mean())
    self.policy_stats = stats

  @torch.no_grad()
  def _record_critic_stats(self) -> None:
    """:attr:`critic_stats` from the rollout's returns and values (both in
    normalized return units); a rollout of constant returns has none."""
    returns = self.storage.returns.flatten(0, 1)[:, 0]
    values = self.storage.values.flatten(0, 1)[:, 0]
    stats: dict[str, float] = {}
    var = float(returns.var())
    if math.isfinite(var) and var > 1e-12:
      residual = float((returns - values).var())
      stats["human/explained_variance"] = 1.0 - residual / var
    self.critic_stats = stats

  ##
  # Inference.
  ##

  def get_policy(self) -> HumanOnlyInferencePolicy:
    """The human's mean, zeros in the OSL rows: what ``play.py`` runs."""
    return self.inference_policy("mean")

  def inference_policy(
    self,
    human_mode: HumanMode,
    action_manager: ActionManager | None = None,
    osl_mode: OslMode = "mean",
  ) -> HumanOnlyInferencePolicy:
    """The evaluation policy: the human's ``"mean"`` or ``"expected"``
    (noise-averaged) action. ``"expected"`` reads the rows' effective ranges from
    ``action_manager``; a given ``action_manager`` must have this algorithm's
    layout. ``osl_mode`` is stage 2's runner's argument and must stay ``"mean"``:
    there is no OSL actor to sample and no teacher to play."""
    if osl_mode != "mean":
      raise ValueError(
        f"osl_mode={osl_mode!r}: stage 1.5 has no OSL actor (the FSM action term "
        "drives the OSL), so there is nothing to sample and no teacher; only "
        "'mean' is accepted."
      )
    bounds = None
    if action_manager is not None:
      layout = ActionLayout.from_action_manager(action_manager)
      if layout != self.layout:
        raise ValueError(f"Action layout {layout.terms} != {self.layout.terms}.")
      if human_mode == "expected":
        bounds = action_bounds(action_manager, self.layout, "human", self.device)
    elif human_mode == "expected":
      raise ValueError("human_mode='expected' needs the env's action_manager.")
    return HumanOnlyInferencePolicy(self.human_actor, self.layout, human_mode, bounds)

  ##
  # Checkpoints.
  ##

  def save(self) -> dict:
    return {
      "algorithm": ALGORITHM,
      **{
        f"{name}_state_dict": model.state_dict() for name, model in self.models.items()
      },
      "optimizer_state_dicts": {
        name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
      },
      "reward_normalization": {
        "mean": self._return_mean.clone(),
        "var": self._return_var.clone(),
        "count": self._return_count.clone(),
      },
      "stage_iteration": int(self.stage_iteration),
    }

  def load(
    self, loaded: dict, strict: bool = True, *, actors_only: bool = False
  ) -> None:
    """Restore a :meth:`save` dict (a stage-1.5 checkpoint).

    By default everything loads, as a resumed run needs. ``actors_only`` loads the
    actor alone, all inference needs (``play.py``); the critic, optimizers, return
    statistics and stage iteration then keep their current values. Either way the
    schedule, if one is set, is re-applied at the stage iteration (the restored one,
    or the current one), so the freeze flag stays consistent.

    Raises:
      ValueError: For anything that is not a stage-1.5 checkpoint (a stage-2 one,
        whose human learned beside an OSL actor; a stage-1 one or a warm-start file,
        which start a run instead), and for a network built otherwise than the
        checkpoint's (a resume that does not repeat its run's network flags).
    """
    if not isinstance(strict, bool):
      # rsl_rl's runners call alg.load(loaded, load_cfg, strict): load_cfg would
      # land here and quietly turn strict loading off.
      raise TypeError(
        f"HumanOnlyPPO.load(loaded, strict) takes a bool strict, got {strict!r}."
      )
    kind = loaded.get("algorithm") if isinstance(loaded, Mapping) else None
    if kind != ALGORITHM:
      raise ValueError(f"Not a stage-1.5 ({ALGORITHM}) checkpoint.{_hint(loaded)}")
    names: tuple[Module, ...] = ("human_actor",) if actors_only else HUMAN_MODULES
    required = [f"{name}_state_dict" for name in names]
    if not actors_only:
      required += ["optimizer_state_dicts", "reward_normalization", "stage_iteration"]
    missing = [key for key in required if key not in loaded]
    if missing:
      raise ValueError(f"The stage-1.5 checkpoint lacks {missing}.")
    for name in names:
      try:
        self.models[name].load_state_dict(loaded[f"{name}_state_dict"], strict=strict)
      except RuntimeError as err:
        raise ValueError(
          f"The checkpoint's {name} does not fit this run's: build it as the "
          f"checkpoint's run did (repeat its --agent.{NETWORK_FLAGS[name]}."
          f"hidden-dims).\n{err}"
        ) from err
    if not actors_only:
      self._load_training_state(loaded, strict)
    if self.schedule is not None:
      self.set_stage_iteration(self.stage_iteration)

  def _load_training_state(self, loaded: dict, strict: bool) -> None:
    """The part of :meth:`load` only a resumed run needs: the optimizers (with this
    run's Muon settings: momentum, Nesterov, Newton-Schulz steps and precision),
    the return statistics and the stage iteration."""
    optimizer_states = loaded["optimizer_state_dicts"]
    if strict and set(optimizer_states) != set(HUMAN_MODULES):
      raise ValueError(
        f"Optimizer states for {sorted(optimizer_states)}, not {HUMAN_MODULES}."
      )
    for name, state in optimizer_states.items():
      optimizer = self.optimizers[name]
      optimizer.load_state_dict(state)
      # The state brings its run's group settings; this run's Muon settings win.
      for group in optimizer.param_groups:
        group.update(self._muon_settings)
      optimizer.ns_dtype = self._ns_dtype
    stats = loaded["reward_normalization"]

    def restored(key: str) -> torch.Tensor:
      return stats[key].to(self.device, torch.float32).reshape(1).clone()

    self._return_mean = restored("mean")
    self._return_var = restored("var")
    self._return_count = restored("count")
    self._return = torch.zeros_like(self._return)
    self.stage_iteration = int(loaded["stage_iteration"])

  def broadcast_parameters(self) -> None:
    raise NotImplementedError("HumanOnlyPPO trains on a single GPU.")


##
# Helpers.
##


def check_osl_knobs(algorithm: Mapping[str, Any]) -> None:
  """Refuse an OSL setting of stage 2 (:data:`OSL_ONLY_KNOBS`) set to anything but
  ``SplitPpoCfg``'s default: stage 1.5 has no OSL actor or critic, so it would do
  nothing."""
  lines = [
    f"  {key} = {algorithm[key]!r} (default {OSL_KNOB_DEFAULTS[key]!r}): {what}"
    for key, what in OSL_ONLY_KNOBS.items()
    if key in algorithm
    and key in OSL_KNOB_DEFAULTS
    and algorithm[key] != OSL_KNOB_DEFAULTS[key]
  ]
  if lines:
    raise ValueError(
      "Stage 2's OSL settings, which stage 1.5 does not use (the FSM drives the OSL; "
      "there is no OSL actor or critic); leave them at their defaults:\n"
      + "\n".join(lines)
    )


def _hint(loaded: Any) -> str:
  """What a checkpoint that :meth:`HumanOnlyPPO.load` refuses is instead."""
  if not isinstance(loaded, Mapping):
    return f" It is a {type(loaded).__name__}, not a checkpoint dictionary."
  if "algorithm" in loaded:
    return f" Its algorithm is {loaded['algorithm']!r}."
  if "osl_actor_state_dict" in loaded or "human_actor_state_dict" in loaded:
    return (
      " This is a stage-2 (two-agent) checkpoint: its human learned beside an OSL "
      "actor, not the FSM, and stage 1.5 neither resumes nor plays it. Start stage "
      "1.5 from a stage-1 run with --agent.warm-start.path."
    )
  if "format" in loaded:
    return (
      " This is a warm-start file: start a run from it with --agent.warm-start.path."
    )
  if "actor_state_dict" in loaded:
    return (
      " This is a stage-1 checkpoint: start stage 1.5 from it with "
      "--agent.warm-start.path."
    )
  return ""
