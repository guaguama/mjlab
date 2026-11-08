"""Two-agent PPO for stage 2: the human and the OSL as separate actors.

Stage 1's :class:`~mjlab.tasks.osl_mimic.rl.ppo.MuscleMimicPPO` trains one actor
and one critic on one reward. Here two agents share the body and every episode
boundary, each with its own actor, critic, reward channel, PPO ratio and clip, and
the four networks with four optimizers:

* **human actor**: the unified actor minus its OSL output rows, warm-started
  exactly (:meth:`TwoAgentPPO.apply_warm_start`), reading the stage-1 ``actor``
  observation group. It keeps sampling at its stage-1 std, held fixed unless
  ``human_learn_std``: the MSK mean commands rely on that noise (clipped noise is
  co-contraction), so lowering it would change what the means do.
* **OSL actor**: a fresh small network on the ``osl`` group (the prosthesis's own
  sensors with history), with a learnable std above a floor that it can always
  rise from again (:func:`_keep_std_inside_range`).
* **critics**: both copies of the stage-1 critic on the ``actor`` group (the OSL
  critic optionally also reads ``osl``), each predicting its own agent's
  normalized return.

Within an update both agents use the same minibatch indices. A frozen module takes
no part: no forward pass in the loss, no optimizer step, no gradient, so its
parameters, std, observation statistics and optimizer state stay bit-identical.
The loss is the sum over the modules that train (they share no parameters), with
one ``backward()``, then one :class:`~mjlab.tasks.osl_mimic.rl.muon.MuonAdam` step
per module. Each optimizer clips its own module's global norm and skips its own
step on a non-finite gradient, so a NaN in the OSL's loss leaves the critics and
the human learning.

Everything else is stage 1's: reward normalization by the running return std (per
agent here), time-outs bootstrapped after it, GAE, the clipped value loss, and the
musclemimic normalizer modes (``"update"`` while acting, ``"transient"`` in the
loss and the bootstrap value, ``"frozen"`` for inference). Two exceptions: the
human's statistics stay ``"frozen"`` while it is frozen, and the OSL actor's keep
updating while its weights are frozen in the critic-only warm-up, so it starts
learning on settled statistics.

**The blend** (optional, off by default): while ``beta < 1`` the env receives
``(1 - beta) * teacher + beta * osl_actor`` for the OSL rows, where the teacher is
the unified policy's OSL head on the frozen human trunk's features (exactly the
unified policy's OSL mean). PPO still learns from the OSL actor's own sample.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict
from torch import nn

from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel, RunningMeanStd
from mjlab.tasks.osl_mimic.rl.muon import MuonAdam

from .expected import action_bounds
from .layout import ActionLayout
from .policy import (
  HumanMode,
  SplitInferencePolicy,
  distribution_std,
  gaussian_distribution,
  std_parameter,
)
from .schedules import MODULES, Module, SplitSchedule
from .storage import MiniBatch, TwoAgentRolloutStorage

if TYPE_CHECKING:
  from mjlab.managers.action_manager import ActionManager

  from .warmstart import WarmStart

OBS_SETS: dict[Module, str] = {
  "human_actor": "actor",
  "human_critic": "critic",
  "osl_actor": "osl_actor",
  "osl_critic": "osl_critic",
}
"""Each module's rsl_rl observation set, which is also its model-config key."""
OSL_OBS_GROUP = "osl"
"""The OSL actor's observation group, appended to the OSL critic's input under
``osl_critic_obs="full+osl"``."""
SCHEDULE_KEYS = frozenset(
  {
    "critic_lr",
    "osl_actor_lr",
    "human_actor_peak_lr",
    "human_actor_ramp_iterations",
    "schedule_iterations",
  }
)
"""Algorithm settings for :class:`~.schedules.SplitSchedule` (and the runner)."""
IGNORED_KEYS = frozenset(
  {
    "class_name",
    "learning_rate",
    "schedule",
    "desired_kl",
    "optimizer",
    "share_cnn_encoders",
    "lr_schedule",
    "lr_schedule_updates",
    "actor_bias_init",
  }
)
"""Inherited from stage 1's config and unused (see ``SplitPpoCfg``): the schedules
replace the single rate, and the human's output bias comes from the warm start."""
RETURN_COUNT_FRESH = 1e-4
"""A fresh return normalizer's count, as in stage 1 (``NormalizeVecReward``)."""
STD_MARGIN = 1e-6
"""How far inside its ``std_range`` the OSL actor's std parameter is kept
(:func:`_keep_std_inside_range`). More than float32's spacing at any bound under 16
in magnitude: every log-space bound of a std in rsl_rl's [1e-6, 1e6], and the OSL
floor in either parameterization (the spacing near log 0.1 is 2.4e-7)."""


class TeacherHead(nn.Linear):
  """The unified policy's OSL output rows on the human trunk's last hidden layer.

  On the frozen human trunk's penultimate features this is exactly the unified
  policy's OSL mean. ``log_std`` is the unified std of those rows, kept for
  reference (the blend uses the mean). Zero until the warm start; never optimized.
  """

  log_std: torch.Tensor

  def __init__(self, in_features: int, out_features: int) -> None:
    super().__init__(in_features, out_features)
    nn.init.zeros_(self.weight)
    nn.init.zeros_(self.bias)
    self.register_buffer("log_std", torch.zeros(out_features))
    self.requires_grad_(False)


@dataclass
class _Transition:
  """What ``act`` saw and did, completed by ``process_env_step``."""

  obs: TensorDict
  actions: dict[str, torch.Tensor]
  log_probs: torch.Tensor
  values: torch.Tensor


@dataclass(frozen=True)
class _AgentTerms:
  """One agent's part of the loss: its actor (None while frozen), its critic, its
  column of the paired tensors and its PPO constants (``clip`` also bounds the
  critic's value change)."""

  name: str
  index: int
  actor: MuscleMimicMLPModel | None
  critic: MuscleMimicMLPModel
  clip: float
  entropy_coef: float
  value_coef: float


class TwoAgentPPO:
  """Two PPO agents, human and OSL, over one env. See the module docstring.

  Construct it with :meth:`construct_algorithm` (rsl_rl's runner does), then
  :meth:`apply_warm_start`, :meth:`set_schedule`, and :meth:`set_stage_iteration`
  before every rollout. Until a schedule is set both actors count as frozen and
  every learning rate is 0, and :meth:`update` refuses to run.

  Paired tensors (log-probs, values, rewards, return statistics) have a trailing
  dimension of 2 in ``("human", "osl")`` order.
  """

  is_multi_gpu: bool = False

  def __init__(
    self,
    human_actor: MuscleMimicMLPModel,
    human_critic: MuscleMimicMLPModel,
    osl_actor: MuscleMimicMLPModel,
    osl_critic: MuscleMimicMLPModel,
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
    osl_clip_param: float = 0.2,
    osl_entropy_coef: float = 0.0,
    osl_value_loss_coef: float = 0.25,
    use_clipped_value_loss: bool = True,
    normalize_advantage_per_mini_batch: bool = True,
    max_grad_norm: float = 1.0,
    muon_beta: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    reward_normalization: bool = True,
    bootstrap_time_outs: bool = True,
    human_learn_std: bool = False,
    return_count_iterations: float = 3.0,
    osl_critic_obs: Literal["full", "full+osl"] = "full",
  ) -> None:
    if osl_critic_obs not in ("full", "full+osl"):
      raise ValueError(f"Unknown osl_critic_obs {osl_critic_obs!r}.")
    self.device = device
    self.layout = layout
    self.storage = storage
    self.human_actor = human_actor.to(device)
    self.human_critic = human_critic.to(device)
    self.osl_actor = osl_actor.to(device)
    self.osl_critic = osl_critic.to(device)
    self.models: dict[str, MuscleMimicMLPModel] = {
      "human_actor": self.human_actor,
      "human_critic": self.human_critic,
      "osl_actor": self.osl_actor,
      "osl_critic": self.osl_critic,
    }
    out_dims = {
      "human_actor": layout.dim("human"),
      "human_critic": 1,
      "osl_actor": layout.dim("osl"),
      "osl_critic": 1,
    }
    for name, model in self.models.items():
      out = _output_layer(model).out_features
      if out != out_dims[name]:
        raise ValueError(f"The {name} outputs {out} values, expected {out_dims[name]}.")
    gaussian_distribution(self.human_actor)
    gaussian_distribution(self.osl_actor)
    self.teacher_head = TeacherHead(
      _output_layer(self.human_actor).in_features, layout.dim("osl")
    ).to(device)

    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.gamma = gamma
    self.lam = lam
    self.clip_param = clip_param
    self.entropy_coef = entropy_coef
    self.value_loss_coef = value_loss_coef
    self.osl_clip_param = osl_clip_param
    self.osl_entropy_coef = osl_entropy_coef
    self.osl_value_loss_coef = osl_value_loss_coef
    self.use_clipped_value_loss = use_clipped_value_loss
    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
    self.max_grad_norm = max_grad_norm
    self.reward_normalization = reward_normalization
    self.bootstrap_time_outs = bootstrap_time_outs
    self.human_learn_std = human_learn_std
    self.return_count_iterations = return_count_iterations
    self.osl_critic_obs = osl_critic_obs

    # Every module's parameters (the human's std included, whatever
    # human_learn_std says), so checkpoints keep one structure; a parameter
    # without a gradient is skipped by MuonAdam and gets no state.
    self.optimizers: dict[str, MuonAdam] = {
      name: MuonAdam(
        model.parameters(),
        lr=0.0,
        max_grad_norm=max_grad_norm,
        beta=muon_beta,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
      )
      for name, model in self.models.items()
    }

    self._human_index = layout.index("human", device)
    self._osl_index = layout.index("osl", device)
    self._return = torch.zeros(storage.num_envs, 2, device=device)
    self._set_return_stats(0.0, 1.0, RETURN_COUNT_FRESH)

    self.stage_iteration = 0
    self.schedule: SplitSchedule | None = None
    self.last_teacher_rms: float | None = None
    """RMS of (OSL actor mean - teacher) over the last rollout, in normalized
    action units, while the human trunk was frozen; None otherwise."""
    self._teacher_sq: torch.Tensor | None = None
    self._teacher_n = 0
    self._transition: _Transition | None = None
    self._apply_freeze(human_frozen=True, osl_frozen=True)

  ##
  # Construction.
  ##

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: Any, cfg: dict, device: str
  ) -> TwoAgentPPO:
    """rsl_rl's hook: build the four models, the storage and the algorithm.

    ``cfg`` is the runner's ``asdict(SplitRunnerCfg)``. Its ``obs_groups`` are
    resolved (and written back, as rsl_rl does) for the sets ``actor``,
    ``critic``, ``osl_actor`` and ``osl_critic``. The model configs are the
    ``actor``, ``critic``, ``osl_actor`` and ``osl_critic`` entries; each model
    class must be a :class:`MuscleMimicMLPModel`.
    """
    if cfg.get("multi_gpu") is not None:
      raise NotImplementedError("TwoAgentPPO trains on a single GPU.")
    layout = ActionLayout.from_action_manager(env.unwrapped.action_manager)
    num_actions = getattr(env, "num_actions", None)
    if num_actions is not None and int(num_actions) != layout.total_dim:
      raise ValueError(f"The env has {num_actions} actions; the layout {layout.terms}.")

    algorithm = {
      k: v
      for k, v in cfg["algorithm"].items()
      if k not in IGNORED_KEYS and k not in SCHEDULE_KEYS
    }
    accepted = {
      name
      for name, p in inspect.signature(TwoAgentPPO.__init__).parameters.items()
      if p.kind is inspect.Parameter.KEYWORD_ONLY and name != "device"
    }
    unknown = sorted(set(algorithm) - accepted)
    if unknown:
      raise ValueError(f"Unknown TwoAgentPPO settings: {unknown}")

    obs_groups = resolve_obs_groups(
      obs,
      {name: list(groups) for name, groups in cfg["obs_groups"].items()},
      list(OBS_SETS.values()),
    )
    critic_set = obs_groups[OBS_SETS["osl_critic"]]
    if (
      algorithm.get("osl_critic_obs") == "full+osl" and OSL_OBS_GROUP not in critic_set
    ):
      obs_groups[OBS_SETS["osl_critic"]] = [*critic_set, OSL_OBS_GROUP]
    cfg["obs_groups"] = obs_groups

    output_dims: dict[Module, int] = {
      "human_actor": layout.dim("human"),
      "human_critic": 1,
      "osl_actor": layout.dim("osl"),
      "osl_critic": 1,
    }
    models: dict[Module, MuscleMimicMLPModel] = {
      name: _build_model(
        obs, obs_groups, OBS_SETS[name], output_dims[name], cfg[OBS_SETS[name]], device
      )
      for name in MODULES
    }
    for name, model in models.items():
      params = sum(p.numel() for p in model.parameters())
      print(
        f"[TwoAgentPPO] {name}: {list(model.obs_groups)} ({model.obs_dim}) -> "
        f"{output_dims[name]}, {params / 1e6:.2f}M parameters"
      )
    stored = [g for name in MODULES for g in obs_groups[OBS_SETS[name]]]
    storage = TwoAgentRolloutStorage(
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      stored,
      {"human": layout.dim("human"), "osl": layout.dim("osl")},
      device,
    )
    return TwoAgentPPO(
      models["human_actor"],
      models["human_critic"],
      models["osl_actor"],
      models["osl_critic"],
      storage,
      layout,
      device=device,
      **algorithm,
    )

  ##
  # Schedule.
  ##

  def set_schedule(self, schedule: SplitSchedule) -> None:
    """Install the stage's schedule and apply it at the current stage iteration."""
    self.schedule = schedule
    self.set_stage_iteration(self.stage_iteration)

  def set_stage_iteration(self, k: int) -> None:
    """Apply stage-local iteration ``k``'s freeze flags and learning rates.

    Call it before each rollout: ``act`` reads the flags (the human's
    normalizer mode, the blend) and ``update`` trains what they leave unfrozen.
    """
    if self.schedule is None:
      raise RuntimeError("Call set_schedule() before set_stage_iteration().")
    self.stage_iteration = int(k)
    self._apply_freeze(
      self.schedule.human_actor_frozen(k), self.schedule.osl_actor_frozen(k)
    )
    for name, optimizer in self.optimizers.items():
      lr = self.schedule.lr(name, k)
      for group in optimizer.param_groups:
        group["lr"] = lr

  def _apply_freeze(self, human_frozen: bool, osl_frozen: bool) -> None:
    self.human_actor.requires_grad_(not human_frozen)
    std_parameter(self.human_actor).requires_grad_(
      not human_frozen and self.human_learn_std
    )
    self.osl_actor.requires_grad_(not osl_frozen)
    self.human_critic.requires_grad_(True)
    self.osl_critic.requires_grad_(True)

  @property
  def human_actor_frozen(self) -> bool:
    if self.schedule is None:
      return True
    return self.schedule.human_actor_frozen(self.stage_iteration)

  @property
  def osl_actor_frozen(self) -> bool:
    if self.schedule is None:
      return True
    return self.schedule.osl_actor_frozen(self.stage_iteration)

  @property
  def blend_beta(self) -> float:
    if self.schedule is None:
      return 1.0
    return self.schedule.beta(self.stage_iteration)

  @property
  def lrs(self) -> dict[str, float]:
    return {
      name: float(optimizer.param_groups[0]["lr"])
      for name, optimizer in self.optimizers.items()
    }

  @property
  def learning_rate(self) -> float:
    """The OSL actor's rate: what rsl_rl's Logger writes as ``Loss/learning_rate``."""
    return self.lrs["osl_actor"]

  @property
  def human_std(self) -> torch.Tensor:
    """(A_h,) the std the human samples with."""
    return distribution_std(self.human_actor).detach()

  @property
  def osl_std(self) -> torch.Tensor:
    """(A_o,) the std the OSL actor samples with (knee, ankle)."""
    return distribution_std(self.osl_actor).detach()

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
    """Start from stage 1's networks and return statistics.

    The human actor (observation statistics included) and the teacher head load
    exactly. Without a reward switch both critics load the stage-1 critic and
    both agents' return statistics take its mean and variance, with the count
    reset to ``return_count_iterations`` iterations' samples so they can follow
    the stage-2 channels (stage 1's ~1e9 would freeze them). An OSL critic that
    also reads the OSL observation gets zero weights on those inputs, so it
    starts as the stage-1 critic, and its observation count is reset to one
    iteration so the new inputs get statistics. With a reward switch the critics
    keep their fresh weights but take the stage-1 critic's observation mean and
    variance (count reset), and the return statistics start fresh. The OSL actor
    is never warm-started.
    """
    one_iteration = float(num_envs * num_steps)
    _load_into(self.human_actor, ws.human_actor, "human actor")
    _load_into(self.teacher_head, ws.teacher_head, "teacher head")
    if reward_switch:
      for critic in (self.human_critic, self.osl_critic):
        state = _normalizer_state(ws.human_critic, critic.obs_dim, one_iteration)
        _normalizer(critic).load_state_dict(state)
      self._set_return_stats(0.0, 1.0, RETURN_COUNT_FRESH)
    else:
      _load_into(self.human_critic, ws.human_critic, "human critic")
      _load_into(
        self.osl_critic,
        _widened_critic_state(ws.human_critic, self.osl_critic, one_iteration),
        "OSL critic",
      )
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
      return torch.full((2,), value, dtype=torch.float32, device=self.device)

    self._return_mean = full(mean)
    self._return_var = full(var)
    self._return_count = full(count)

  ##
  # Modes.
  ##

  def train_mode(self) -> None:
    for model in self.models.values():
      model.train()
    self.teacher_head.train()

  def eval_mode(self) -> None:
    for model in self.models.values():
      model.eval()
      model.set_norm_mode("frozen")
    self.teacher_head.eval()

  ##
  # Rollout.
  ##

  @torch.no_grad()
  def act(self, obs: TensorDict) -> torch.Tensor:
    """Sample both agents' actions; returns the env action ``(B, A_total)``.

    The human runs through its trunk explicitly so the teacher can share its
    penultimate features. Each agent's own sample is stored for PPO; the env
    receives the blend in the OSL rows while ``beta < 1``.
    """
    human_frozen = self.human_actor_frozen
    beta = self.blend_beta
    self.human_actor.set_norm_mode("frozen" if human_frozen else "update")
    self.osl_actor.set_norm_mode("update")
    self.human_critic.set_norm_mode("update")
    self.osl_critic.set_norm_mode("update")

    latent = self.human_actor.get_latent(obs)
    features = self.human_actor.mlp[:-1](latent)
    dist = gaussian_distribution(self.human_actor)
    dist.update(self.human_actor.mlp[-1](features))
    human = dist.sample()
    human_log_prob = dist.log_prob(human)

    osl = self.osl_actor(obs, stochastic_output=True)
    osl_log_prob = self.osl_actor.get_output_log_prob(osl)
    osl_command = osl
    if human_frozen or beta < 1.0:
      teacher = self.teacher_head(features)
      if beta < 1.0:
        osl_command = (1.0 - beta) * teacher + beta * osl
      if human_frozen:
        sq = (self.osl_actor.output_mean - teacher).square().sum()
        self._teacher_sq = sq if self._teacher_sq is None else self._teacher_sq + sq
        self._teacher_n += teacher.numel()

    values = torch.cat((self.human_critic(obs), self.osl_critic(obs)), dim=-1)
    actions = human.new_empty(human.shape[0], self.layout.total_dim)
    actions.index_copy_(1, self._human_index, human)
    actions.index_copy_(1, self._osl_index, osl_command)
    self._transition = _Transition(
      obs=obs,
      actions={"human": human, "osl": osl},
      log_probs=torch.stack((human_log_prob, osl_log_prob), dim=-1),
      values=values,
    )
    return actions

  def _normalize_rewards(
    self, rewards: torch.Tensor, dones: torch.Tensor
  ) -> torch.Tensor:
    """Stage 1's ``NormalizeVecReward``, per agent: ``(N, 2)`` rewards divided by
    each agent's running discounted-return std."""
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

    ``rewards`` is ``(B, 2)``, the agents' rewards (not the env's team reward).
    Normalized first, then time-outs bootstrapped with the stored values, which
    predict normalized returns (stage 1's order).
    """
    del obs
    transition = self._transition
    if transition is None:
      raise RuntimeError("process_env_step() without a preceding act().")
    rewards = rewards.to(device=self.device, dtype=torch.float32)
    if rewards.shape != transition.values.shape:
      raise ValueError(
        f"Expected (num_envs, 2) agent rewards, got {tuple(rewards.shape)}."
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
    self.osl_critic.set_norm_mode("transient")
    last_values = torch.cat((self.human_critic(obs), self.osl_critic(obs)), dim=-1)
    self.storage.compute_returns(last_values, self.gamma, self.lam)
    if not self.normalize_advantage_per_mini_batch:
      advantages = self.storage.advantages
      flat = advantages.flatten(0, 1)
      advantages.copy_((advantages - flat.mean(0)) / (flat.std(0) + 1e-8))

  ##
  # Update.
  ##

  def update(self) -> dict[str, float]:
    """One PPO update of every module the schedule leaves unfrozen.

    Returns the minibatch means of ``"<agent>/value"`` (always) and of
    ``"<agent>/surrogate"``, ``"<agent>/entropy"`` and ``"<agent>/clip_fraction"``
    for each actor that trained, with agent ``human`` or ``osl``. Every OSL-actor
    step is followed by :func:`_keep_std_inside_range`.
    """
    if self.schedule is None:
      raise RuntimeError("TwoAgentPPO.update() needs a schedule: set_schedule().")
    if not self.storage.is_full:
      raise RuntimeError("update() needs a full rollout.")
    train_human = not self.human_actor_frozen
    train_osl = not self.osl_actor_frozen
    training: list[Module] = ["human_critic", "osl_critic"]
    if train_human:
      training.append("human_actor")
    if train_osl:
      training.append("osl_actor")
    for name, model in self.models.items():
      model.set_norm_mode("transient" if name in training else "frozen")
    self._close_teacher_rms()
    for optimizer in self.optimizers.values():
      optimizer.zero_grad(set_to_none=True)

    agents = (
      _AgentTerms(
        "human",
        0,
        self.human_actor if train_human else None,
        self.human_critic,
        self.clip_param,
        self.entropy_coef,
        self.value_loss_coef,
      ),
      _AgentTerms(
        "osl",
        1,
        self.osl_actor if train_osl else None,
        self.osl_critic,
        self.osl_clip_param,
        self.osl_entropy_coef,
        self.osl_value_loss_coef,
      ),
    )
    sums: dict[str, torch.Tensor] = {}
    count = 0
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      advantages = batch.advantages
      if self.normalize_advantage_per_mini_batch:
        advantages = (advantages - advantages.mean(0)) / (advantages.std(0) + 1e-8)
      stats: dict[str, torch.Tensor] = {}
      loss = torch.zeros((), device=self.device)
      for agent in agents:
        i = agent.index
        if agent.actor is not None:  # A frozen actor takes no part at all.
          surrogate, entropy, clip_fraction = _policy_loss(
            agent.actor,
            batch.observations,
            batch.actions[agent.name],
            batch.old_log_probs[:, i],
            advantages[:, i],
            agent.clip,
          )
          loss = loss + surrogate - agent.entropy_coef * entropy
          stats[f"{agent.name}/surrogate"] = surrogate
          stats[f"{agent.name}/entropy"] = entropy
          stats[f"{agent.name}/clip_fraction"] = clip_fraction
        value = self._value_loss(agent.critic, batch, i, agent.clip)
        loss = loss + agent.value_coef * value
        stats[f"{agent.name}/value"] = value

      for name in training:
        self.optimizers[name].zero_grad(set_to_none=True)
      loss.backward()
      for name in training:
        self.optimizers[name].step()
      if train_osl:
        _keep_std_inside_range(self.osl_actor)
      for key, value in stats.items():
        sums[key] = sums[key] + value.detach() if key in sums else value.detach()
      count += 1

    for optimizer in self.optimizers.values():
      optimizer.zero_grad(set_to_none=True)  # Frees the gradients until next time.
    self.storage.clear()
    return {key: float(value) / max(count, 1) for key, value in sums.items()}

  def _value_loss(
    self, critic: MuscleMimicMLPModel, batch: MiniBatch, index: int, clip: float
  ) -> torch.Tensor:
    values = critic(batch.observations).squeeze(-1)
    old, returns = batch.values[:, index], batch.returns[:, index]
    if self.use_clipped_value_loss:
      clipped = old + (values - old).clamp(-clip, clip)
      return torch.max((values - returns).square(), (clipped - returns).square()).mean()
    return (values - returns).square().mean()

  def _close_teacher_rms(self) -> None:
    if self._teacher_sq is not None and self._teacher_n > 0:
      self.last_teacher_rms = math.sqrt(float(self._teacher_sq) / self._teacher_n)
    else:
      self.last_teacher_rms = None
    self._teacher_sq = None
    self._teacher_n = 0

  ##
  # Inference.
  ##

  def get_policy(self) -> SplitInferencePolicy:
    """Both actors' means: what ``play.py`` runs."""
    return self.inference_policy("mean")

  def inference_policy(
    self, human_mode: HumanMode, action_manager: ActionManager | None = None
  ) -> SplitInferencePolicy:
    """The deterministic policy, with the human's ``"mean"`` or ``"expected"``
    (noise-averaged, see :mod:`.expected`) action. ``"expected"`` reads the
    muscle rows' [0, 1] range from ``action_manager``; a given ``action_manager``
    must have this algorithm's layout."""
    bounds = None
    if action_manager is not None:
      layout = ActionLayout.from_action_manager(action_manager)
      if layout != self.layout:
        raise ValueError(f"Action layout {layout.terms} != {self.layout.terms}.")
      if human_mode == "expected":
        bounds = action_bounds(action_manager, self.layout, "human", self.device)
    elif human_mode == "expected":
      raise ValueError("human_mode='expected' needs the env's action_manager.")
    return SplitInferencePolicy(
      self.human_actor, self.osl_actor, self.layout, human_mode, bounds
    )

  ##
  # Checkpoints.
  ##

  def save(self) -> dict:
    return {
      **{
        f"{name}_state_dict": model.state_dict() for name, model in self.models.items()
      },
      "teacher_head_state_dict": self.teacher_head.state_dict(),
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
    """Restore a :meth:`save` dict (a stage-2 checkpoint).

    By default everything loads, as a resumed run needs (there is no rsl_rl
    ``load_cfg``). ``actors_only`` loads just the two actors and the teacher head,
    which is all inference needs: ``play.py`` builds the algorithm from the
    registered config, whose critics need not match the run's (a run trained with
    ``osl_critic_obs="full+osl"`` has a wider OSL critic). The critics, optimizers,
    return statistics and stage iteration then keep their current values. Either
    way the schedule, if one is set, is re-applied at the stage iteration (the
    restored one, or the current one), so the freeze flags stay consistent.

    Raises:
      ValueError: For anything that is not a stage-2 checkpoint (a stage-1 one is
        pointed to the warm start).
    """
    if not isinstance(strict, bool):
      # rsl_rl's runners call alg.load(loaded, load_cfg, strict): load_cfg would
      # land here and quietly turn strict loading off.
      raise TypeError(
        f"TwoAgentPPO.load(loaded, strict) takes a bool strict, got {strict!r}."
      )
    names: tuple[Module, ...] = ("human_actor", "osl_actor") if actors_only else MODULES
    required = [f"{name}_state_dict" for name in names]
    required.append("teacher_head_state_dict")
    if not actors_only:
      required += ["optimizer_state_dicts", "reward_normalization", "stage_iteration"]
    missing = [key for key in required if key not in loaded]
    if missing:
      hint = (
        " This is a stage-1 checkpoint: start stage 2 from it with "
        "--agent.warm-start.path."
        if "actor_state_dict" in loaded
        else ""
      )
      raise ValueError(f"Not a stage-2 checkpoint (missing {missing}).{hint}")
    for name in names:
      self.models[name].load_state_dict(loaded[f"{name}_state_dict"], strict=strict)
    self.teacher_head.load_state_dict(loaded["teacher_head_state_dict"], strict=strict)
    if not actors_only:
      self._load_training_state(loaded, strict)
    if self.schedule is not None:
      self.set_stage_iteration(self.stage_iteration)

  def _load_training_state(self, loaded: dict, strict: bool) -> None:
    """The part of :meth:`load` only a resumed run needs: the optimizers, the
    return statistics and the stage iteration."""
    optimizer_states = loaded["optimizer_state_dicts"]
    if strict and set(optimizer_states) != set(MODULES):
      raise ValueError(
        f"Optimizer states for {sorted(optimizer_states)}, not {MODULES}."
      )
    for name, state in optimizer_states.items():
      self.optimizers[name].load_state_dict(state)
    stats = loaded["reward_normalization"]
    self._return_mean = stats["mean"].to(self.device, torch.float32).clone()
    self._return_var = stats["var"].to(self.device, torch.float32).clone()
    self._return_count = stats["count"].to(self.device, torch.float32).clone()
    self._return = torch.zeros_like(self._return)
    self.stage_iteration = int(loaded["stage_iteration"])

  def broadcast_parameters(self) -> None:
    raise NotImplementedError("TwoAgentPPO trains on a single GPU.")


##
# Helpers.
##


def _build_model(
  obs: TensorDict,
  obs_groups: dict[str, list[str]],
  obs_set: str,
  output_dim: int,
  model_cfg: Mapping[str, Any],
  device: str,
) -> MuscleMimicMLPModel:
  """One model from its config: ``class_name`` resolved, and the None-valued
  optional keys rsl_rl's configs carry (``cnn_cfg``, ``distribution_cfg``,
  ``rnn_*``) dropped, as ``MjlabOnPolicyRunner`` does for the actor and critic."""
  kwargs = dict(model_cfg)
  class_name = kwargs.pop("class_name", None)
  model_class = (
    MuscleMimicMLPModel if class_name is None else resolve_callable(class_name)
  )
  if not (
    isinstance(model_class, type) and issubclass(model_class, MuscleMimicMLPModel)
  ):
    raise ValueError(
      f"The {obs_set!r} model must be a MuscleMimicMLPModel, got {class_name!r}."
    )
  for key in ("cnn_cfg", "distribution_cfg"):
    if key in kwargs and kwargs[key] is None:
      del kwargs[key]
  if kwargs.get("rnn_type") is None:
    for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      kwargs.pop(key, None)
  if "cnn_cfg" in kwargs or "rnn_type" in kwargs:
    raise ValueError(f"The {obs_set!r} model must be an MLP (no CNN or RNN).")
  return model_class(obs, obs_groups, obs_set, output_dim, **kwargs).to(device)


def _output_layer(model: MuscleMimicMLPModel) -> nn.Linear:
  layer = model.mlp[-1]
  if not isinstance(layer, nn.Linear):
    raise TypeError(f"Expected the MLP to end in a Linear, got {type(layer).__name__}.")
  return layer


def _normalizer(model: MuscleMimicMLPModel) -> RunningMeanStd:
  normalizer = model.obs_normalizer
  if not isinstance(normalizer, RunningMeanStd):
    raise TypeError("The warm start needs the critics' observation normalization on.")
  return normalizer


def _load_into(module: nn.Module, state: Mapping[str, torch.Tensor], what: str) -> None:
  try:
    module.load_state_dict(dict(state), strict=True)
  except RuntimeError as err:
    raise ValueError(f"The warm start does not fit the stage-2 {what}: {err}") from err


def _normalizer_state(
  source: Mapping[str, torch.Tensor], dim: int, count: float
) -> dict[str, torch.Tensor]:
  """The stage-1 critic's observation mean and variance for ``dim`` inputs (its
  own first; mean 0 and variance 1 for any appended), with ``count``."""
  mean = source["obs_normalizer.mean"]
  var = source["obs_normalizer.var"]
  d = mean.numel()
  if d > dim:
    raise ValueError(f"The stage-1 critic reads {d} inputs; this one only {dim}.")
  wide_mean = torch.zeros(dim, dtype=mean.dtype)
  wide_var = torch.ones(dim, dtype=var.dtype)
  wide_mean[:d] = mean.cpu()
  wide_var[:d] = var.cpu()
  return {"mean": wide_mean, "var": wide_var, "count": torch.tensor(count)}


def _widened_critic_state(
  source: Mapping[str, torch.Tensor], critic: MuscleMimicMLPModel, count: float
) -> dict[str, torch.Tensor]:
  """The stage-1 critic's state for ``critic``: verbatim when the inputs match;
  when ``critic`` also reads inputs appended after the stage-1 ones, zero first
  layer weights on them (so it starts as the stage-1 critic) and the observation
  statistics of :func:`_normalizer_state`."""
  dim = source["obs_normalizer.mean"].numel()
  if critic.obs_dim == dim:
    return dict(source)
  first = next(i for i, m in enumerate(critic.mlp) if isinstance(m, nn.Linear))
  key = f"mlp.{first}.weight"
  weight = source[key]
  target = critic.state_dict()[key]
  if target.shape[0] != weight.shape[0] or target.shape[1] < weight.shape[1]:
    raise ValueError(f"{key}: stage-1 {tuple(weight.shape)}, stage-2 {target.shape}.")
  wide = torch.zeros(target.shape, dtype=weight.dtype)
  wide[:, : weight.shape[1]] = weight.cpu()
  state = dict(source)
  state[key] = wide
  for name, value in _normalizer_state(source, critic.obs_dim, count).items():
    state[f"obs_normalizer.{name}"] = value
  return state


def _keep_std_inside_range(actor: MuscleMimicMLPModel) -> None:
  """Project the actor's std parameter :data:`STD_MARGIN` inside its ``std_range``.

  rsl_rl enforces the range by clamping the parameter, and the clamp passes no
  gradient at or beyond its bound, the float32-rounded bound included. Without this,
  the first step that carries the parameter past the floor (Adam's momentum does)
  would pin the std there: no later gradient could raise it, whatever the advantages
  said. A PPO update takes many steps, so it runs after each one.
  """
  dist = gaussian_distribution(actor)
  low, high = dist.log_std_range if dist.std_type == "log" else dist.std_range
  with torch.no_grad():
    std_parameter(actor).clamp_(low + STD_MARGIN, high - STD_MARGIN)


def _policy_loss(
  actor: MuscleMimicMLPModel,
  obs: TensorDict,
  actions: torch.Tensor,
  old_log_prob: torch.Tensor,
  advantages: torch.Tensor,
  clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """PPO's clipped surrogate, the mean entropy and the clip fraction."""
  dist = gaussian_distribution(actor)
  dist.update(actor.mlp(actor.get_latent(obs)))
  ratio = torch.exp(dist.log_prob(actions) - old_log_prob)
  clipped = ratio.clamp(1.0 - clip, 1.0 + clip)
  surrogate = torch.max(-advantages * ratio, -advantages * clipped).mean()
  clip_fraction = ((ratio - 1.0).abs() > clip).float().mean()
  return surrogate, dist.entropy.mean(), clip_fraction
