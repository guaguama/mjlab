"""rsl_rl PPO with musclemimic's update rules (``musclemimic/algorithms/ppo``).

Differences from stock rsl_rl PPO, each reproducing musclemimic:

* **Optimizer:** :class:`~.muon.MuonAdam` (Muon on weight matrices, AdamW on the
  rest) behind one global-norm clip over actor and critic together, skipping
  non-finite steps. rsl_rl clips actor and critic separately and uses Adam.
* **Learning rate:** linear from ``learning_rate`` to 0 over
  ``lr_schedule_updates`` updates, constant within an update
  (``linear_lr_schedule``; ``min_lr_ratio`` does not apply to it).
* **Rewards** are divided by the running standard deviation of the discounted
  return (``NormalizeVecReward``) before they are stored. Logged rewards are raw.
* **No bootstrapping at any episode end.** musclemimic treats the trajectory end
  and the horizon as terminal, so time-outs are not bootstrapped either.
* **Value loss:** musclemimic's ``0.5 * max(clipped, unclipped)`` times
  ``vf_coef`` 0.5. Set ``value_loss_coef = 0.25`` to match.
* **Observation statistics** update on every rollout step and transiently inside
  the loss (see :class:`~.models.RunningMeanStd`).
* **actor_bias_init="qpos0"** starts each position servo's mean command at the
  model's ``qpos0`` (``PPOJax._resolve_actor_bias_init``), capped at +-0.95.

Tuning knobs that are not musclemimic's (all off by default):

* **An action-std cap** that falls geometrically from ``std_cap_start`` to
  ``std_cap_end`` between two update counts, and a floor ``std_floor``. Both are
  projections of the learnable std parameter after every optimizer step, so the std
  stays learnable below the cap (a clamp inside the distribution would pass no
  gradient at the bound). Update ``k``'s cap is applied before the rollout it
  learns from, and held through its steps. With clipped muscle commands a large std drives the plant
  by itself (see :mod:`.expected`), so a policy that keeps it learns to lean on the
  noise and its raw mean fails; the cap makes the mean carry the drive.
* **The linear rate schedule** can start counting after ``lr_schedule_offset``
  updates (a resumed run's count, for a fresh schedule) and end at
  ``lr_final_ratio`` of the rate instead of 0.
* **Newton-Schulz in bfloat16** (``muon_ns_dtype``), as the reference Muon runs it.

Every update also records :attr:`MuscleMimicPPO.policy_stats` (the std per action
term, the cap, and how far the noise moves the effective action away from the
clipped mean), which the runner logs under ``Policy/``.
"""

from __future__ import annotations

import math
from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from .expected import action_bounds, action_term_slices, expected_clipped_gaussian
from .models import MuscleMimicMLPModel, NormMode, gaussian_std, gaussian_std_parameter
from .muon import MuonAdam


def qpos0_action_bias(env) -> torch.Tensor:
  """Normalized ``qpos0`` of each position-servo action, capped at +-0.95.

  Matches musclemimic's ``ctrl0 = (Kp * gear * q0 - Kp * mid) / (Kp * half)``
  mapped through ``DefaultControl``'s identity normalization for a [-1, 1]
  ctrlrange, i.e. ``(q0 - mid) / half``. Non-servo actions get 0.
  """
  from mjlab.envs.mdp.actions import JointPositionAction

  model = env.sim.mj_model
  biases = []
  for name in env.action_manager.active_terms:
    term = env.action_manager.get_term(name)
    dim = term.action_dim
    if not isinstance(term, JointPositionAction):
      biases.append(torch.zeros(dim, device=env.device))
      continue
    robot = env.scene[term.cfg.entity_name]
    names = list(robot.joint_names)
    q_adr = robot.indexing.joint_q_adr
    targets = term.target_names
    q0 = torch.tensor(
      [float(model.qpos0[int(q_adr[names.index(t)])]) for t in targets],
      device=env.device,
    )
    offset = torch.as_tensor(term.offset, device=env.device).reshape(-1, dim)[0]
    scale = torch.as_tensor(term.scale, device=env.device).reshape(-1, dim)[0]
    biases.append(((q0 - offset) / scale).clamp(-0.95, 0.95))
  return torch.cat(biases)


def geometric_schedule(
  update: int, start: float, end: float, begin: int, finish: int
) -> float:
  """``start`` through update ``begin``, ``end`` from ``finish`` on, and the
  geometric interpolation in between (linear in log space). ``finish <= begin`` is a
  step from ``start`` to ``end`` after ``begin``."""
  if update <= begin:
    return start
  if update >= finish:
    return end
  t = (update - begin) / (finish - begin)
  return math.exp((1.0 - t) * math.log(start) + t * math.log(end))


class MuscleMimicPPO(PPO):
  actor: MuscleMimicMLPModel
  critic: MuscleMimicMLPModel

  def __init__(
    self,
    actor: MuscleMimicMLPModel,
    critic: MuscleMimicMLPModel,
    storage: RolloutStorage,
    *,
    muon_beta: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    lr_schedule: str = "linear",
    lr_schedule_updates: int = 0,
    lr_schedule_offset: int = 0,
    lr_final_ratio: float = 0.0,
    reward_normalization: bool = True,
    actor_bias_init: str = "none",
    bootstrap_time_outs: bool = False,
    std_cap_start: float = 0.0,
    std_cap_end: float = 0.0,
    std_cap_begin_update: int = 0,
    std_cap_end_update: int = 0,
    std_floor: float = 0.0,
    init_std_terms: dict[str, float] | None = None,
    muon_ns_dtype: str = "float32",
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if lr_schedule not in ("linear", "constant"):
      raise ValueError(f"Unknown lr_schedule {lr_schedule!r}")
    if not 0.0 <= lr_final_ratio <= 1.0:
      raise ValueError(f"lr_final_ratio must be in [0, 1], got {lr_final_ratio}")
    if std_cap_start < 0.0 or std_cap_end < 0.0 or std_floor < 0.0:
      raise ValueError("std_cap_start, std_cap_end and std_floor must be >= 0.")
    if std_cap_start == 0.0 and std_cap_end > 0.0:
      raise ValueError("std_cap_end needs std_cap_start (0 disables the cap).")
    cap_end = std_cap_end if std_cap_end > 0.0 else std_cap_start
    if std_floor > 0.0 and std_cap_start > 0.0 and std_floor > cap_end:
      raise ValueError(f"std_floor {std_floor} exceeds the final std cap {cap_end}.")
    self.optimizer = MuonAdam(
      chain(self.actor.parameters(), self.critic.parameters()),
      lr=self.learning_rate,
      max_grad_norm=self.max_grad_norm,
      beta=muon_beta,
      nesterov=muon_nesterov,
      ns_steps=muon_ns_steps,
      ns_dtype=muon_ns_dtype,
    )
    # Loading an optimizer state restores its groups' settings too; a resumed run
    # keeps this run's (see load).
    self._muon_settings = {
      "beta": muon_beta,
      "nesterov": muon_nesterov,
      "ns_steps": muon_ns_steps,
    }
    self.base_learning_rate = float(self.learning_rate)
    self.lr_schedule = lr_schedule
    self.lr_schedule_updates = int(lr_schedule_updates)
    self.lr_schedule_offset = int(lr_schedule_offset)
    self.lr_final_ratio = float(lr_final_ratio)
    self.num_updates_done = 0
    self.actor_bias_init = actor_bias_init
    self.bootstrap_time_outs = bootstrap_time_outs
    self.std_cap_start = float(std_cap_start)
    self.std_cap_end = float(cap_end)
    self.std_cap_begin_update = int(std_cap_begin_update)
    self.std_cap_end_update = int(std_cap_end_update)
    self.std_floor = float(std_floor)
    self.init_std_terms = {str(k): float(v) for k, v in (init_std_terms or {}).items()}
    if any(v <= 0.0 for v in self.init_std_terms.values()):
      raise ValueError(f"init_std_terms must be positive: {self.init_std_terms}")
    # Set by construct_algorithm from the env's action terms.
    self.action_lo: torch.Tensor | None = None
    self.action_hi: torch.Tensor | None = None
    self.action_slices: dict[str, slice] = {}
    self.policy_stats: dict[str, float] = {}

    self.reward_normalization = reward_normalization
    num_envs = storage.num_envs
    self._return = torch.zeros(num_envs, device=self.device)
    self._return_mean = torch.zeros((), device=self.device)
    self._return_var = torch.ones((), device=self.device)
    self._return_count = torch.tensor(1e-4, device=self.device)

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
    alg = PPO.construct_algorithm(obs, env, cfg, device)
    assert isinstance(alg, MuscleMimicPPO)
    if alg.actor_bias_init == "qpos0":
      alg.actor.set_output_bias(qpos0_action_bias(env.unwrapped))  # type: ignore[attr-defined]
    elif alg.actor_bias_init != "none":
      raise ValueError(f"Unknown actor_bias_init {alg.actor_bias_init!r}")
    manager = env.unwrapped.action_manager  # type: ignore[attr-defined]
    alg.action_lo, alg.action_hi = action_bounds(manager, device=alg.device)
    alg.action_slices = action_term_slices(manager)
    alg.set_initial_std()
    alg.apply_std_bounds()
    return alg

  ##
  # Action std.
  ##

  def std_cap(self, update: int | None = None) -> float | None:
    """The std cap at ``update`` (default: the update in progress), or None."""
    if self.std_cap_start <= 0.0:
      return None
    update = self.num_updates_done if update is None else update
    return geometric_schedule(
      update,
      self.std_cap_start,
      self.std_cap_end,
      self.std_cap_begin_update,
      self.std_cap_end_update,
    )

  def set_initial_std(self) -> None:
    """Start each action term in ``init_std_terms`` at its own std (the rest keep the
    distribution's ``init_std``). A resumed run's checkpoint overwrites it."""
    unknown = sorted(set(self.init_std_terms) - set(self.action_slices))
    if unknown:
      raise ValueError(
        f"init_std_terms names unknown action terms {unknown}; the policy's terms "
        f"are {list(self.action_slices)}."
      )
    if not self.init_std_terms:
      return
    param, log_space = gaussian_std_parameter(self.actor)
    with torch.no_grad():
      for name, std in self.init_std_terms.items():
        param[self.action_slices[name]] = math.log(std) if log_space else std

  def apply_std_bounds(self) -> None:
    """Project the actor's std parameter under the cap and above the floor."""
    cap = self.std_cap()
    floor = self.std_floor if self.std_floor > 0.0 else None
    if cap is None and floor is None:
      return
    param, log_space = gaussian_std_parameter(self.actor)
    to_param = math.log if log_space else float
    with torch.no_grad():
      if cap is not None:
        param.clamp_(max=to_param(cap))
      if floor is not None:
        param.clamp_(min=to_param(floor))

  def _record_policy_stats(self, mean: torch.Tensor) -> None:
    """Per action term: the mean std, and how far the noise moves the effective
    action from the clipped mean, ``|E[clip(mu + sigma eps)] - clip(mu)|``
    (``noise_gap``), with the fraction of means outside the effective range
    (``mean_outside``), over the batch ``mean`` (B, action_dim)."""
    with torch.no_grad():
      std = gaussian_std(self.actor)
      stats: dict[str, float] = {}
      cap = self.std_cap()
      if cap is not None:
        stats["std_cap"] = cap
      if self.action_lo is not None and self.action_hi is not None:
        lo, hi = self.action_lo, self.action_hi
        expected = expected_clipped_gaussian(mean, std, lo, hi)
        clipped = torch.minimum(torch.maximum(mean, lo), hi)
        gap = (expected - clipped).abs()
        outside = ((mean < lo) | (mean > hi)).float()
        for name, cols in self.action_slices.items():
          stats[f"std/{name}"] = float(std[cols].mean())
          stats[f"noise_gap/{name}"] = float(gap[:, cols].mean())
          stats[f"mean_outside/{name}"] = float(outside[:, cols].mean())
      self.policy_stats = stats

  ##
  # Normalization modes.
  ##

  def set_norm_mode(self, mode: NormMode) -> None:
    self.actor.set_norm_mode(mode)
    self.critic.set_norm_mode(mode)

  def eval_mode(self) -> None:
    super().eval_mode()
    self.set_norm_mode("frozen")

  ##
  # Rollout.
  ##

  def act(self, obs: TensorDict) -> torch.Tensor:
    self.set_norm_mode("update")
    return super().act(obs)

  def _normalize_rewards(
    self, rewards: torch.Tensor, dones: torch.Tensor
  ) -> torch.Tensor:
    ret = self._return * self.gamma * (1.0 - dones.float()) + rewards
    n = ret.numel()
    batch_mean = ret.mean()
    batch_var = ret.var(unbiased=False)
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

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    del obs
    if self.reward_normalization:
      rewards = self._normalize_rewards(rewards, dones)
    if self.bootstrap_time_outs and "time_outs" in extras:
      # rsl_rl's bootstrap, applied after normalization because the critic learns
      # normalized returns.
      assert self.transition.values is not None
      time_outs = extras["time_outs"].to(self.device).float()
      rewards = rewards + self.gamma * self.transition.values.squeeze(-1) * time_outs
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, obs: TensorDict) -> None:
    self.set_norm_mode("transient")
    super().compute_returns(obs)

  ##
  # Update.
  ##

  def scheduled_learning_rate(self) -> float:
    if self.lr_schedule == "constant" or self.lr_schedule_updates <= 0:
      return self.base_learning_rate
    done = max(self.num_updates_done - self.lr_schedule_offset, 0)
    frac = max(1.0 - done / self.lr_schedule_updates, 0.0)
    final = self.lr_final_ratio
    return self.base_learning_rate * (final + (1.0 - final) * frac)

  def update(self) -> dict[str, float]:
    self.set_norm_mode("transient")
    self.learning_rate = self.scheduled_learning_rate()
    for group in self.optimizer.param_groups:
      group["lr"] = self.learning_rate

    sums = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
    count = 0
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      advantages = batch.advantages
      assert advantages is not None
      if self.normalize_advantage_per_mini_batch:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
      advantages = advantages.squeeze(-1)

      self.actor(batch.observations, stochastic_output=True)
      if count == 0:
        self._record_policy_stats(self.actor.output_mean.detach())
      log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
      values = self.critic(batch.observations)
      entropy = self.actor.output_entropy

      old_log_prob = batch.old_actions_log_prob.squeeze(-1)  # type: ignore[union-attr]
      ratio = torch.exp(log_prob - old_log_prob)
      clipped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
      surrogate = torch.max(-advantages * ratio, -advantages * clipped_ratio).mean()

      old_values, returns = batch.values, batch.returns
      assert old_values is not None and returns is not None
      if self.use_clipped_value_loss:
        clipped = old_values + (values - old_values).clamp(
          -self.clip_param, self.clip_param
        )
        value_loss = torch.max(
          (values - returns).square(), (clipped - returns).square()
        )
        value_loss = value_loss.mean()
      else:
        value_loss = (values - returns).square().mean()

      loss = (
        surrogate
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      self.apply_std_bounds()

      sums["value"] += value_loss.item()
      sums["surrogate"] += surrogate.item()
      sums["entropy"] += entropy.mean().item()
      sums["clip_fraction"] += (
        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
      )
      count += 1

    self.num_updates_done += 1
    # The next rollout samples under the next update's cap, which that update's
    # steps then hold: a falling cap never shifts the std between a rollout and the
    # minibatches that score it.
    self.apply_std_bounds()
    self.storage.clear()
    return {k: v / max(count, 1) for k, v in sums.items()}

  ##
  # Checkpoints.
  ##

  def save(self) -> dict:
    saved = super().save()
    saved["muscle_mimic_state"] = {
      "num_updates_done": self.num_updates_done,
      "return_mean": self._return_mean,
      "return_var": self._return_var,
      "return_count": self._return_count,
    }
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    iteration = super().load(loaded_dict, load_cfg, strict)
    for group in self.optimizer.param_groups:
      group.update(self._muon_settings)
    state = loaded_dict.get("muscle_mimic_state")
    if state is not None:
      self.num_updates_done = int(state["num_updates_done"])
      self._return_mean = state["return_mean"].to(self.device)
      self._return_var = state["return_var"].to(self.device)
      self._return_count = state["return_count"].to(self.device)
    # A resumed run continues under this run's std schedule from the first rollout.
    self.apply_std_bounds()
    return iteration
