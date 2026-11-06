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
"""

from __future__ import annotations

from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from .models import MuscleMimicMLPModel, NormMode
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
    reward_normalization: bool = True,
    actor_bias_init: str = "none",
    bootstrap_time_outs: bool = False,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if lr_schedule not in ("linear", "constant"):
      raise ValueError(f"Unknown lr_schedule {lr_schedule!r}")
    self.optimizer = MuonAdam(
      chain(self.actor.parameters(), self.critic.parameters()),
      lr=self.learning_rate,
      max_grad_norm=self.max_grad_norm,
      beta=muon_beta,
      nesterov=muon_nesterov,
      ns_steps=muon_ns_steps,
    )
    self.base_learning_rate = float(self.learning_rate)
    self.lr_schedule = lr_schedule
    self.lr_schedule_updates = int(lr_schedule_updates)
    self.num_updates_done = 0
    self.actor_bias_init = actor_bias_init
    self.bootstrap_time_outs = bootstrap_time_outs

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
    return alg

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
    frac = 1.0 - self.num_updates_done / self.lr_schedule_updates
    return self.base_learning_rate * max(frac, 0.0)

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

      sums["value"] += value_loss.item()
      sums["surrogate"] += surrogate.item()
      sums["entropy"] += entropy.mean().item()
      sums["clip_fraction"] += (
        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
      )
      count += 1

    self.num_updates_done += 1
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
    state = loaded_dict.get("muscle_mimic_state")
    if state is not None:
      self.num_updates_done = int(state["num_updates_done"])
      self._return_mean = state["return_mean"].to(self.device)
      self._return_var = state["return_var"].to(self.device)
      self._return_count = state["return_count"].to(self.device)
    return iteration
