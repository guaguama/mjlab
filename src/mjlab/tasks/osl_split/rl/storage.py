"""Rollout storage for the two agents.

rsl_rl's ``RolloutStorage`` holds one action, value and return per step. Here each
step holds both agents' actions and a ``(human, osl)`` pair of log-probabilities,
values, rewards, returns and advantages, with one shared ``done``: the agents share
the body, so they share every episode boundary.

**Observations are stored once per env observation group,** however many networks
read the group. The human actor and both critics read the stage-1 ``actor`` group
(2338 dims for MSK): one copy is about 3 GB at 4096 envs x 80 steps, and a copy per
reader would be 9 GB. A minibatch gathers each group once as well.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from tensordict import TensorDict


def obs_group(obs: TensorDict, group: str) -> torch.Tensor:
  """``obs[group]``, checked to be a tensor (TensorDict indexing is untyped)."""
  value = obs.get(group)
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Observation group {group!r} is {type(value).__name__}.")
  return value


@dataclass
class MiniBatch:
  """One minibatch. The trailing 2 of the paired fields is ``(human, osl)``."""

  observations: TensorDict
  """(b,), with the stored groups."""
  actions: dict[str, torch.Tensor]
  """Agent -> (b, A_agent): each agent's own sampled action."""
  old_log_probs: torch.Tensor
  """(b, 2)"""
  values: torch.Tensor
  """(b, 2)"""
  returns: torch.Tensor
  """(b, 2)"""
  advantages: torch.Tensor
  """(b, 2), not normalized."""


class TwoAgentRolloutStorage:
  """Buffers for one rollout of ``num_steps`` steps of ``num_envs`` envs.

  Args:
    num_envs, num_steps: The rollout's shape.
    obs: An example observation (batch ``(num_envs,)``) for shapes and dtypes.
    obs_groups: The groups to store: the union of what the networks read.
    action_dims: ``{agent: action dim}``; its order is the order of the paired
      fields' last dimension.
    device: Where the buffers live.
  """

  def __init__(
    self,
    num_envs: int,
    num_steps: int,
    obs: TensorDict,
    obs_groups: Sequence[str],
    action_dims: Mapping[str, int],
    device: str | torch.device = "cpu",
  ) -> None:
    self.num_envs = int(num_envs)
    self.num_steps = int(num_steps)
    self.device = device
    self.groups: tuple[str, ...] = tuple(dict.fromkeys(obs_groups))
    missing = [g for g in self.groups if g not in obs.keys()]
    if missing:
      raise KeyError(f"Observation groups {missing} not in {list(obs.keys())}.")
    shape = (self.num_steps, self.num_envs)
    self.observations = TensorDict(batch_size=list(shape), device=device)
    for group in self.groups:
      value = obs_group(obs, group)
      if value.shape[0] != self.num_envs:
        raise ValueError(
          f"Group {group!r} has batch {value.shape[0]}, expected {self.num_envs}."
        )
      self.observations.set(
        group, torch.zeros(*shape, *value.shape[1:], dtype=value.dtype, device=device)
      )
    # The same tensors, typed, for the per-step writes.
    self._obs_buffers = {g: obs_group(self.observations, g) for g in self.groups}
    self.agents: tuple[str, ...] = tuple(action_dims)
    pair = len(self.agents)
    self.actions = {
      agent: torch.zeros(*shape, int(dim), device=device)
      for agent, dim in action_dims.items()
    }
    self.log_probs = torch.zeros(*shape, pair, device=device)
    self.values = torch.zeros(*shape, pair, device=device)
    self.rewards = torch.zeros(*shape, pair, device=device)
    self.dones = torch.zeros(*shape, dtype=torch.bool, device=device)
    self.returns = torch.zeros(*shape, pair, device=device)
    self.advantages = torch.zeros(*shape, pair, device=device)
    self.step = 0

  @property
  def is_full(self) -> bool:
    return self.step == self.num_steps

  def add(
    self,
    obs: TensorDict,
    actions: Mapping[str, torch.Tensor],
    log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
  ) -> None:
    """Record one step: obs ``(N, ...)``, actions ``{agent: (N, A)}``, log-probs,
    values and rewards ``(N, 2)``, dones ``(N,)``."""
    if self.step >= self.num_steps:
      raise OverflowError("Rollout storage is full; call clear() first.")
    t = self.step
    for group, buffer in self._obs_buffers.items():
      buffer[t].copy_(obs_group(obs, group))
    for agent in self.agents:
      self.actions[agent][t].copy_(actions[agent])
    self.log_probs[t].copy_(log_probs)
    self.values[t].copy_(values)
    self.rewards[t].copy_(rewards)
    self.dones[t].copy_(dones.reshape(-1).bool())
    self.step += 1

  def compute_returns(
    self, last_values: torch.Tensor, gamma: float, lam: float
  ) -> None:
    """GAE as rsl_rl computes it, per agent with the shared dones.

    ``delta_t = r_t + (1 - d_t) gamma V_{t+1} - V_t``, ``A_t = delta_t + (1 - d_t)
    gamma lam A_{t+1}`` and ``R_t = A_t + V_t``, where ``V_T`` is ``last_values``
    ``(N, 2)``. Time-outs were already bootstrapped into ``r_t``.
    """
    if not self.is_full:
      raise RuntimeError(
        f"compute_returns needs a full rollout ({self.step}/{self.num_steps} steps)."
      )
    advantage = torch.zeros_like(last_values)
    for t in reversed(range(self.num_steps)):
      next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
      not_done = 1.0 - self.dones[t].float().unsqueeze(-1)
      delta = self.rewards[t] + not_done * gamma * next_values - self.values[t]
      advantage = delta + not_done * gamma * lam * advantage
      self.returns[t].copy_(advantage + self.values[t])
    self.advantages.copy_(self.returns - self.values)

  def mini_batch_generator(
    self, num_mini_batches: int, num_epochs: int
  ) -> Iterator[MiniBatch]:
    """Shuffled minibatches, the same samples for both agents.

    Each epoch draws a fresh permutation and drops ``batch % num_mini_batches``
    random samples (rsl_rl reuses one permutation and always drops the last ones;
    stage 1's 4096 x 80 / 128 leaves no remainder either way).
    """
    batch_size = self.num_envs * self.num_steps
    mini = batch_size // num_mini_batches
    if mini < 1:
      raise ValueError(f"{batch_size} samples cannot fill {num_mini_batches} batches.")
    observations = self.observations.flatten(0, 1)  # Views: nothing is copied.
    assert isinstance(observations, TensorDict)
    actions = {agent: a.flatten(0, 1) for agent, a in self.actions.items()}
    log_probs = self.log_probs.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    for _ in range(num_epochs):
      order = torch.randperm(batch_size, device=self.device)
      for i in range(num_mini_batches):
        idx = order[i * mini : (i + 1) * mini]
        yield MiniBatch(
          observations=observations[idx],
          actions={agent: a[idx] for agent, a in actions.items()},
          old_log_probs=log_probs[idx],
          values=values[idx],
          returns=returns[idx],
          advantages=advantages[idx],
        )

  def clear(self) -> None:
    self.step = 0
