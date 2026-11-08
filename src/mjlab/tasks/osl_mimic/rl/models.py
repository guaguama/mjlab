"""musclemimic's actor-critic network as rsl_rl models.

``musclemimic/algorithms/common/networks.py``:

* ``FullyConnectedNet``: ``Dense -> LayerNorm -> activation`` per hidden layer,
  kernels orthogonal with gain sqrt(2) and zero biases, then a linear output layer
  orthogonal with gain 0.01 (actor mean and critic alike).
* ``RunningMeanStd`` in front, shared by actor and critic. Here each model has its
  own copy; both see the same observations at the same times, so their statistics
  stay identical.
* A state-independent, learnable log standard deviation (rsl_rl's
  ``GaussianDistribution`` with ``std_type="log"``).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import (
  GaussianDistribution,
  HeteroscedasticGaussianDistribution,
)
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict
from torch import nn

NormMode = Literal["update", "transient", "frozen"]


class RunningMeanStd(nn.Module):
  """musclemimic's ``RunningMeanStd``: a Welford update, then normalization.

  Its flax module updates the statistics with every batch it sees *before*
  normalizing that batch, and the caller decides whether the update persists:

  * ``"update"``: rollout steps, where musclemimic keeps the new ``run_stats``;
  * ``"transient"``: PPO minibatches and the bootstrap value, where it discards
    them, so a batch is normalized with statistics that include itself;
  * ``"frozen"``: inference and export, with the stored statistics.

  Statistics are float32 with ``count`` starting at 1e-6, as in musclemimic.
  """

  mean: torch.Tensor
  var: torch.Tensor
  count: torch.Tensor

  def __init__(self, dim: int) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-6))
    self.mode: NormMode = "frozen"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.mode == "frozen":
      return (x - self.mean) / torch.sqrt(self.var + 1e-8)
    batch = x.reshape(-1, x.shape[-1])
    n = batch.shape[0]
    batch_mean = batch.mean(0)
    batch_var = batch.var(0, unbiased=False) + 1e-6
    total = self.count + n
    delta = batch_mean - self.mean
    mean = self.mean + delta * n / total
    m2 = self.var * self.count + batch_var * n + delta.square() * self.count * n / total
    var = m2 / total
    if self.mode == "update":
      self.mean.copy_(mean.detach())
      self.var.copy_(var.detach())
      self.count.copy_(total.detach())
    return (x - mean) / torch.sqrt(var + 1e-8)


def build_mlp(
  in_dim: int,
  out_dim: int,
  hidden_dims: tuple[int, ...] | list[int],
  activation: str,
  layer_norm: bool,
  layer_norm_eps: float,
  hidden_gain: float,
  output_gain: float,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  last = in_dim
  for dim in hidden_dims:
    linear = nn.Linear(last, dim)
    nn.init.orthogonal_(linear.weight, gain=hidden_gain)
    nn.init.zeros_(linear.bias)
    layers.append(linear)
    if layer_norm:
      layers.append(nn.LayerNorm(dim, eps=layer_norm_eps))
    layers.append(resolve_nn_activation(activation))
    last = dim
  output = nn.Linear(last, out_dim)
  nn.init.orthogonal_(output.weight, gain=output_gain)
  nn.init.zeros_(output.bias)
  layers.append(output)
  return nn.Sequential(*layers)


def _gaussian(model: MLPModel) -> GaussianDistribution:
  dist = getattr(model, "distribution", None)
  if not isinstance(dist, GaussianDistribution) or isinstance(
    dist, HeteroscedasticGaussianDistribution
  ):
    raise TypeError(
      f"Expected a state-independent GaussianDistribution, got {type(dist).__name__}."
    )
  return dist


def gaussian_std_parameter(model: MLPModel) -> tuple[nn.Parameter, bool]:
  """The learnable std parameter of ``model``'s state-independent Gaussian, and
  whether it is in log space (``std_type="log"``)."""
  dist = _gaussian(model)
  if dist.std_type == "log":
    return dist.log_std_param, True
  return dist.std_param, False


def gaussian_std(model: MLPModel) -> torch.Tensor:
  """(action_dim,) the std the distribution samples with: the parameter clamped to
  its ``std_range`` (``exp`` of the clamped log std for ``std_type="log"``). Unlike
  ``output_std``, it does not depend on the last stochastic forward pass."""
  dist = _gaussian(model)
  if dist.std_type == "log":
    low, high = dist.log_std_range
    return torch.exp(dist.log_std_param.clamp(low, high))
  low, high = dist.std_range
  return dist.std_param.clamp(low, high)


def _freeze(module: nn.Module) -> None:
  """Exported copies normalize with the stored statistics."""
  for sub in module.modules():
    if isinstance(sub, RunningMeanStd):
      sub.mode = "frozen"


class MuscleMimicMLPModel(MLPModel):
  """An rsl_rl ``MLPModel`` with musclemimic's MLP and observation normalizer.

  Extra args:
    layer_norm: LayerNorm after each hidden linear layer (``use_layernorm``).
    layer_norm_eps: Its epsilon (``layernorm_eps``).
    hidden_gain, output_gain: Orthogonal init gains (sqrt(2) and 0.01).
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (1024, 1024),
    activation: str = "swish",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    layer_norm: bool = True,
    layer_norm_eps: float = 1e-5,
    hidden_gain: float = math.sqrt(2.0),
    output_gain: float = 0.01,
  ) -> None:
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=False,
      distribution_cfg=distribution_cfg,
    )
    self.obs_normalization = obs_normalization
    self.obs_normalizer = (
      RunningMeanStd(self.obs_dim) if obs_normalization else nn.Identity()
    )
    out = self.distribution.input_dim if self.distribution is not None else output_dim
    assert isinstance(out, int)
    self.mlp = build_mlp(
      self._get_latent_dim(),
      out,
      hidden_dims,
      activation,
      layer_norm,
      layer_norm_eps,
      hidden_gain,
      output_gain,
    )

  def set_norm_mode(self, mode: NormMode) -> None:
    if isinstance(self.obs_normalizer, RunningMeanStd):
      self.obs_normalizer.mode = mode

  def as_jit(self) -> nn.Module:
    module = super().as_jit()
    _freeze(module)
    return module

  def as_onnx(self, verbose: bool) -> nn.Module:
    module = super().as_onnx(verbose)
    _freeze(module)
    return module

  def update_normalization(self, obs: TensorDict) -> None:
    """No-op: statistics update during rollouts, as in musclemimic."""
    del obs

  def set_output_bias(self, bias: torch.Tensor) -> None:
    """``actor_bias_init``: start the policy mean at ``bias``."""
    output = self.mlp[-1]
    assert isinstance(output, nn.Linear)
    with torch.no_grad():
      output.bias.copy_(bias.to(output.bias))
