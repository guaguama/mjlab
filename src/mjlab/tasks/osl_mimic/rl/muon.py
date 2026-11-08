"""A torch port of the optimizer musclemimic trains with.

musclemimic builds ``optax.chain(clip_by_global_norm(max_grad_norm),
optax.contrib.muon(...))`` wrapped in ``optax.apply_if_finite`` (optax 0.2.8). That
is, per step:

1. Clip the gradient of *all* parameters (actor and critic together) to a global
   L2 norm: ``g <- g * max_norm / ||g||`` when ``||g|| >= max_norm``.
2. Skip the update entirely if any gradient is non-finite.
3. For 2-D parameters (every Linear weight), Muon:

   * momentum ``mu <- beta mu + (1 - beta) g`` and, with Nesterov, the
     bias-corrected ``mu_hat = beta mu / (1 - beta^(t+1)) + (1 - beta) g / (1 -
     beta^t)``;
   * orthogonalize ``mu_hat``: normalize by its Frobenius norm, then five quintic
     Newton-Schulz steps ``X <- a X + (b A + c A^2) X`` with ``A = X X^T`` and
     ``(a, b, c) = (3.4445, -4.7750, 2.0315)``, on the wide orientation;
   * scale by ``sqrt(max(1, fan_out / fan_in))`` and by the learning rate.

4. For every other parameter (biases, LayerNorm, log std), AdamW with Nesterov
   momentum (``b1 0.9, b2 0.999, eps 1e-8``, no weight decay), as optax's muon
   configures it.

A flax ``Dense`` kernel is ``(in, out)`` and a torch ``Linear`` weight ``(out,
in)``. The Newton-Schulz map satisfies ``f(X^T) = f(X)^T`` and the Frobenius norm is
transpose-invariant, so orthogonalizing the torch weight gives the transpose of what
optax computes for the kernel, which is the same update. ``fan_out / fan_in`` is
``rows / cols`` of the torch weight.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_DTYPES: dict[str, torch.dtype | None] = {"float32": None, "bfloat16": torch.bfloat16}
"""Newton-Schulz precisions; ``None`` keeps the parameter's own (float32)."""


def orthogonalize(
  g: torch.Tensor,
  steps: int = 5,
  eps: float = 1e-8,
  dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """``orthogonalize_via_newton_schulz`` with Frobenius preconditioning.

  ``dtype`` runs the iteration in another precision (``torch.bfloat16`` as in the
  reference Muon implementation, which is several times faster on tensor cores);
  the result is returned in ``g``'s dtype. The quintic iteration is built to be
  stable in bfloat16: it only needs the singular values to land in roughly
  [0.7, 1.2], not exactly on 1.
  """
  a, b, c = NS_COEFFS
  x = g if dtype is None else g.to(dtype)
  transposed = x.shape[0] > x.shape[1]
  if transposed:
    x = x.T
  x = x / (torch.linalg.matrix_norm(x, ord="fro") + eps)
  for _ in range(steps):
    gram = x @ x.T
    x = a * x + (b * gram + c * gram @ gram) @ x
  x = x.T if transposed else x
  return x.to(g.dtype)


class MuonAdam(torch.optim.Optimizer):
  """``optax.contrib.muon`` behind a global-norm clip and an ``apply_if_finite``.

  Args:
    params: Parameters; those with ``ndim == 2`` get Muon, the rest AdamW.
    lr: Learning rate, shared by both branches (``adam_learning_rate=None``).
    max_grad_norm: Global clip norm over every parameter in the optimizer.
    beta: Muon momentum.
    nesterov: Nesterov momentum, for Muon and for the AdamW branch alike.
    ns_steps: Newton-Schulz iterations.
    eps: Muon's normalization epsilon and AdamW's denominator epsilon.
    adam_b1, adam_b2: AdamW moments.
    ns_dtype: Precision of the Newton-Schulz iteration: ``"float32"`` (optax's)
      or ``"bfloat16"`` (the reference Muon implementation's; faster).
  """

  def __init__(
    self,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    max_grad_norm: float = 1.0,
    beta: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    eps: float = 1e-8,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    ns_dtype: str = "float32",
  ) -> None:
    if ns_dtype not in NS_DTYPES:
      raise ValueError(f"Unknown ns_dtype {ns_dtype!r}; valid: {sorted(NS_DTYPES)}")
    defaults = dict(
      lr=lr,
      beta=beta,
      nesterov=nesterov,
      ns_steps=ns_steps,
      eps=eps,
      adam_b1=adam_b1,
      adam_b2=adam_b2,
    )
    self.ns_dtype = NS_DTYPES[ns_dtype]
    super().__init__(params, defaults)
    self.max_grad_norm = max_grad_norm
    self.skipped_steps = 0

  def _params_with_grad(self) -> list[torch.Tensor]:
    return [
      p for group in self.param_groups for p in group["params"] if p.grad is not None
    ]

  @torch.no_grad()
  def step(self, closure=None):
    assert closure is None, "MuonAdam does not support closures."
    params = self._params_with_grad()
    if not params:
      return None
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.linalg.vector_norm(
      torch.stack([torch.linalg.vector_norm(g) for g in grads])
    )
    if not torch.isfinite(norm):
      self.skipped_steps += 1
      return None
    if norm >= self.max_grad_norm:
      scale = self.max_grad_norm / norm
      for g in grads:
        g.mul_(scale)

    for group in self.param_groups:
      lr = group["lr"]
      for p in group["params"]:
        if p.grad is None:
          continue
        if p.ndim == 2:
          self._muon_update(p, group, lr)
        else:
          self._adam_update(p, group, lr)
    return None

  def _muon_update(self, p: torch.Tensor, group: dict, lr: float) -> None:
    g = p.grad
    assert g is not None
    state = self.state[p]
    if not state:
      state["count"] = 0
      state["mu"] = torch.zeros_like(p)
    beta = group["beta"]
    mu = state["mu"]
    mu.mul_(beta).add_(g, alpha=1.0 - beta)
    state["count"] += 1
    t = state["count"]
    if group["nesterov"]:
      mu_hat = beta * mu / (1.0 - beta ** (t + 1)) + (1.0 - beta) * g / (1.0 - beta**t)
    else:
      mu_hat = mu / (1.0 - beta**t)
    update = orthogonalize(mu_hat, group["ns_steps"], group["eps"], self.ns_dtype)
    rows, cols = p.shape
    update = update * max(1.0, rows / cols) ** 0.5
    p.add_(update, alpha=-lr)

  def _adam_update(self, p: torch.Tensor, group: dict, lr: float) -> None:
    g = p.grad
    assert g is not None
    state = self.state[p]
    if not state:
      state["count"] = 0
      state["mu"] = torch.zeros_like(p)
      state["nu"] = torch.zeros_like(p)
    b1, b2 = group["adam_b1"], group["adam_b2"]
    mu, nu = state["mu"], state["nu"]
    mu.mul_(b1).add_(g, alpha=1.0 - b1)
    nu.mul_(b2).addcmul_(g, g, value=1.0 - b2)
    state["count"] += 1
    t = state["count"]
    if group["nesterov"]:
      mu_hat = b1 * mu / (1.0 - b1 ** (t + 1)) + (1.0 - b1) * g / (1.0 - b1**t)
    else:
      mu_hat = mu / (1.0 - b1**t)
    nu_hat = nu / (1.0 - b2**t)
    p.add_(mu_hat / (nu_hat.sqrt() + group["eps"]), alpha=-lr)
