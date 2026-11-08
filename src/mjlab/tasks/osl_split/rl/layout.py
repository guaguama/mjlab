"""Which rows of the policy action belong to which agent.

The stage-1 action is the concatenation of the env's action terms in action-manager
order. The OSL agent owns the ``osl_motors`` term; the human agent owns every other
term, in order. The stage-1 orders (``osl_mimic.config.env_cfgs._actions``) are

* musculoskeletal model: ``osl_motors`` (2), then ``muscles`` (339): OSL rows 0:2;
* robot model: ``servos`` (27), then ``osl_motors`` (2): OSL rows 27:29.

So the OSL rows are *not* always the last two (musclemimic's convention). Always
slice by term name through this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OSL_TORQUE_JOINTS, Model

if TYPE_CHECKING:
  from mjlab.managers.action_manager import ActionManager

Agent = Literal["human", "osl"]
AGENTS: tuple[Agent, Agent] = ("human", "osl")
OSL_ACTION_TERM = "osl_motors"
"""The action term of the OSL motors (``JointEffortActionCfg`` over
``OSL_TORQUE_JOINTS``)."""
STAGE1_TERM_ORDER: dict[Model, tuple[str, str]] = {
  "msk": ("osl_motors", "muscles"),
  "robot": ("servos", "osl_motors"),
}
"""The stage-1 action-term order per model, as ``osl_mimic`` builds it."""


@dataclass(frozen=True)
class ActionLayout:
  """The action terms as ``(name, dim)`` pairs, in action-manager order."""

  terms: tuple[tuple[str, int], ...]
  osl_term: str = OSL_ACTION_TERM

  def __post_init__(self) -> None:
    names = [name for name, _ in self.terms]
    if names.count(self.osl_term) != 1:
      raise ValueError(
        f"Expected exactly one {self.osl_term!r} action term, got {names}."
      )
    if len(set(names)) != len(names):
      raise ValueError(f"Duplicate action terms: {names}")
    if any(dim <= 0 for _, dim in self.terms):
      raise ValueError(f"Every action term needs a positive dimension: {self.terms}")
    if len(self.terms) < 2:
      raise ValueError("The split needs at least one human action term.")

  @property
  def total_dim(self) -> int:
    return sum(dim for _, dim in self.terms)

  def term_slice(self, name: str) -> slice:
    start = 0
    for term, dim in self.terms:
      if term == name:
        return slice(start, start + dim)
      start += dim
    raise KeyError(f"No action term {name!r} in {self.terms}")

  def rows(self, agent: Agent) -> list[int]:
    """The action rows ``agent`` owns, in action order."""
    out: list[int] = []
    start = 0
    for term, dim in self.terms:
      if (term == self.osl_term) == (agent == "osl"):
        out.extend(range(start, start + dim))
      start += dim
    return out

  def dim(self, agent: Agent) -> int:
    return len(self.rows(agent))

  def index(self, agent: Agent, device: torch.device | str = "cpu") -> torch.Tensor:
    """``rows(agent)`` as a long tensor, for gathering and scattering."""
    return torch.tensor(self.rows(agent), dtype=torch.long, device=device)

  def human_terms(self) -> list[str]:
    return [name for name, _ in self.terms if name != self.osl_term]

  def to_list(self) -> list[list[str | int]]:
    """Plain-type form for checkpoints and YAML."""
    return [[name, dim] for name, dim in self.terms]

  @classmethod
  def from_list(
    cls, terms: list[list[str | int]] | list[tuple[str, int]]
  ) -> ActionLayout:
    return cls(tuple((str(name), int(dim)) for name, dim in terms))

  @classmethod
  def from_action_manager(cls, action_manager: ActionManager) -> ActionLayout:
    return cls(
      tuple(
        (name, int(action_manager.get_term(name).action_dim))
        for name in action_manager.active_terms
      )
    )

  @classmethod
  def for_model(cls, model: Model, num_actions: int) -> ActionLayout:
    """The stage-1 layout of a ``num_actions``-dim policy for ``model``."""
    osl_dim = len(OSL_TORQUE_JOINTS)
    human_dim = num_actions - osl_dim
    if human_dim <= 0:
      raise ValueError(f"{num_actions} actions cannot hold {osl_dim} OSL motors.")
    dims = {OSL_ACTION_TERM: osl_dim}
    first, second = STAGE1_TERM_ORDER[model]
    human_term = second if first == OSL_ACTION_TERM else first
    dims[human_term] = human_dim
    return cls(((first, dims[first]), (second, dims[second])))
