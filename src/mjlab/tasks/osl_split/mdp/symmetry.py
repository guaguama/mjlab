"""The gait-symmetry term: the OSL side's tracking error against the intact side's,
half a stride earlier.

For each step t and each joint pair the asymmetry is

  ``d(t) = e_osl(t) - e_intact(t - T/2)``, with ``e = q - q_ref``,

the right (OSL) leg now against the left (intact) leg half a stride earlier, T the
current clip's stride period. Each side is measured against its own reference, so
asymmetry the reference already has (hip rotation, turns, pivots, backward walking)
never counts, only what the policy adds. The pairs (:data:`SYMMETRY_PAIRS`) are the
knee (the OSL knee against the intact knee), the ankle, and the hip's flexion,
adduction and rotation; a group's d² is the mean of its pairs'
(:data:`SYMMETRY_GROUPS`), and the step's asymmetry is ``s = (d_knee² + d_ankle² +
d_hip²) / 3`` (rad²).

One env step advances the library's reference one frame (its clips are at the
control rate), so half a stride is ``num_frames[clip] / 2`` steps, rounded half up.
A ring buffer per env keeps the intact side's errors since the env's history last
restarted, long enough for the library's longest half stride (the idle clip's
excluded). d counts (``valid``) only where the buffer reaches half a stride back; it
is 0 everywhere else:

* on the idle clip, which has no stride;
* for the first T/2 steps of an episode (``reset``);
* for the first T/2 steps on a new clip. A clip change restarts the history, and so
  does every step whose reference still blends the outgoing clip (time-warped onto
  the new period) into the new one: a pair never spans the switch, and the T/2
  steps start once the reference is the new clip's alone.

Pushes do not mask it: the reference stays continuous through them.

The term serves two managers:

* the reward manager, as the opt-in penalty (``--env.gait-symmetry.weight``,
  ``config/env_cfgs.py``): it returns ``1 - exp(-s / sigma^2)``, 0 where d does not
  count, which the term's weight ``-w`` turns into a penalty of at most w per step;
* the metrics manager of the stage-2 runner's validation env, always
  (:data:`SYMMETRY_METRIC`): the validation pass accumulates its per-step
  ``group_sq`` and ``valid`` into RMS asymmetries per clip family.

It needs a command with a stride period per clip
(:func:`stride_periods`: the gait library's).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

SYMMETRY_PAIRS: tuple[tuple[str, str, str], ...] = (
  ("knee", "OSL_KA_L1_osl_knee_angle_r", "knee_angle_l"),
  ("ankle", "OSL_KA_L1_osl_ankle_angle_r", "ankle_angle_l"),
  ("hip_flexion", "hip_flexion_r", "hip_flexion_l"),
  ("hip_adduction", "hip_adduction_r", "hip_adduction_l"),
  ("hip_rotation", "hip_rotation_r", "hip_rotation_l"),
)
"""``(pair, OSL-side joint, intact-side joint)``. Each pair's positive angles are the
same anatomical motion on both sides: the model's left axes are the mirror images of
the right ones, and on the library's straight-forward clips every right joint
correlates positively with its partner half a stride earlier."""
SYMMETRY_GROUPS: dict[str, tuple[int, ...]] = {
  "knee": (0,),
  "ankle": (1,),
  "hip": (2, 3, 4),
}
"""Each group's pairs, as indices into :data:`SYMMETRY_PAIRS`."""
SYMMETRY_SIGMA = 0.15
"""Radians: the penalty kernel's default asymmetry scale."""
SYMMETRY_METRIC = "gait_symmetry"
"""The validation env's metrics term (and the training env's reward term, when the
penalty is on)."""
SYMMETRY_DIAGNOSTICS: tuple[str, ...] = (
  "gait_symmetry_penalty",
  "gait_symmetry_s",
  "gait_symmetry_valid",
)
"""The keys :class:`gait_symmetry` keeps in ``components``, (B,) each: the value it
returns, ``s`` (rad², 0 where d does not count) and whether d counts (1/0)."""


def stride_periods(command: Any) -> torch.Tensor:
  """(C,) the stride period of each of a command's clips, in env steps: the gait
  library's ``num_frames``, one frame per step.

  Raises:
    ValueError: For a command without one period per clip, such as a single clip
      that does not loop (the stairs' command).
  """
  frames = getattr(command, "num_frames", None)
  if (
    not isinstance(frames, torch.Tensor)
    or frames.ndim != 1
    or not isinstance(getattr(command, "clip", None), torch.Tensor)
  ):
    raise ValueError(
      f"The gait-symmetry term needs a looping clip per env with its stride period "
      f"(the gait library's command); a {type(command).__name__} has none."
    )
  return frames


class gait_symmetry(ManagerTermBase):
  """The OSL side's tracking error against the intact side's half a stride earlier
  (see the module docstring).

  Params:
    command_name: The motion command: a gait library
      (:class:`~mjlab.tasks.osl_gait_library.mdp.GaitLibraryCommand`).
    sigma: Radians, the asymmetry scale of the returned ``1 - exp(-s / sigma^2)``.

  After every call:
    d: (B, 5) each pair's asymmetry (rad, :data:`SYMMETRY_PAIRS` order), 0 where it
      does not count.
    valid: (B,) whether it counts.
    group_sq: (B, 3) each group's d² (:data:`SYMMETRY_GROUPS` order), 0 where it
      does not count.
    s: (B,) the step's asymmetry (rad²), the mean of ``group_sq``.
    components: :data:`SYMMETRY_DIAGNOSTICS`.

  A reset restarts the history of the envs it names; the step's outputs stay, as
  the reward terms keep their diagnostics, so an episode's last step counts.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    params = cfg.params
    _check_sigma(float(params.get("sigma", SYMMETRY_SIGMA)))
    cmd = cast(Any, env.command_manager.get_term(params["command_name"]))
    periods = stride_periods(cmd)
    names = list(cmd.robot.joint_names)
    missing = [
      joint
      for _, right, left in SYMMETRY_PAIRS
      for joint in (right, left)
      if joint not in names
    ]
    if missing:
      raise ValueError(f"The robot has no joints {missing} for the symmetry pairs.")
    device = env.device
    self._right = torch.tensor(
      [names.index(right) for _, right, _ in SYMMETRY_PAIRS], device=device
    )
    self._left = torch.tensor(
      [names.index(left) for _, _, left in SYMMETRY_PAIRS], device=device
    )
    self._groups = [
      torch.tensor(pairs, device=device) for pairs in SYMMETRY_GROUPS.values()
    ]

    half = torch.div(periods.to(device) + 1, 2, rounding_mode="floor")
    idle: int | None = getattr(cmd, "idle_clip", None)
    gait = torch.ones_like(half, dtype=torch.bool)
    if idle is not None:
      gait[idle] = False
    if not bool(gait.any()):
      raise ValueError("The gait-symmetry term needs a library with a walking clip.")
    longest = int(half[gait].max())
    self.half_periods = half.clamp(max=longest)
    """(C,) half of each clip's stride, in steps (the idle clip's capped)."""
    self.history_length = longest + 1
    """Ring-buffer slots per env: the longest half stride and the current step."""
    self._idle_clip = idle

    n, pairs = env.num_envs, len(SYMMETRY_PAIRS)
    self._history = torch.zeros(n, self.history_length, pairs, device=device)
    self._count = torch.zeros(n, dtype=torch.long, device=device)
    self._clip = torch.full((n,), -1, dtype=torch.long, device=device)
    self._rows = torch.arange(n, device=device)
    self.d = torch.zeros(n, pairs, device=device)
    self.valid = torch.zeros(n, dtype=torch.bool, device=device)
    self.group_sq = torch.zeros(n, len(SYMMETRY_GROUPS), device=device)
    self.s = torch.zeros(n, device=device)
    self.components: dict[str, torch.Tensor] = {}

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._count[slice(None) if env_ids is None else env_ids] = 0

  def __call__(
    self, env: ManagerBasedRlEnv, command_name: str, sigma: float = SYMMETRY_SIGMA
  ) -> torch.Tensor:
    cmd = cast(Any, env.command_manager.get_term(command_name))
    # The reward's own comparison: the robot now against reference(1).
    error = cmd.robot.data.joint_pos - cmd.reference(1).joint_pos
    osl = error.index_select(-1, self._right)
    intact = error.index_select(-1, self._left)

    clip = cmd.clip
    restart = clip != self._clip
    blend_count = getattr(cmd, "blend_count", None)
    if isinstance(blend_count, torch.Tensor):
      # A blend is running: reference(1) still mixes in the outgoing clip, or (its
      # last step) is the new clip's first unblended frame, which the history keeps.
      restart = restart | (blend_count < int(cmd.blend_steps))
    # In place, as the stride twist's buffers: a tensor rebound inside inference
    # mode could not be reset outside it.
    self._clip.copy_(clip)
    self._count.masked_fill_(restart, 0)

    slot = self._count % self.history_length
    self._history[self._rows, slot] = intact
    self._count += 1
    half = self.half_periods[clip]
    lagged = self._history[self._rows, (slot - half) % self.history_length]

    valid = self._count > half
    if self._idle_clip is not None:
      valid = valid & (clip != self._idle_clip)
    d = osl - lagged
    group_sq = torch.stack(
      [d.index_select(-1, pairs).square().mean(-1) for pairs in self._groups], dim=-1
    )
    s = group_sq.mean(-1)
    valid = valid & torch.isfinite(s)
    self.valid = valid
    self.d = torch.where(valid[:, None], d, torch.zeros_like(d))
    self.group_sq = torch.where(valid[:, None], group_sq, torch.zeros_like(group_sq))
    self.s = torch.where(valid, s, torch.zeros_like(s))
    value = torch.where(
      valid, 1.0 - torch.exp(-self.s / sigma**2), torch.zeros_like(self.s)
    )
    self.components = {
      "gait_symmetry_penalty": value,
      "gait_symmetry_s": self.s,
      "gait_symmetry_valid": valid.float(),
    }
    return value


def _check_sigma(sigma: float) -> None:
  if not (math.isfinite(sigma) and sigma > 0.0):
    raise ValueError(f"The gait-symmetry sigma must be positive and finite: {sigma}")
