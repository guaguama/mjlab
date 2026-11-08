"""The stage-1 imitation reward, split into a human and an OSL reward channel.

Stage 2 trains two agents in one env, and each needs its own reward. Both must start
where stage 1 left off: the warm-started critics learned stage 1's reward, and
``Train/mean_reward`` should continue stage 1's curve. So one term computes both
channels from the stage-1 term's own intermediates (``osl_mimic``'s ``mimic_reward``
stashes them in ``_cache`` and ``_clf_cache``) and hands mjlab's reward manager the
stage-1 (team) reward, unchanged. The runner reads the channels from the term's
``channel_rewards`` instead.

A channel re-weights stage 1's coordinates *inside* the same kernels. The coordinate
groups are

* ``human``: the root and the 13 non-OSL mimic sites (and, in MimicReward's joint
  kernels, every non-OSL joint, the compliant socket's included);
* ``osl_sites``: the three OSL mimic sites (knee, ankle and foot assemblies). They
  are shared: the hip, the socket and the knee all move them;
* ``osl_joints``: the OSL knee and ankle joint errors, the only coordinates the OSL
  alone controls and senses. Stage 1's CLF has no joint coordinates, so under the
  CLF the group is new: ``e = dq / 0.2 rad`` and ``ed = dqd tau / 0.2``, paired like
  the CLF's own groups, with its own error-rate history. It starts at weight 0, so
  both channels start without it.

Every kernel takes the weighted mean over its coordinates, so only relative weights
matter. V and V-dot are sums over coordinates and split exactly; everything
nonlinear (exp, clamp, floor, clip) comes after the weighted sum, per channel. At the
stage-1 weights (:func:`stage1_channel_weights`) both channels therefore equal the
stage-1 imitation reward, apart from the owner penalty split
(``SplitRewardCfg.owner_penalties``, on by default):

* each agent pays the out-of-bounds penalty over its own action rows, normalized by
  its own action count, and the OSL motor torque penalty goes to the OSL alone. The
  human's 339 muscles would otherwise drown the OSL's two rows, and the OSL would
  be charged for muscle commands it cannot change;
* each channel's penalty is floored at -1 on its own, and its CLF violation is
  subtracted outside the floor, as stage 1 does.

The additive terms (the twist reward) are separate manager terms; the runner adds
them to the channels.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OSL_TORQUE_JOINTS
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.osl_mimic.mdp.reference import MimicReferenceSource
from mjlab.tasks.osl_mimic.mdp.rewards import (
  CLF_GROUPS,
  MIMIC_EXPS,
  ClfCfg,
  mimic_reward,
)
from mjlab.tasks.osl_split.rl.layout import AGENTS, ActionLayout

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CHANNELS: tuple[str, str] = ("human", "osl")
"""The reward channels, in :attr:`split_mimic_reward.channel_rewards` column order:
the agents' (``rl.layout.AGENTS``), so the runner can index rewards by agent."""
if CHANNELS != AGENTS:
  raise ImportError(f"Reward channels {CHANNELS} must follow the agents {AGENTS}.")
GROUPS: tuple[str, str, str] = ("human", "osl_sites", "osl_joints")
"""The coordinate groups a channel weighs."""
OSL_SITE_PREFIX = "OSL_KA_L1_"
"""Mimic sites on bodies with this prefix belong to ``osl_sites``."""

SITE_KERNELS = ("rpos", "rquat", "rvel_rot", "rvel_lin")
"""MimicReward's site kernels: per-site means over the axes, then the weighted mean
over the sites."""
JOINT_KERNELS = ("qpos", "qvel")
"""MimicReward's joint kernels: the root's coordinates (human), then every joint."""
FIXED_KERNELS = ("root_pos", "root_vel")
"""MimicReward's human-only kernels: they keep their stage-1 values in both
channels, whatever the weights."""

SPLIT_CLF_DIAGNOSTICS: tuple[str, ...] = (
  "split_V_norm_human",
  "split_V_norm_osl",
  "split_level_human",
  "split_level_osl",
  "split_violation_human",
  "split_violation_osl",
  "split_qform_human",
  "split_qform_osl_sites",
  "split_qform_osl_joints",
)
"""The CLF-only diagnostics. ``split_qform_<group>`` is the group's mean quadratic
form per coordinate in V_max units: the V_norm it would have alone."""
SPLIT_DIAGNOSTICS: tuple[str, ...] = (
  "split_reward_human",
  "split_reward_osl",
  "split_preclip_human",
  "split_preclip_osl",
  "split_clipped_human",
  "split_clipped_osl",
  "split_penalty_human",
  "split_penalty_osl",
  "split_err_osl_knee",
  "split_err_osl_ankle",
  *SPLIT_CLF_DIAGNOSTICS,
)
"""Every key :class:`split_mimic_reward` adds to ``components``, (B,) each, read by
``osl_mimic.mdp.mimic_component``. The penalties include the CLF violation, as
stage 1's ``penalty_total`` does; ``split_err_osl_*`` are ``|q - q_ref|`` in rad."""

Weights = dict[str, dict[str, float]]


def stage1_channel_weights(clf: bool) -> Weights:
  """Both channels at the weights that reproduce the stage-1 imitation reward.

  Stage 1's CLF has no joint coordinates (``osl_joints`` 0); MimicReward's joint
  kernels hold the OSL joints like every other joint (1). A reward with both the
  CLF and non-zero MimicReward weights has no exact stage-1 equivalent, since one
  ``osl_joints`` weight serves both; ``clf=True`` keeps its CLF exact.
  """
  group = {"human": 1.0, "osl_sites": 1.0, "osl_joints": 0.0 if clf else 1.0}
  return {channel: dict(group) for channel in CHANNELS}


@dataclass
class SplitRewardCfg:
  """The ``channels`` parameter of :class:`split_mimic_reward`."""

  owner_penalties: bool = True
  """Each agent pays the action penalties of its own rows (see the module
  docstring). False gives both channels the team penalty."""
  osl_joint_scale: float = 0.2
  """Radians: the CLF error scale of the OSL knee/ankle joint group, the CLF's own
  ``rangles`` scale."""
  initial_weights: dict[str, dict[str, float]] | None = None
  """``{channel: {group: weight}}`` until the runner sets its first; None: the
  stage-1 equivalents (:func:`stage1_channel_weights`)."""


def _index(values: list[int], device: str) -> torch.Tensor:
  return torch.tensor(values, dtype=torch.long, device=device)


def _group_sums(
  values: torch.Tensor, index: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
  """Per group, the sum of ``values`` (B, n) over the group's coordinates.

  Index-select, not a masked product (``0 * inf`` is NaN) or a matmul (mjlab
  enables TF32).
  """
  return {group: values.index_select(-1, idx).sum(-1) for group, idx in index.items()}


class split_mimic_reward(mimic_reward):
  """:class:`~mjlab.tasks.osl_mimic.mdp.rewards.mimic_reward` plus two reward
  channels.

  Returns the stage-1 reward unchanged, so it needs ``weight=1`` and
  ``scale_rewards_by_dt=False`` as the stage-1 term does.

  Params: every ``mimic_reward`` param, plus
    channels: A :class:`SplitRewardCfg` (or its fields as a dict); None takes the
      defaults.

  After every call:
    channel_rewards: (B, 2) the imitation reward of each channel (``CHANNELS``
      order), >= 0, a new tensor per call. A reset never touches it: the runner
      reads it after the env has auto-reset the envs that finished.
    components: the stage-1 diagnostics plus ``SPLIT_DIAGNOSTICS``.
  """

  channel_rewards: torch.Tensor

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    params = cfg.params
    channels = params.get("channels")
    if channels is None:
      channels = SplitRewardCfg()
    elif isinstance(channels, Mapping):
      channels = SplitRewardCfg(**channels)
    if not isinstance(channels, SplitRewardCfg):
      raise TypeError(f"channels must be a SplitRewardCfg, got {type(channels)}")
    if not channels.osl_joint_scale > 0.0:
      raise ValueError(f"osl_joint_scale must be > 0, got {channels.osl_joint_scale}")
    self._split = channels
    device = env.device
    cmd = cast(
      MimicReferenceSource, env.command_manager.get_term(params["command_name"])
    )

    # Mimic sites: the first is the main site; the relative sites follow, in order.
    site_bodies = tuple(getattr(getattr(cmd, "cfg", None), "site_bodies", ()))
    if len(site_bodies) != len(cmd.site_body_ids) or len(site_bodies) < 2:
      raise ValueError(
        "The split reward needs the command's cfg.site_bodies, one per site id; "
        f"got {len(site_bodies)} names for {len(cmd.site_body_ids)} sites."
      )
    relative = site_bodies[1:]
    osl_sites = [i for i, b in enumerate(relative) if b.startswith(OSL_SITE_PREFIX)]
    if len(osl_sites) != 3:
      raise ValueError(f"Expected 3 OSL mimic sites, got {osl_sites} of {relative}")
    human_sites = [i for i in range(len(relative)) if i not in osl_sites]
    self._site_index = {
      "human": _index(human_sites, device),
      "osl_sites": _index(osl_sites, device),
    }

    # Joints: MimicReward's qpos coordinates are root xyz then the joints; its qvel
    # coordinates are the root's six then the joints.
    names = list(cmd.robot.joint_names)
    missing = [j for j in OSL_TORQUE_JOINTS if j not in names]
    if missing:
      raise ValueError(f"The robot has no OSL joints {missing}.")
    osl_joints = [names.index(j) for j in OSL_TORQUE_JOINTS]
    human_joints = [j for j in range(len(names)) if j not in osl_joints]
    self._osl_joint_ids = _index(osl_joints, device)
    self._qpos_index = {
      "human": _index([0, 1, 2] + [3 + j for j in human_joints], device),
      "osl_joints": _index([3 + j for j in osl_joints], device),
    }
    self._qvel_index = {
      "human": _index(list(range(6)) + [6 + j for j in human_joints], device),
      "osl_joints": _index([6 + j for j in osl_joints], device),
    }

    # Penalty owners: action rows by term, torque-penalized joints by name.
    layout = ActionLayout.from_action_manager(env.action_manager)
    self._action_rows = {agent: layout.index(agent, device) for agent in AGENTS}
    self._action_dims = {agent: layout.dim(agent) for agent in AGENTS}
    torque = list(params.get("joint_torque_weights") or {})
    self._torque_index = {
      "human": _index(
        [i for i, j in enumerate(torque) if j not in OSL_TORQUE_JOINTS], device
      ),
      "osl": _index(
        [i for i, j in enumerate(torque) if j in OSL_TORQUE_JOINTS], device
      ),
    }

    # Coordinates per group in each kernel that mixes groups.
    n_sites = {"human": len(human_sites), "osl_sites": len(osl_sites), "osl_joints": 0}
    n_human, n_osl = len(human_joints), len(osl_joints)
    self._kernel_counts: dict[str, dict[str, int]] = {
      "qpos": {"human": 3 + n_human, "osl_sites": 0, "osl_joints": n_osl},
      "qvel": {"human": 6 + n_human, "osl_sites": 0, "osl_joints": n_osl},
      **{kernel: dict(n_sites) for kernel in SITE_KERNELS},
    }
    w = self._weights
    self._kernel_weights: dict[str, float] = {
      "qpos": w["qpos"],
      "qvel": w["qvel"],
      "root_pos": w["root_pos"],
      "root_vel": w["root_vel"],
      "rpos": w["rpos"],
      "rquat": w["rquat"],
      "rvel_rot": w["rvel"],
      "rvel_lin": w["rvel"],
    }
    self._mimic_active = any(v != 0.0 for v in w.values())

    if self._clf is not None:
      self._setup_clf_split(self._clf, len(relative), osl_sites, env)

    self.channel_rewards = torch.zeros(env.num_envs, len(CHANNELS), device=device)
    self._group_weights: Weights = {}
    initial = channels.initial_weights
    self.set_group_weights(
      initial if initial is not None else stage1_channel_weights(self._clf is not None)
    )

  def _setup_clf_split(
    self, clf: ClfCfg, n_rel: int, osl_sites: list[int], env: ManagerBasedRlEnv
  ) -> None:
    # The CLF's coordinates are its groups' errors in CLF_GROUPS order: per relative
    # site three (site-major) for rpos and rangles, then three each for the root.
    owner: list[str] = []
    for group in self._clf_groups:
      if group in ("rpos", "rangles"):
        for site in range(n_rel):
          owner += ["osl_sites" if site in osl_sites else "human"] * 3
      else:
        owner += ["human"] * 3
    assert len(owner) == self._clf_n and set(self._clf_groups) <= set(CLF_GROUPS)
    self._clf_index = {
      group: _index([i for i, o in enumerate(owner) if o == group], env.device)
      for group in ("human", "osl_sites")
    }
    self._kernel_counts["clf"] = {
      "human": owner.count("human"),
      "osl_sites": owner.count("osl_sites"),
      "osl_joints": len(OSL_TORQUE_JOINTS),
    }
    self._kernel_weights["clf"] = clf.w
    self._edj_prev = torch.zeros(
      env.num_envs, len(OSL_TORQUE_JOINTS), device=env.device
    )
    self._clf_joint: dict[str, torch.Tensor] = {}

  ##
  # Group weights.
  ##

  @property
  def group_weights(self) -> Weights:
    """``{channel: {group: weight}}``, a copy."""
    return copy.deepcopy(self._group_weights)

  def set_group_weights(self, weights: Mapping[str, Mapping[str, float]]) -> None:
    """Replace both channels' group weights (the runner does every iteration).

    Every channel and every group must be given, each weight finite and >= 0, and
    no kernel in use may be left with all-zero weights (:meth:`check_group_weights`).
    A refused update keeps the current weights.
    """
    self._group_weights = self.check_group_weights(weights)

  def check_group_weights(self, weights: Mapping[str, Mapping[str, float]]) -> Weights:
    """``weights`` as :meth:`set_group_weights` would store them, without storing
    them; raises ``ValueError`` where it would.

    Which kernels are in use depends on the reward's own weights, so only the term
    can tell whether a channel leaves one of them without coordinates (MimicReward's
    site kernels, say, with only ``osl_joints`` weighted). The runner checks the
    channel schedule's ends with it before training, not when the anneal gets there.
    """
    if set(weights) != set(CHANNELS):
      raise ValueError(f"Need weights for channels {CHANNELS}, got {sorted(weights)}")
    checked: Weights = {}
    for channel in CHANNELS:
      groups = weights[channel]
      if set(groups) != set(GROUPS):
        raise ValueError(
          f"Channel {channel!r} needs weights for groups {GROUPS}, got {sorted(groups)}"
        )
      values = {group: float(groups[group]) for group in GROUPS}
      bad = {g: v for g, v in values.items() if not (math.isfinite(v) and v >= 0.0)}
      if bad:
        raise ValueError(f"Channel {channel!r}: weights must be finite and >= 0: {bad}")
      for kernel in self._mixed_kernels():
        counts = self._kernel_counts[kernel]
        if sum(values[g] * counts[g] for g in GROUPS) <= 0.0:
          raise ValueError(
            f"Channel {channel!r} weighs every coordinate of the {kernel!r} kernel "
            f"at 0 ({values}); its weighted mean is undefined."
          )
      checked[channel] = values
    return checked

  def _mixed_kernels(self) -> list[str]:
    """The kernels in use that take a weighted mean over groups."""
    kernels = ["clf"] if self._clf is not None else []
    kernels += [
      k for k in (*JOINT_KERNELS, *SITE_KERNELS) if self._kernel_weights[k] != 0.0
    ]
    return kernels

  def kernel_group_counts(self) -> dict[str, dict[str, int]]:
    """Per kernel in use that mixes groups, its coordinates per group. A site counts
    once in the site kernels, which average its three axes first."""
    return {k: dict(self._kernel_counts[k]) for k in self._mixed_kernels()}

  def group_coordinate_counts(self) -> dict[str, int]:
    """The coordinates each group's weight multiplies: under the CLF alone, those of
    the channel V (84 human, 18 OSL-site, 2 OSL-joint on the OSL models). With
    MimicReward weights, summed over every kernel in use."""
    counts = dict.fromkeys(GROUPS, 0)
    for per_group in self.kernel_group_counts().values():
      for group, n in per_group.items():
        counts[group] += n
    return counts

  def effective_shares(self) -> Weights:
    """Per channel, each group's share of the channel's imitation reward.

    Each kernel's weight is split among the groups by ``weight x coordinates /
    sum`` inside it, and the shares are averaged over the kernels by their weights.
    Under the CLF alone this is exactly ``weight x count / sum`` over V's
    coordinates. The human-only kernels (root position and velocity) count fully
    as human; the qpos kernel's root-angle term is not counted.
    """
    kernels = {k: self._kernel_weights[k] for k in self._mixed_kernels()}
    if self._clf is not None and self._clf.w == 0.0:
      kernels.pop("clf")
    fixed = {k: self._kernel_weights[k] for k in FIXED_KERNELS}
    total = sum(kernels.values()) + sum(fixed.values())
    shares: Weights = {}
    for channel in CHANNELS:
      w = self._group_weights[channel]
      share = dict.fromkeys(GROUPS, 0.0)
      share["human"] += sum(fixed.values())
      for kernel, kernel_weight in kernels.items():
        counts = self._kernel_counts[kernel]
        den = sum(w[g] * counts[g] for g in GROUPS)
        for group in GROUPS:
          share[group] += kernel_weight * w[group] * counts[group] / den
      shares[channel] = {
        g: (s / total if total > 0.0 else 0.0) for g, s in share.items()
      }
    return shares

  ##
  # The reward.
  ##

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    weights: dict[str, float],
    action_out_of_bounds_coeff: float = 0.01,
    joint_torque_weights: dict[str, float] | None = None,
    clf: ClfCfg | None = None,
    channels: SplitRewardCfg | dict[str, Any] | None = None,
  ) -> torch.Tensor:
    del channels  # Read once, in __init__.
    team = super().__call__(
      env,
      command_name,
      weights,
      action_out_of_bounds_coeff,
      joint_torque_weights,
      clf,
    )
    self._compute_channels(team)
    return team

  def _stash(self, key: str) -> torch.Tensor:
    value = self._cache[key]
    if value is None:
      raise KeyError(f"mimic_reward stashed no {key!r} this step")
    return value

  def _weighted_mean(
    self, kernel: str, sums: Mapping[str, torch.Tensor], weights: Mapping[str, float]
  ) -> torch.Tensor:
    """``sum_g w_g S_g / sum_g w_g n_g`` over the groups the channel weighs. Groups
    at weight 0 are left out entirely, so a non-finite error there cannot leak."""
    counts = self._kernel_counts[kernel]
    num: torch.Tensor | None = None
    den = 0.0
    for group, total in sums.items():
      w, n = weights[group], counts[group]
      if w == 0.0 or n == 0:
        continue
      term = total * w
      num = term if num is None else num + term
      den += w * n
    if num is None or den <= 0.0:  # Prevented by set_group_weights.
      raise RuntimeError(f"No weighted coordinates in the {kernel!r} kernel.")
    return num / den

  def _compute_channels(self, team: torch.Tensor) -> None:
    totals = {c: torch.zeros_like(team) for c in CHANNELS}
    if self._mimic_active:
      sums = self._mimic_group_sums()
      for channel in CHANNELS:
        totals[channel] = self._mimic_total(sums, self._group_weights[channel])

    penalties = self._penalties()
    diagnostics: dict[str, torch.Tensor] = {}
    if self._clf is not None:
      clf_channels = self._clf_channels(diagnostics)
      for channel in CHANNELS:
        terms = clf_channels[channel]
        totals[channel] = totals[channel] + self._clf.w * terms["level"]
        # Outside the -1 floor, as in stage 1.
        penalties[channel] = penalties[channel] - self._clf.pen_w * terms["violation"]

    rewards = []
    for channel in CHANNELS:
      preclip = totals[channel] + penalties[channel]
      rewards.append(torch.nan_to_num(preclip.clamp(min=0.0), nan=0.0))
      diagnostics[f"split_reward_{channel}"] = rewards[-1]
      diagnostics[f"split_preclip_{channel}"] = preclip
      diagnostics[f"split_clipped_{channel}"] = (preclip < 0.0).float()
      diagnostics[f"split_penalty_{channel}"] = penalties[channel]
    self.channel_rewards = torch.stack(rewards, dim=-1)

    osl_err = self._stash("joint_q_err").index_select(-1, self._osl_joint_ids).abs()
    diagnostics["split_err_osl_knee"] = osl_err[:, 0]
    diagnostics["split_err_osl_ankle"] = osl_err[:, 1]
    self.components.update(diagnostics)

  ##
  # MimicReward kernels.
  ##

  def _mimic_group_sums(self) -> dict[str, dict[str, torch.Tensor]]:
    """Per active kernel that mixes groups, the group sums of its squared errors
    (per-site axis means for the site kernels), shared by both channels."""
    kw = self._kernel_weights
    sums: dict[str, dict[str, torch.Tensor]] = {}
    if kw["qpos"] != 0.0:
      sums["qpos"] = _group_sums(self._stash("qpos_err").square(), self._qpos_index)
    if kw["qvel"] != 0.0:
      sums["qvel"] = _group_sums(self._stash("qvel_err").square(), self._qvel_index)
    if kw["rpos"] != 0.0:
      per_site = self._stash("rpos_err").square().mean(-1)
      sums["rpos"] = _group_sums(per_site, self._site_index)
    if kw["rquat"] != 0.0:
      per_site = self._stash("rangles_err").square().mean(-1)
      sums["rquat"] = _group_sums(per_site, self._site_index)
    if kw["rvel_rot"] != 0.0:
      rvel_sq = self._stash("rvel_err").square()
      sums["rvel_rot"] = _group_sums(rvel_sq[..., :3].mean(-1), self._site_index)
      sums["rvel_lin"] = _group_sums(rvel_sq[..., 3:].mean(-1), self._site_index)
    return sums

  def _mimic_total(
    self, sums: Mapping[str, Mapping[str, torch.Tensor]], weights: Mapping[str, float]
  ) -> torch.Tensor:
    """MimicReward's weighted kernel sum for one channel.

    The qpos kernel's root-angle term (added unsquared in stage 1) scales with the
    human weight relative to the kernel's mean coordinate weight, which is 1 at
    uniform weights; the root position and velocity kernels are human-only and
    keep their stage-1 values.
    """
    kw, k = self._kernel_weights, MIMIC_EXPS
    terms: list[torch.Tensor] = []
    if "qpos" in sums:
      counts = self._kernel_counts["qpos"]
      w_sum = sum(weights[g] * counts[g] for g in GROUPS)
      angle_weight = weights["human"] * sum(counts.values()) / w_sum
      dist = self._weighted_mean("qpos", sums["qpos"], weights)
      dist = dist + angle_weight * self._stash("quat_angle")
      terms.append(kw["qpos"] * torch.exp(-k["qpos"] * dist))
    if "qvel" in sums:
      dist = self._weighted_mean("qvel", sums["qvel"], weights)
      terms.append(kw["qvel"] * torch.exp(-k["qvel"] * dist))
    if kw["root_pos"] != 0.0:
      terms.append(kw["root_pos"] * self._stash("r_root_pos"))
    if kw["root_vel"] != 0.0:
      terms.append(kw["root_vel"] * self._stash("r_root_vel"))
    for kernel in ("rpos", "rquat"):
      if kernel in sums:
        dist = self._weighted_mean(kernel, sums[kernel], weights)
        terms.append(kw[kernel] * torch.exp(-k[kernel] * dist))
    if "rvel_rot" in sums:
      rot = self._weighted_mean("rvel_rot", sums["rvel_rot"], weights)
      lin = self._weighted_mean("rvel_lin", sums["rvel_lin"], weights)
      kernels = torch.exp(-k["rvel"] * rot) + torch.exp(-k["rvel"] * lin)
      terms.append(self._weights["rvel"] * kernels)
    total = terms[0]
    for term in terms[1:]:
      total = total + term
    return total

  ##
  # The CLF.
  ##

  def _clf_channels(
    self, diagnostics: dict[str, torch.Tensor]
  ) -> dict[str, dict[str, torch.Tensor]]:
    """Per channel, the CLF's level and violation over the weighted coordinates:
    ``V = sum_i w_i qf_i / sum_i w_i`` with ``qf = p11 e^2 + 2 p12 e ed + p22
    ed^2``, and V-dot likewise, the OSL joint group included."""
    assert self._clf is not None
    cc = self._clf_cache
    e, ed, edd, first, valid = cc["e"], cc["ed"], cc["edd"], cc["first"], cc["valid"]
    tau, scale = self._clf.tau, self._split.osl_joint_scale

    # The OSL joint group, paired as the CLF's own groups: ed = tau * d/dt e.
    e_j = self._stash("joint_q_err").index_select(-1, self._osl_joint_ids) / scale
    ed_j = self._stash("joint_v_err").index_select(-1, self._osl_joint_ids)
    ed_j = ed_j * (tau / scale)
    edd_j = torch.where(
      first[:, None],
      torch.zeros_like(ed_j),
      (ed_j - self._edj_prev) * (tau / self._clf_dt),
    )
    self._edj_prev = ed_j.detach().clone()
    self._clf_joint = {"e": e_j, "ed": ed_j, "edd": edd_j}

    p11, p12, p22 = self._p11, self._p12, self._p22

    def qform(e_: torch.Tensor, ed_: torch.Tensor) -> torch.Tensor:
      return p11 * e_.square() + (2.0 * p12) * e_ * ed_ + p22 * ed_.square()

    def qform_rate(
      e_: torch.Tensor, ed_: torch.Tensor, edd_: torch.Tensor
    ) -> torch.Tensor:
      return ed_ * (p11 * e_ + p12 * ed_) + edd_ * (p12 * e_ + p22 * ed_)

    q_sums = _group_sums(qform(e, ed), self._clf_index)
    q_sums["osl_joints"] = qform(e_j, ed_j).sum(-1)
    d_sums = _group_sums(qform_rate(e, ed, edd), self._clf_index)
    d_sums["osl_joints"] = qform_rate(e_j, ed_j, edd_j).sum(-1)

    out: dict[str, dict[str, torch.Tensor]] = {}
    for channel in CHANNELS:
      weights = self._group_weights[channel]
      v = self._weighted_mean("clf", q_sums, weights).clamp(min=0.0)
      vdot = 2.0 * self._weighted_mean("clf", d_sums, weights)
      violation = ((vdot + self._alpha * v - self._eps_dead) / self._viol_norm).clamp(
        0, 1
      )
      violation = violation * (~first).float() * valid
      v_norm = v / self._v_max
      level = torch.where(
        torch.isfinite(v_norm), torch.exp(-v_norm.clamp(max=5.0)), torch.zeros_like(v)
      )
      out[channel] = {"level": level, "violation": violation}
      diagnostics[f"split_V_norm_{channel}"] = v_norm
      diagnostics[f"split_level_{channel}"] = level
      diagnostics[f"split_violation_{channel}"] = violation

    counts = self._kernel_counts["clf"]
    for group in GROUPS:
      n = counts[group]
      diagnostics[f"split_qform_{group}"] = (
        q_sums[group] / (n * self._v_max)
        if n
        else torch.zeros_like(q_sums["osl_joints"])
      )
    return out

  ##
  # Penalties.
  ##

  def _penalties(self) -> dict[str, torch.Tensor]:
    """Per channel, the action penalties before the CLF violation, floored at -1."""
    if not self._split.owner_penalties:
      team = self._stash("penalty")
      return {channel: team for channel in CHANNELS}
    action = self._stash("action")
    excess = (-1.0 - action).clamp(min=0.0) + (action - 1.0).clamp(min=0.0)
    squared = excess.square()
    qfrc = self._cache["qfrc"]
    torque = None if qfrc is None else self._torque_weights * qfrc.square()
    out: dict[str, torch.Tensor] = {}
    for agent in AGENTS:
      rows = squared.index_select(-1, self._action_rows[agent]).sum(-1)
      penalty = -self._oob_coeff * rows / self._action_dims[agent]
      index = self._torque_index[agent]
      if torque is not None and index.numel():
        penalty = penalty - torque.index_select(-1, index).sum(-1)
      out[agent] = penalty.clamp(min=-1.0)
    return out
