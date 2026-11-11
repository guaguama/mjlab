"""A twist command over a looping gait library, serving a musclemimic-style reference.

The library is a folder of clipper ``--format mjlab`` clips at the control rate. Each
walking clip is one stride that loops, with phase 0 at the OSL foot's touchdown, a
``twist`` label ``[vx, vy, wz]`` (forward = the root's body -y, vy left, wz CCW) and
the stride's planar displacement ``stride_se2``. ``idle.npz`` holds the standing
pose with a zero twist.

The command follows sesteban's G1 unicycle task (``jog_unicycle`` over
``crawling_fwd_blending``):

* **Sampling.** A per-env timer draws a mode: idle for ``rel_static_envs``,
  otherwise the unicycle modes (pivot, forward or backward arc). Then either
  (``sampling="library"``) a clip of that mode uniformly at random, commanding its
  label exactly, or (``"continuous"``, sesteban's sampler) a continuous twist from
  the mode's ranges, snapped to the nearest clip under a weighted L2 metric. An arc's
  speed and yaw rate come from one box, or from a union of boxes when the library's
  grid is not a rectangle. A command that snaps to the idle clip is zeroed, so idle,
  zero twist and a blanked phase always coincide. At load, the library must hold
  every mode the sampler draws, and in continuous mode every commandable twist must
  snap within ``max_snap_error`` of its clip, so a library that does not cover the
  sampler fails loudly instead of silently serving its edge clips.
* **Twist cap.** ``set_twist_limit`` (or ``max_twist``) restricts draws to clips
  under a speed and yaw-rate cap, which a curriculum can raise during training
  (:class:`~.curriculums.twist_limit_curriculum`).
* **Switching.** The new clip keeps the stride phase (``frame / T``). Over
  ``blend_time_s`` the reference fades from the outgoing clip, read at the shared
  phase, to the new one. Positions lerp, orientations nlerp, and velocities lerp plus
  ``alpha_dot * (new - old)``, so a served velocity is the derivative of its served
  position. The outgoing clip is time-warped onto the new period, so its velocities
  scale with the period ratio.
* **Episode starts.** A sampled twist, a uniformly random phase and the exact
  reference state, as musclemimic starts. No noise.

Two mechanisms replace the stock tracking command's absolute root target, which
looping clips break (their root jumps back one stride at every loop):

* **Unrolling.** Each env holds a planar anchor (x, y, yaw) for its clip, composed
  with the clip's ``stride_se2`` at every loop. The served root and sites then advance
  continuously, stride after stride. This is exact; sesteban integrates the reference
  velocity instead.
* **Re-basing.** At every timer resample the anchors move so that the served root's
  planar pose equals the robot's. This is sesteban's egocentric target: drift is
  forgiven at each command change, never in between. Height, roll and pitch are never
  re-based.

Timing is :class:`~mjlab.tasks.osl_mimic.mdp.MimicMotionCommand`'s: ``frame`` is the
frame the last observation was built at, and rewards compare with ``reference(1)``.

For play (sesteban's ``play.py`` additions), the twist can be pinned for every env
(``fixed_twist``, ``set_fixed_twist``; the viewer's keyboard and slider controls go
through it), and with ``debug_vis`` on the viewer draws the served reference as a
ghost of the robot model.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  SOCKET_JOINTS,
  Socket,
  get_spec,
)
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.osl_mimic.mdp import (
  MIMIC_BODIES,
  ROOT_BODY,
  MimicLookahead,
  MimicReference,
  pad_clip_joints,
  relative_site_quantities,
  relative_site_velocities,
)
from mjlab.tasks.tracking.config.osl.mdp import check_motion_fps
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)

if TYPE_CHECKING:
  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

Frame = dict[str, torch.Tensor]
"""Reference fields of one frame per env, before :class:`MimicReference` is built."""

TwistBox = tuple[tuple[float, float], tuple[float, float], float]
"""One sampling box of the arc modes: ``((vx_lo, vx_hi), (wz_lo, wz_hi), share)``."""

# Fields that interpolate linearly, and each velocity with the position it is the
# derivative of (for the blend's ``alpha_dot`` term).
_POSITIONS = ("root_pos", "joint_pos", "site_pos", "rpos", "rangles")
_VELOCITY_OF = {
  "root_lin_vel": "root_pos",
  "joint_vel": "joint_pos",
  "v_rel": "rpos",
  "omega_rel": "rangles",
}
_VELOCITIES = (*_VELOCITY_OF, "root_ang_vel_w", "rvel")

GHOST_GEOM_GROUP = 0
"""The geom group the reference ghost shows: the visible anatomy and the socket and
OSL meshes. All but the foot shell collide, so the stock tracking ghost's filter
(hide every colliding geom) would drop them. Wrapping geometry and the collision
proxies sit in groups 3-5."""
GUI_TWIST_STEP = 0.05
"""Slider step of the viewer's twist controls: the unicycle library's finest grid."""


@dataclass(frozen=True)
class _Ghost:
  """The reference ghost: a recoloured robot model and where the reference goes."""

  model: mujoco.MjModel
  root_adr: int
  """qpos address of the free joint."""
  qpos_adr: np.ndarray
  """qpos address of each mapped joint."""
  joint_cols: torch.Tensor
  """The reference ``joint_pos`` column of each mapped joint."""


def _as_twist(twist: Sequence[float] | None) -> tuple[float, float, float] | None:
  if twist is None:
    return None
  if len(twist) != 3:
    raise ValueError(f"A twist is (vx, vy, wz); got {tuple(twist)}.")
  return float(twist[0]), float(twist[1]), float(twist[2])


##
# Planar (SE(2)) poses: (..., 3) tensors (x, y, yaw).
##


def se2_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """``a ∘ b``: apply ``b``, then ``a``."""
  c, s = torch.cos(a[..., 2]), torch.sin(a[..., 2])
  x = a[..., 0] + c * b[..., 0] - s * b[..., 1]
  y = a[..., 1] + s * b[..., 0] + c * b[..., 1]
  return torch.stack([x, y, a[..., 2] + b[..., 2]], dim=-1)


def se2_inverse(a: torch.Tensor) -> torch.Tensor:
  c, s = torch.cos(a[..., 2]), torch.sin(a[..., 2])
  x = -(c * a[..., 0] + s * a[..., 1])
  y = s * a[..., 0] - c * a[..., 1]
  return torch.stack([x, y, -a[..., 2]], dim=-1)


def quat_yaw(quat: torch.Tensor) -> torch.Tensor:
  """Yaw of a (w, x, y, z) quaternion: the azimuth of its body x axis."""
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def planar_pose(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
  return torch.stack([pos[..., 0], pos[..., 1], quat_yaw(quat)], dim=-1)


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  half = 0.5 * yaw
  zero = torch.zeros_like(half)
  return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


def _rotate_xy(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  """Rotate the xy part of (B, ..., 3) vectors by a per-env yaw."""
  shape = (-1,) + (1,) * (v.ndim - 2)
  c, s = cos.view(shape), sin.view(shape)
  x, y = v[..., 0], v[..., 1]
  return torch.stack([c * x - s * y, s * x + c * y, v[..., 2]], dim=-1)


def _bcast(w: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
  """(B,) per-env weights broadcast against a (B, ...) tensor."""
  return w.view((-1,) + (1,) * (like.ndim - 1))


def _nlerp(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
  b = torch.where((a * b).sum(-1, keepdim=True) < 0.0, -b, b)
  q = torch.lerp(a, b, _bcast(w, a))
  return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def heading_twist(
  quat: torch.Tensor, lin_vel_w: torch.Tensor, ang_vel_w: torch.Tensor
) -> torch.Tensor:
  """(B, 3) planar twist [forward, left, yaw rate] in the root's heading frame.

  Forward is the root's body -y (the osl_fullbody convention, which the library's
  twist labels use), projected onto the ground.
  """
  fwd = torch.zeros_like(lin_vel_w)
  fwd[:, 1] = -1.0
  fwd = quat_apply(quat, fwd)
  heading = torch.atan2(fwd[:, 1], fwd[:, 0])
  c, s = torch.cos(heading), torch.sin(heading)
  forward = c * lin_vel_w[:, 0] + s * lin_vel_w[:, 1]
  left = -s * lin_vel_w[:, 0] + c * lin_vel_w[:, 1]
  return torch.stack([forward, left, ang_vel_w[:, 2]], dim=-1)


@dataclass(kw_only=True)
class GaitLibraryCommandCfg(CommandTermCfg):
  """Configuration for :class:`GaitLibraryCommand`."""

  library_dir: str = ""
  """Folder of clipper mjlab clips at the control rate (every ``*.npz`` is loaded)."""
  entity_name: str = "robot"
  site_bodies: tuple[str, ...] = MIMIC_BODIES
  lookahead_steps: int = 5
  """``GoalTrajMimic.n_step_lookahead``: frames in the goal, the current one included."""
  lookahead_stride: int = 20
  """``GoalTrajMimic.n_step_stride``, in frames."""
  resampling_time_range: tuple[float, float] = (3.0, 8.0)
  """Seconds between twist resamples (sesteban's walk and jog unicycle)."""
  blend_time_s: float = 0.4
  rebase_on_resample: bool = True
  """Move the root target onto the robot (x, y, yaw) at every timer resample."""
  twist_metric_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
  """Weights of the nearest-clip metric on (vx, vy, wz). Use
  ``span_normalized_weights`` once a library spans both vx and wz."""
  sampling: Literal["library", "continuous"] = "continuous"
  """``"library"``: a clip of the drawn mode uniformly at random, commanded at its
  label. ``"continuous"``: sesteban's sampler, a twist from the mode's ranges (the
  fields below) snapped to the nearest clip."""
  max_twist: tuple[float, float, float] | None = None
  """Initial twist cap (forward vx, backward |vx|, |wz|): draws pick only clips under
  it, and a mode with none idles. None: no cap. See ``set_twist_limit``."""
  rel_static_envs: float = 0.15
  """Fraction of draws that command a zero twist (the idle clip)."""
  rel_turn_envs: float = 0.0
  """Fraction of draws that pivot in place, ``[0, 0, ±wz]``."""
  rel_back_envs: float = 0.0
  """Fraction of arc draws that walk backward."""
  vx_fwd_range: tuple[float, float] | None = None
  """Forward arc speed (m/s). None: the library's forward span."""
  vx_bck_range: tuple[float, float] = (0.0, 0.0)
  arc_wz_range: tuple[float, float] = (0.0, 0.0)
  """Arc yaw-rate magnitude (rad/s), random sign; 0 walks straight."""
  wz_range: tuple[float, float] = (0.0, 0.0)
  """Pivot yaw-rate magnitude (rad/s), random sign."""
  fwd_arc_regions: tuple[TwistBox, ...] = ()
  """Forward arcs drawn from a union of boxes ``((vx_lo, vx_hi), (wz_lo, wz_hi),
  share)``, wz a magnitude with random sign: a box is picked with probability
  ``share``, then a point uniformly inside it. Empty: the single box
  ``vx_fwd_range`` x ``arc_wz_range``."""
  bck_arc_regions: tuple[TwistBox, ...] = ()
  """Backward arcs, as ``fwd_arc_regions``. Empty: ``vx_bck_range`` x
  ``arc_wz_range``."""
  max_snap_error: float = 0.1
  """Largest per-axis gap allowed at load between a commandable twist and the label
  of the clip it snaps to."""
  fixed_clips: bool = False
  """Validation: env ``i`` plays clip ``i % num_clips`` from phase 0 at its label,
  forever (set ``resampling_time_range`` past the episode)."""
  start_yaw_range: float = 0.0
  """Half-width (rad) of the uniform heading each episode starts at, about the
  clip's own. musclemimic's observation is world-frame (root quaternion, linear
  velocity, relative site positions, the goal), so a policy only generalizes to the
  headings it trained at; at 0, every episode starts on the clip's native heading
  and straight walking never leaves it. ``pi`` covers every heading. Ignored with
  ``fixed_clips``."""
  fixed_twist: tuple[float, float, float] | None = None
  """Pin every env to this twist ``(vx, vy, wz)``: each draw takes the clip nearest
  to it (the twist cap ignored) in place of the sampler. Library sampling commands
  that clip's label, as such a policy saw in training; continuous sampling commands
  the twist itself. None: the sampler. Play's ``--twist`` sets it; the viewer's
  twist controls change it at run time through ``set_fixed_twist``."""
  ghost_color: tuple[float, float, float, float] = (0.2, 0.65, 1.0, 0.28)
  """Colour of the reference ghost drawn when ``debug_vis`` is on, alpha included:
  osl_deploy's reference ghost, flat and untextured."""

  def build(self, env: ManagerBasedRlEnv) -> GaitLibraryCommand:
    return GaitLibraryCommand(self, env)


def span_normalized_weights(
  vx_range: tuple[float, float], wz_range: tuple[float, float]
) -> tuple[float, float, float]:
  """sesteban's metric weights: a full-span miss costs the same on vx as on wz."""
  wx = 1.0 / max(abs(vx_range[1] - vx_range[0]), 1e-6) ** 2
  wz = 1.0 / max(abs(wz_range[1] - wz_range[0]), 1e-6) ** 2
  m = max(wx, wz)
  return (wx / m, wx / m, wz / m)


TIGHT_ARC_WZ = 0.5
"""Yaw rate (rad/s) above which a forward arc counts as tight: sesteban's G1 boundary
between the arc and pivot modes, and the edge of unicycle_v001's 0.05-step band."""


def clip_group(twist: np.ndarray) -> str:
  """A clip's family from its twist label, for per-family validation metrics:
  ``idle``, ``turn`` (pivot), ``bck``, ``fwd`` or ``fwd_tight``."""
  vx, wz = float(twist[0]), float(twist[2])
  if abs(vx) < 1e-6:
    return "idle" if abs(wz) < 1e-6 else "turn"
  if vx < 0.0:
    return "bck"
  return "fwd_tight" if abs(wz) > TIGHT_ARC_WZ + 1e-6 else "fwd"


class GaitLibraryCommand(CommandTerm):
  cfg: GaitLibraryCommandCfg
  _ghost: _Ghost | None = None
  """Built on the first debug draw: only viewers and video renders need it."""

  def __init__(self, cfg: GaitLibraryCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if not cfg.library_dir:
      raise ValueError(
        "No gait library: pass --env.commands.motion.library-dir <folder of clipper "
        "--format mjlab clips at the task's control rate>."
      )
    self.robot: Entity = env.scene[cfg.entity_name]
    self._step_dt = float(env.step_dt)
    self._load_library(Path(cfg.library_dir))

    n, dev = self.num_envs, self.device
    self.clip = torch.zeros(n, dtype=torch.long, device=dev)
    self.frame = torch.zeros(n, dtype=torch.long, device=dev)
    self.twist_command = torch.zeros(n, 3, device=dev)
    self.anchor = torch.zeros(n, 3, device=dev)
    self.z_offset = torch.zeros(n, device=dev)
    self.blend_steps = max(1, round(cfg.blend_time_s / self._step_dt))
    self._alpha_rate = 1.0 / (self.blend_steps * self._step_dt)
    # The clip being faded out: its anchor, its own clock (fractional frame, frames
    # per step), and the steps into the fade (blend_steps: no blend running).
    self.clip_prev = torch.zeros(n, dtype=torch.long, device=dev)
    self.anchor_prev = torch.zeros(n, 3, device=dev)
    self.frame_prev = torch.zeros(n, device=dev)
    self.rate_prev = torch.ones(n, device=dev)
    self.blend_count = torch.full((n,), self.blend_steps, dtype=torch.long, device=dev)
    # World-frame xy the served root lags the lerp of the two clips by, during a blend
    # (see ``_serve``); folded into the anchor when the blend ends.
    self.catchup = torch.zeros(n, 2, device=dev)

    self._just_reset = torch.zeros(n, dtype=torch.bool, device=dev)
    self._jump_step = torch.full((n,), -10, dtype=torch.long, device=dev)
    self._episode_reset = False
    self._cache: dict[int, MimicReference] = {}

  ##
  # Loading.
  ##

  def _load_library(self, folder: Path) -> None:
    files = sorted(folder.glob("*.npz"))
    if not files:
      raise ValueError(f"No *.npz clips in {folder}.")
    body_names = list(self.robot.body_names)
    joint_names = list(self.robot.joint_names)
    root = body_names.index(ROOT_BODY)
    self.site_body_ids = [body_names.index(b) for b in self.cfg.site_bodies]
    dev = self.device

    clips: list[dict[str, torch.Tensor]] = []
    twists, strides, lengths = [], [], []
    for path in files:
      check_motion_fps(str(path), self._step_dt)
      with np.load(path) as data:
        a = {
          k: torch.tensor(data[k], dtype=torch.float32, device=dev)
          for k in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
          )
        }
        twist = np.asarray(data["twist"], dtype=np.float64).reshape(3)
        se2 = (
          np.asarray(data["stride_se2"], dtype=np.float64)
          if "stride_se2" in data.files
          else np.eye(3)
        )
      if a["body_pos_w"].shape[1] != len(body_names):
        raise ValueError(
          f"{path.name} has {a['body_pos_w'].shape[1]} bodies; the entity has "
          f"{len(body_names)}."
        )
      joint_pos, joint_vel = pad_clip_joints(
        a["joint_pos"], a["joint_vel"], joint_names
      )
      quat = a["body_quat_w"][:, root]
      site_pos = a["body_pos_w"][:, self.site_body_ids]
      site_mat = matrix_from_quat(a["body_quat_w"][:, self.site_body_ids])
      ang = a["body_ang_vel_w"][:, self.site_body_ids]
      lin = a["body_lin_vel_w"][:, self.site_body_ids]
      rpos, rangles, rvel = relative_site_quantities(site_pos, site_mat, ang, lin)
      v_rel, omega_rel = relative_site_velocities(site_mat, ang, lin)
      clip = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos": a["body_pos_w"][:, root],
        "root_quat": quat,
        "root_lin_vel": a["body_lin_vel_w"][:, root],
        "root_ang_vel_w": a["body_ang_vel_w"][:, root],
        "root_ang_vel_b": quat_apply_inverse(quat, a["body_ang_vel_w"][:, root]),
        "site_pos": site_pos,
        "rpos": rpos,
        "rangles": rangles,
        "rvel": rvel,
        "v_rel": v_rel,
        "omega_rel": omega_rel,
      }
      stride = (float(se2[0, 2]), float(se2[1, 2]), math.atan2(se2[1, 0], se2[0, 0]))
      self._check_loop(path.name, clip, stride)
      clips.append(clip)
      twists.append(twist)
      strides.append(stride)
      lengths.append(int(joint_pos.shape[0]))

    t_max = max(lengths)
    self.num_clips = len(clips)
    self.clip_names = [p.stem for p in files]
    self.num_frames = torch.tensor(lengths, dtype=torch.long, device=dev)
    self.twists = torch.tensor(np.stack(twists), dtype=torch.float32, device=dev)
    self.strides = torch.tensor(strides, dtype=torch.float32, device=dev)
    # (C, T_max, ...), short clips padded with their last frame (never read).
    self.data: dict[str, torch.Tensor] = {}
    for key in clips[0]:
      padded = []
      for clip in clips:
        x = clip[key]
        pad = x[-1:].expand(t_max - x.shape[0], *x.shape[1:])
        padded.append(torch.cat([x, pad], dim=0))
      self.data[key] = torch.stack(padded)
    self.data["root_planar"] = planar_pose(
      self.data["root_pos"], self.data["root_quat"]
    )
    self._index_library()

  def _index_library(self) -> None:
    """Everything derived from the loaded clips (``twists``, ``num_frames``): the idle
    clip, the speed span, the sampling tables, the coverage check and the pin."""
    zero = self.twists.norm(dim=-1) < 1e-6
    idle = torch.nonzero(zero).flatten()
    self.idle_clip: int | None = int(idle[0]) if idle.numel() else None
    if self.cfg.rel_static_envs > 0.0 and self.idle_clip is None:
      raise ValueError("rel_static_envs > 0 needs a zero-twist (idle) clip.")
    forward = self.twists[:, 0][self.twists[:, 0] > 1e-6]
    if self.cfg.vx_fwd_range is not None:
      self.vx_fwd_range = self.cfg.vx_fwd_range
    elif forward.numel():
      self.vx_fwd_range = (float(forward.min()), float(forward.max()))
    else:
      self.vx_fwd_range = (0.0, 0.0)
    self._metric = torch.tensor(self.cfg.twist_metric_weights, device=self.device)
    self.clip_groups = [clip_group(t) for t in self.twists.cpu().numpy()]

    # Loops any served offset can cross: lookahead plus the blend's +1 interpolation.
    reach = (self.cfg.lookahead_steps - 1) * self.cfg.lookahead_stride + 2
    self._max_wraps = max((n - 1 + reach) // n for n in self.num_frames.tolist()) + 1

    self._fwd_boxes = self.cfg.fwd_arc_regions or (
      (self.vx_fwd_range, self.cfg.arc_wz_range, 1.0),
    )
    self._bck_boxes = self.cfg.bck_arc_regions or (
      (self.cfg.vx_bck_range, self.cfg.arc_wz_range, 1.0),
    )
    self._full_boxes = {
      "fwd": self._boxes_to_tensors(self._fwd_boxes),
      "bck": self._boxes_to_tensors(self._bck_boxes),
    }
    # The clips of each sampling mode, by label: idle, pivot, forward, backward.
    vx, wz = self.twists[:, 0], self.twists[:, 2]
    self._mode_masks = (
      zero,
      (vx.abs() < 1e-6) & (wz.abs() > 1e-6),
      vx > 1e-6,
      vx < -1e-6,
    )
    self.set_twist_limit(self.cfg.max_twist)
    self._check_coverage()
    # The pin. Labels and metric are kept on the CPU as well, so viewer threads can
    # snap without touching the device; they post pin changes to the mailbox, which
    # the sim thread drains in ``_update_command``.
    self.twist_labels: np.ndarray = self.twists.cpu().numpy().astype(np.float64)
    """(C, 3) ``twists`` on the CPU, for viewer threads."""
    self._metric_np = np.asarray(self.cfg.twist_metric_weights, dtype=np.float64)
    self._fixed_twist = _as_twist(self.cfg.fixed_twist)
    self._twist_requests: deque[tuple[float, float, float] | None] = deque(maxlen=1)

  def _boxes_to_tensors(
    self, boxes: tuple[TwistBox, ...]
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(K, 2) lower and upper (vx, |wz|) corners and (K,) normalized shares."""
    for (vx_lo, vx_hi), (wz_lo, wz_hi), share in boxes:
      if vx_lo > vx_hi or wz_lo > wz_hi or wz_lo < 0.0 or share <= 0.0:
        raise ValueError(
          f"Bad twist box {((vx_lo, vx_hi), (wz_lo, wz_hi), share)}: need lo <= hi, "
          "a non-negative yaw-rate magnitude and a positive share."
        )
    lo = torch.tensor([[b[0][0], b[1][0]] for b in boxes], device=self.device)
    hi = torch.tensor([[b[0][1], b[1][1]] for b in boxes], device=self.device)
    share = torch.tensor([b[2] for b in boxes], device=self.device)
    return lo, hi, share / share.sum()

  def _commandable_twists(self, step: float = 0.01) -> torch.Tensor:
    """(M, 3) a grid of spacing ``step`` over every twist the sampler can draw."""
    cfg = self.cfg

    def box(vx: tuple[float, float], wz: tuple[float, float]) -> torch.Tensor:
      gx = torch.linspace(vx[0], vx[1], max(2, math.ceil((vx[1] - vx[0]) / step) + 1))
      gz = torch.linspace(wz[0], wz[1], max(2, math.ceil((wz[1] - wz[0]) / step) + 1))
      x, z = torch.meshgrid(gx, gz, indexing="ij")
      grid = torch.stack([x.flatten(), torch.zeros(x.numel()), z.flatten()], dim=-1)
      return torch.cat([grid, grid * torch.tensor([1.0, 1.0, -1.0])])

    parts = []
    if cfg.rel_turn_envs > 0.0:
      parts.append(box((0.0, 0.0), cfg.wz_range))
    if cfg.rel_static_envs + cfg.rel_turn_envs < 1.0:
      if cfg.rel_back_envs < 1.0:
        parts += [box(vx, wz) for vx, wz, _ in self._fwd_boxes]
      if cfg.rel_back_envs > 0.0:
        parts += [box(vx, wz) for vx, wz, _ in self._bck_boxes]
    if not parts:
      return torch.zeros(0, 3, device=self.device)
    return torch.cat(parts).to(self.device)

  def _check_coverage(self) -> None:
    """The library must hold a clip of every mode the sampler draws, and in
    continuous mode every twist the sampler can draw must snap to a clip within
    ``max_snap_error`` per axis. Outside the library's grid a command silently snaps
    to an edge clip, and the twist reward then chases a twist no clip shows. The
    twist cap is ignored here: it only ever narrows what is drawn."""
    cfg = self.cfg
    if cfg.fixed_clips:
      return
    arc = cfg.rel_static_envs + cfg.rel_turn_envs < 1.0
    drawn = {
      "pivot": (cfg.rel_turn_envs > 0.0, 1),
      "forward arc": (arc and cfg.rel_back_envs < 1.0, 2),
      "backward arc": (arc and cfg.rel_back_envs > 0.0, 3),
    }
    for name, (needed, mode) in drawn.items():
      if needed and not bool(self._mode_masks[mode].any()):
        raise ValueError(
          f"The sampler draws {name} commands, but the gait library has no {name} "
          "clip. Use a library that holds them or set that mode's fraction to 0."
        )
    if cfg.sampling == "library":
      return
    commands = self._commandable_twists()
    if commands.shape[0] == 0:
      return
    errors = []
    for chunk in commands.split(4096):
      snapped = self._snap(chunk, capped=False)
      errors.append((chunk - self.twists[snapped]).abs().amax(-1))
    error = torch.cat(errors)
    worst = int(error.argmax())
    if float(error[worst]) > cfg.max_snap_error:
      command = commands[worst]
      clip = int(self._snap(command[None], capped=False)[0])
      raise ValueError(
        f"The gait library does not cover the commanded twists: "
        f"{[round(v, 3) for v in command.tolist()]} snaps to {self.clip_names[clip]} "
        f"(label {[round(v, 3) for v in self.twists[clip].tolist()]}), "
        f"{float(error[worst]):.3f} off on one axis (max_snap_error "
        f"{cfg.max_snap_error}). Read the sampler's ranges off the library."
      )

  def set_twist_limit(self, limit: tuple[float, float, float] | None) -> None:
    """Cap the clips a draw may pick: forward vx <= ``limit[0]``, backward |vx| <=
    ``limit[1]`` and |wz| <= ``limit[2]``, the idle clip always included. None lifts
    the cap. Takes effect at the next draw; clips already playing are kept.

    Library sampling draws only clips under the cap. Continuous sampling draws from
    the sampling boxes and pivot range cut down to the cap, each box keeping its
    share times the fraction of it left, and snaps among the clips under the cap.
    Either way a mode left with nothing idles, so a cap never piles draws onto its
    edge or turns one mode into another.
    """
    vx, wz = self.twists[:, 0], self.twists[:, 2]
    eligible = torch.ones(self.num_clips, dtype=torch.bool, device=self.device)
    capped = (
      None if limit is None else (float(limit[0]), float(limit[1]), float(limit[2]))
    )
    if capped is not None:
      fwd, bck, yaw = capped
      eligible = (vx <= fwd + 1e-6) & (-vx <= bck + 1e-6) & (wz.abs() <= yaw + 1e-6)
    if self.idle_clip is not None:
      eligible[self.idle_clip] = True
    self.twist_limit = capped
    self._eligible = eligible
    self._box_tensors = {
      kind: self._cap_boxes(boxes, capped) for kind, boxes in self._full_boxes.items()
    }
    lo, hi = self.cfg.wz_range
    if capped is not None:
      hi = min(hi, capped[2])
    self._pivot_range = (lo, hi) if lo <= hi else None
    # One row of eligible clips per mode, and a last row of every eligible clip (the
    # fallback of a library without an idle clip), zero-padded.
    tables = [torch.nonzero(m & eligible).flatten() for m in self._mode_masks]
    tables.append(torch.nonzero(eligible).flatten())
    width = max(1, max(t.numel() for t in tables))
    self._mode_clips = torch.zeros(
      len(tables), width, dtype=torch.long, device=self.device
    )
    for row, table in enumerate(tables):
      self._mode_clips[row, : table.numel()] = table
    self._mode_counts = torch.tensor(
      [t.numel() for t in tables], dtype=torch.long, device=self.device
    )

  @staticmethod
  def _cap_boxes(
    boxes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    limit: tuple[float, float, float] | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Sampling boxes cut down to a twist cap, each share scaled by the fraction of
    the box left (an axis of zero width counts as kept while inside the cap). None
    when nothing is left."""
    lo, hi, share = boxes
    if limit is None:
      return boxes
    fwd, bck, yaw = limit
    cap_lo = torch.tensor([-bck, 0.0], device=lo.device)
    cap_hi = torch.tensor([fwd, yaw], device=lo.device)
    new_lo, new_hi = torch.maximum(lo, cap_lo), torch.minimum(hi, cap_hi)
    width, new_width = hi - lo, new_hi - new_lo
    kept = torch.where(
      width > 0, new_width.clamp(min=0) / width.clamp(min=1e-12), torch.ones_like(width)
    ).prod(-1) * (new_width >= 0).all(-1)
    new_share = share * kept
    if float(new_share.sum()) <= 0.0:
      return None
    return new_lo, new_hi, new_share / new_share.sum()

  def _check_loop(
    self, name: str, clip: dict[str, torch.Tensor], stride: tuple[float, float, float]
  ) -> None:
    """The stride transform must carry frame 0 onto the frame after the last one."""
    dt = self._step_dt
    nxt_pos = clip["root_pos"][-1] + clip["root_lin_vel"][-1] * dt
    nxt_yaw = quat_yaw(clip["root_quat"][-1]) + clip["root_ang_vel_w"][-1, 2] * dt
    s = torch.tensor(stride, dtype=torch.float32, device=self.device)
    first = planar_pose(clip["root_pos"][0], clip["root_quat"][0])
    unrolled = se2_compose(s, first)
    pos_err = float((unrolled[:2] - nxt_pos[:2]).norm())
    yaw_err = abs(float(torch.remainder(unrolled[2] - nxt_yaw + math.pi, 2 * math.pi)))
    yaw_err = abs(yaw_err - math.pi)
    if pos_err > 0.01 or yaw_err > 0.02:
      raise ValueError(
        f"{name}: stride_se2 does not close the loop (position {pos_err * 1e3:.1f} mm, "
        f"yaw {math.degrees(yaw_err):.2f} deg). Expected world-frame "
        "P(frame T) = stride_se2 ∘ P(frame 0)."
      )

  ##
  # Serving the reference.
  ##

  def _gather(
    self,
    clip: torch.Tensor,
    f: torch.Tensor,
    anchor: torch.Tensor,
    z: torch.Tensor | None = None,
  ) -> Frame:
    """The reference of ``clip`` at unbounded frame ``f >= 0``, loops unrolled, under
    ``anchor`` and raised by ``z``. Every argument is per env (or per selected env)."""
    n = self.num_frames[clip]
    wraps = torch.div(f, n, rounding_mode="floor")
    g = f - wraps * n
    a = self._unroll(anchor, clip, wraps)
    cos, sin = torch.cos(a[:, 2]), torch.sin(a[:, 2])
    d = self.data
    root_pos = _rotate_xy(d["root_pos"][clip, g], cos, sin)
    site_pos = _rotate_xy(d["site_pos"][clip, g], cos, sin)
    height = torch.zeros_like(a[:, :1]) if z is None else z[:, None]
    shift = torch.cat([a[:, :2], height], dim=-1)
    return {
      "root_pos": root_pos + shift,
      "root_quat": quat_mul(_yaw_quat(a[:, 2]), d["root_quat"][clip, g]),
      "root_lin_vel": _rotate_xy(d["root_lin_vel"][clip, g], cos, sin),
      "root_ang_vel_w": _rotate_xy(d["root_ang_vel_w"][clip, g], cos, sin),
      "joint_pos": d["joint_pos"][clip, g],
      "joint_vel": d["joint_vel"][clip, g],
      "site_pos": site_pos + shift[:, None, :],
      "rpos": _rotate_xy(d["rpos"][clip, g], cos, sin),
      "rangles": d["rangles"][clip, g],
      "rvel": d["rvel"][clip, g],
      "v_rel": _rotate_xy(d["v_rel"][clip, g], cos, sin),
      "omega_rel": d["omega_rel"][clip, g],
    }

  def _unroll(
    self, anchor: torch.Tensor, clip: torch.Tensor, wraps: torch.Tensor
  ) -> torch.Tensor:
    """``anchor ∘ stride**wraps``: the anchor of the ``wraps``-th loop."""
    stride = self.strides[clip]
    a = anchor
    for i in range(self._max_wraps):
      a = torch.where((wraps > i)[:, None], se2_compose(a, stride), a)
    return a

  def _root_xy(
    self, clip: torch.Tensor, x: torch.Tensor, anchor: torch.Tensor
  ) -> torch.Tensor:
    """(B, 2) the anchored root xy of ``clip`` at unbounded fractional frame ``x``."""

    def at(f: torch.Tensor) -> torch.Tensor:
      n = self.num_frames[clip]
      wraps = torch.div(f, n, rounding_mode="floor")
      a = self._unroll(anchor, clip, wraps)
      p = self.data["root_pos"][clip, f - wraps * n]
      c, s = torch.cos(a[:, 2]), torch.sin(a[:, 2])
      return (
        torch.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], -1)
        + (a[:, :2])
      )

    lo = x.floor().long()
    return torch.lerp(at(lo), at(lo + 1), (x - lo.float())[:, None])

  @property
  def alpha(self) -> torch.Tensor:
    """(B,) the current clip's blend weight: 1 when no blend is running."""
    return self.blend_count.float() / self.blend_steps

  def _outgoing(self, ids: torch.Tensor | slice, offset: int) -> Frame:
    """The outgoing clip ``offset`` steps ahead on its own clock, with velocities
    scaled to that clock's rate."""
    prev = self.clip_prev[ids]
    rate = self.rate_prev[ids]
    x = self.frame_prev[ids] + offset * rate
    lo = x.floor().long()
    frac = x - lo.float()
    anchor, z = self.anchor_prev[ids], self.z_offset[ids]
    a, b = self._gather(prev, lo, anchor, z), self._gather(prev, lo + 1, anchor, z)
    out = {k: torch.lerp(a[k], b[k], _bcast(frac, a[k])) for k in a if k != "root_quat"}
    out["root_quat"] = _nlerp(a["root_quat"], b["root_quat"], frac)
    for k in _VELOCITIES:
      out[k] = out[k] * _bcast(rate, out[k])
    return out

  def _serve(self, offset: int, ids: torch.Tensor | slice = slice(None)) -> Frame:
    clip = self.clip[ids]
    f = self.frame[ids] + offset
    cur = self._gather(clip, f, self.anchor[ids], self.z_offset[ids])
    count = self.blend_count[ids]
    alpha = (count + offset).clamp(max=self.blend_steps).float() / self.blend_steps
    blending = alpha < 1.0
    old = self._outgoing(ids, offset)
    alpha_dot = torch.where(
      blending, torch.full_like(alpha, self._alpha_rate), torch.zeros_like(alpha)
    )
    mixed: Frame = {}
    for k in _POSITIONS:
      mixed[k] = torch.lerp(old[k], cur[k], _bcast(alpha, cur[k]))
    mixed["root_quat"] = _nlerp(old["root_quat"], cur["root_quat"], alpha)
    for vel, pos in _VELOCITY_OF.items():
      mixed[vel] = torch.lerp(old[vel], cur[vel], _bcast(alpha, cur[vel])) + _bcast(
        alpha_dot, cur[vel]
      ) * (cur[pos] - old[pos])
    # The two clips' roots drift apart in xy at their speed difference, so a lerp of
    # absolute positions would overshoot to catch up. The served root instead moves at
    # the lerp of their velocities: its xy trails the lerp by catchup, (1/T_blend) times
    # the integral of the gap, and its xy velocity loses the alpha_dot term.
    mixed["root_lin_vel"][:, :2] -= alpha_dot[:, None] * (
      cur["root_pos"][:, :2] - old["root_pos"][:, :2]
    )
    q_old = old["root_quat"]
    q_cur = torch.where(
      (q_old * cur["root_quat"]).sum(-1, keepdim=True) < 0.0,
      -cur["root_quat"],
      cur["root_quat"],
    )
    turn = axis_angle_from_quat(quat_mul(q_cur, quat_conjugate(q_old)))
    mixed["root_ang_vel_w"] = (
      torch.lerp(old["root_ang_vel_w"], cur["root_ang_vel_w"], alpha[:, None])
      + alpha_dot[:, None] * turn
    )
    mixed["rvel"] = torch.lerp(old["rvel"], cur["rvel"], _bcast(alpha, cur["rvel"]))
    served = {
      k: torch.where(_bcast(blending, v), mixed[k], v) if k in mixed else v
      for k, v in cur.items()
    }
    # Catch-up at this offset: the state's value, plus the step to come (rectangle
    # rule, as ``_update_command`` integrates it) while the blend is still running.
    gap = cur["root_pos"][:, :2] - old["root_pos"][:, :2]
    running = (count < self.blend_steps) & (offset > 0)
    catchup = self.catchup[ids] + (
      running.float()[:, None] * (offset * self._step_dt * self._alpha_rate) * gap
    )
    served["root_pos"] = served["root_pos"].clone()
    served["root_pos"][:, :2] -= catchup
    served["site_pos"] = served["site_pos"].clone()
    served["site_pos"][..., :2] -= catchup[:, None, :]
    return served

  @staticmethod
  def _as_reference(f: Frame) -> MimicReference:
    return MimicReference(
      root_pos_w=f["root_pos"],
      root_quat=f["root_quat"],
      root_lin_vel_w=f["root_lin_vel"],
      root_ang_vel_b=quat_apply_inverse(f["root_quat"], f["root_ang_vel_w"]),
      joint_pos=f["joint_pos"],
      joint_vel=f["joint_vel"],
      site_pos_w=f["site_pos"],
      rpos=f["rpos"],
      rangles=f["rangles"],
      rvel=f["rvel"],
      v_rel=f["v_rel"],
      omega_rel=f["omega_rel"],
    )

  ##
  # The imitation terms' interface (``osl_mimic.mdp.reference``).
  ##

  def reference(self, offset: int) -> MimicReference:
    if offset not in self._cache:
      self._cache[offset] = self._as_reference(self._serve(offset))
    return self._cache[offset]

  def lookahead(self) -> MimicLookahead:
    """The goal's lookahead, on the current clip (a blend is not looked through)."""
    now = self._gather(self.clip, self.frame, self.anchor)
    deltas, qvel_deltas, rpos = [], [], []
    qvel_now = self._root_qvel(now)
    for i in range(1, self.cfg.lookahead_steps):
      future = self._gather(
        self.clip, self.frame + i * self.cfg.lookahead_stride, self.anchor
      )
      deltas.append(future["root_pos"] - now["root_pos"])
      qvel_deltas.append(self._root_qvel(future) - qvel_now)
      rpos.append(future["rpos"])
    return MimicLookahead(
      root_pos_delta=torch.stack(deltas, dim=1),
      root_qvel_delta=torch.stack(qvel_deltas, dim=1),
      rpos=torch.stack(rpos, dim=1),
    )

  @staticmethod
  def _root_qvel(f: Frame) -> torch.Tensor:
    ang_b = quat_apply_inverse(f["root_quat"], f["root_ang_vel_w"])
    return torch.cat([f["root_lin_vel"], ang_b], dim=-1)

  @property
  def phase(self) -> torch.Tensor:
    """Stride phase in [0, 1): ``frame / T`` of the current clip."""
    return self.frame.float() / self.num_frames[self.clip].float()

  @property
  def is_idle(self) -> torch.Tensor:
    if self.idle_clip is None:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return self.clip == self.idle_clip

  @property
  def reference_jumped(self) -> torch.Tensor:
    return (self._env.common_step_counter - self._jump_step) == 1

  def mark_discontinuity(self, env_ids: torch.Tensor | slice) -> None:
    """Flag a jump of the robot's state, such as a push, made after this step's
    reward. The next reward then sees ``reference_jumped`` and the CLF skips the
    finite difference across it, as it does across a reference jump."""
    self._jump_step[env_ids] = self._env.common_step_counter

  @property
  def root_qpos(self) -> torch.Tensor:
    return self.robot.data.data.qpos[:, self.robot.indexing.free_joint_q_adr]

  @property
  def root_qvel(self) -> torch.Tensor:
    return self.robot.data.data.qvel[:, self.robot.indexing.free_joint_v_adr]

  def site_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = self.robot.data
    ids = self.site_body_ids
    return (
      data.body_link_pos_w[:, ids],
      matrix_from_quat(data.body_link_quat_w[:, ids]),
      data.body_link_ang_vel_w[:, ids],
      data.body_link_lin_vel_w[:, ids],
    )

  @property
  def command(self) -> torch.Tensor:
    return self.twist_command

  ##
  # Twist sampling (sesteban's unicycle sampler) and snapping.
  ##

  def _uniform(self, bounds: tuple[float, float], n: int) -> torch.Tensor:
    return torch.empty(n, device=self.device).uniform_(bounds[0], bounds[1])

  def _sign(self, n: int) -> torch.Tensor:
    return torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)

  def _sample_boxes(self, kind: str, n: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """(n,) vx and (n,) yaw-rate magnitudes, uniform over the ``kind`` boxes'
    union weighted by their shares, and whether any box is left under the cap."""
    boxes = self._box_tensors[kind]
    if boxes is None:
      zero = torch.zeros(n, device=self.device)
      return zero, zero, False
    lo, hi, share = boxes
    if share.numel() == 1:
      idx = torch.zeros(n, dtype=torch.long, device=self.device)
    else:
      idx = torch.multinomial(share, n, replacement=True)
    point = lo[idx] + (hi[idx] - lo[idx]) * torch.rand(n, 2, device=self.device)
    return point[:, 0], point[:, 1], True

  def _sample_twist(self, n: int) -> torch.Tensor:
    """sesteban's unicycle draw. A mode left empty by the twist cap draws a zero
    twist, which snaps to the idle clip."""
    cfg = self.cfg
    twist = torch.zeros(n, 3, device=self.device)
    mode = torch.rand(n, device=self.device)
    is_turn = (mode >= cfg.rel_static_envs) & (
      mode < cfg.rel_static_envs + cfg.rel_turn_envs
    )
    is_arc = mode >= cfg.rel_static_envs + cfg.rel_turn_envs
    back = torch.rand(n, device=self.device) < cfg.rel_back_envs
    fwd_vx, fwd_wz, fwd_ok = self._sample_boxes("fwd", n)
    bck_vx, bck_wz, bck_ok = self._sample_boxes("bck", n)
    vx = torch.where(back, bck_vx, fwd_vx)
    arc_wz = torch.where(back, bck_wz, fwd_wz) * self._sign(n)
    is_arc = is_arc & ((back & bck_ok) | (~back & fwd_ok))
    if self._pivot_range is None:
      pivot_wz = torch.zeros(n, device=self.device)
      is_turn = torch.zeros_like(is_turn)
    else:
      pivot_wz = self._uniform(self._pivot_range, n) * self._sign(n)
    twist[:, 0] = torch.where(is_arc, vx, twist[:, 0])
    twist[:, 2] = torch.where(is_arc, arc_wz, torch.where(is_turn, pivot_wz, 0.0))
    return twist

  def _snap(self, twist: torch.Tensor, capped: bool = True) -> torch.Tensor:
    """(n,) the nearest clip to each twist, among the clips under the twist cap."""
    dist = ((self.twists[None] - twist[:, None]) ** 2 * self._metric).sum(-1)
    if capped:
      dist = dist.masked_fill(~self._eligible[None], float("inf"))
    return torch.argmin(dist, dim=-1)

  def _draw_modes(self, n: int) -> torch.Tensor:
    """(n,) sampling modes: 0 idle, 1 pivot, 2 forward arc, 3 backward arc."""
    cfg = self.cfg
    u = torch.rand(n, device=self.device)
    back = torch.rand(n, device=self.device) < cfg.rel_back_envs
    mode = torch.zeros(n, dtype=torch.long, device=self.device)
    mode = torch.where(u >= cfg.rel_static_envs, 1, mode)
    arc = u >= cfg.rel_static_envs + cfg.rel_turn_envs
    return torch.where(arc, torch.where(back, 3, 2), mode)

  def _draw_library(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A clip of each drawn mode, uniformly among those under the twist cap, and its
    label as the command. A mode with no such clip idles (or, in a library without
    an idle clip, takes any clip under the cap)."""
    mode = self._draw_modes(n)
    fallback = 0 if self.idle_clip is not None else len(self._mode_masks)
    mode = torch.where(self._mode_counts[mode] == 0, fallback, mode)
    count = self._mode_counts[mode].clamp(min=1)
    pick = (torch.rand(n, device=self.device) * count).long()
    clip = self._mode_clips[mode, torch.minimum(pick, count - 1)]
    return self.twists[clip].clone(), clip

  def _draw_pinned(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The pinned twist's clip for every draw, commanded as ``fixed_twist`` says."""
    assert self._fixed_twist is not None
    index = self.nearest_clip(self._fixed_twist)
    clip = torch.full((n,), index, dtype=torch.long, device=self.device)
    if self.cfg.sampling == "library":
      return self.twists[clip].clone(), clip
    twist = torch.tensor(self._fixed_twist, device=self.device).repeat(n, 1)
    return twist, clip

  def _draw(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, 3) commanded twists and (n,) the clips that play them."""
    if self._fixed_twist is not None:
      return self._draw_pinned(n)
    if self.cfg.sampling == "library":
      return self._draw_library(n)
    twist = self._sample_twist(n)
    return twist, self._snap(twist)

  ##
  # Pinning the twist (play's --twist and the viewer's twist controls).
  ##

  @property
  def fixed_twist(self) -> tuple[float, float, float] | None:
    return self._fixed_twist

  @property
  def twist_bounds(self) -> tuple[list[float], list[float]]:
    """Per-axis lower and upper bounds of the library's twist labels."""
    lo, hi = self.twist_labels.min(0), self.twist_labels.max(0)
    # Labels are float32: round off the noise (-0.800000011920929).
    return np.round(lo, 6).tolist(), np.round(hi, 6).tolist()

  def nearest_clip(self, twist: Sequence[float]) -> int:
    """The clip a pin of ``twist`` plays: the nearest label under the twist metric,
    the twist cap ignored. CPU only, so viewer threads may call it."""
    gap = self.twist_labels - np.asarray(twist, dtype=np.float64)
    return int(np.argmin((gap**2 * self._metric_np).sum(-1)))

  def set_fixed_twist(self, twist: Sequence[float] | None) -> None:
    """Pin every env's twist from the next step on, or release the pin with None.

    The change takes the timer-resample path for every env: the pinned clip at the
    shared phase, blended in, the target re-based on the robot; no episode reset.
    Later resamples keep the pin until it is released. Only a request is posted
    here, so viewer and GUI threads may call it.
    """
    self._twist_requests.append(_as_twist(twist))

  ##
  # CommandTerm interface.
  ##

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    self._episode_reset = True
    try:
      return super().reset(env_ids)
    finally:
      self._episode_reset = False

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    if self._episode_reset:
      self._start_episode(env_ids)
    else:
      self._switch(env_ids)
    self._cache.clear()

  def _start_episode(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if self.cfg.fixed_clips:
      clip = env_ids % self.num_clips
      twist = self.twists[clip].clone()
      frame = torch.zeros(n, dtype=torch.long, device=self.device)
    else:
      twist, clip = self._draw(n)
      length = self.num_frames[clip]
      frame = (torch.rand(n, device=self.device) * length).long().clamp(max=length - 1)
    if self.idle_clip is not None:
      twist[clip == self.idle_clip] = 0.0

    origins = self._env.scene.env_origins[env_ids]
    start = self.data["root_pos"][clip, frame]
    # The anchor (x, y, yaw) puts the start frame's root on the env origin, turned
    # by the start heading.
    anchor = torch.zeros(n, 3, device=self.device)
    if self.cfg.start_yaw_range > 0.0 and not self.cfg.fixed_clips:
      anchor[:, 2] = (2.0 * torch.rand(n, device=self.device) - 1.0) * (
        self.cfg.start_yaw_range
      )
    turned = _rotate_xy(start, torch.cos(anchor[:, 2]), torch.sin(anchor[:, 2]))
    anchor[:, :2] = origins[:, :2] - turned[:, :2]
    self.clip[env_ids] = clip
    self.frame[env_ids] = frame
    self.twist_command[env_ids] = twist
    self.anchor[env_ids] = anchor
    self.z_offset[env_ids] = origins[:, 2]
    self.clip_prev[env_ids] = clip
    self.anchor_prev[env_ids] = anchor
    self.frame_prev[env_ids] = frame.float()
    self.rate_prev[env_ids] = 1.0
    self.blend_count[env_ids] = self.blend_steps
    self.catchup[env_ids] = 0.0
    self._jump_step[env_ids] = -10
    self._just_reset[env_ids] = True

    ref = self._serve(0, env_ids)
    root_state = torch.cat(
      [ref["root_pos"], ref["root_quat"], ref["root_lin_vel"], ref["root_ang_vel_w"]],
      dim=-1,
    )
    self.robot.write_joint_state_to_sim(
      ref["joint_pos"], ref["joint_vel"], env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    # The reset zeroes the joint targets, and the env writes them to the actuators
    # once more before the policy acts. With actuator delays, that write fills the
    # freshly reset delay history, so the first delayed substeps would drive the
    # servos toward the all-zero pose. Holding the start pose instead is what a
    # servo running before the episode would do.
    self.robot.set_joint_position_target(ref["joint_pos"], env_ids=env_ids)

  def _switch(self, env_ids: torch.Tensor) -> None:
    """Timer resample: new twist, phase-preserving clip switch, blend, re-base."""
    if self.cfg.fixed_clips:
      return
    old_clip, old_frame = self.clip[env_ids], self.frame[env_ids]
    if self.cfg.rebase_on_resample:
      data = self.robot.data
      target = planar_pose(
        data.root_link_pos_w[env_ids], data.root_link_quat_w[env_ids]
      )
    else:
      served = self._serve(0, env_ids)
      target = planar_pose(served["root_pos"], served["root_quat"])

    twist, clip = self._draw(len(env_ids))
    if self.idle_clip is not None:
      twist[clip == self.idle_clip] = 0.0
    n_old, n_new = self.num_frames[old_clip], self.num_frames[clip]
    same = clip == old_clip
    raw = torch.round(old_frame.float() * n_new.float() / n_old.float()).long()
    frame = torch.where(same, old_frame, raw % n_new)

    # The outgoing clip's clock. Into a gait: sesteban's shared phase, the old clip
    # time-warped onto the new period. Into idle, whose 50 frames are no period: the
    # old clip keeps its own frame and speed, so the reference slows to a stand
    # instead of sprinting through a warped stride.
    into_gait = torch.ones_like(same)
    if self.idle_clip is not None:
      into_gait = clip != self.idle_clip
    warp = n_old.float() / n_new.float()
    x = torch.where(into_gait, frame.float() * warp, old_frame.float())
    rate = torch.where(into_gait, warp, torch.ones_like(warp))

    # Anchors that put both clips' served root at the target pose right now.
    new_root = self.data["root_planar"][clip, frame]
    anchor = se2_compose(target, se2_inverse(new_root))
    old_root = self._clip_planar(old_clip, x)
    anchor_prev = se2_compose(target, se2_inverse(old_root))

    self.clip[env_ids] = clip
    self.frame[env_ids] = frame
    self.twist_command[env_ids] = twist
    self.anchor[env_ids] = anchor
    self.clip_prev[env_ids] = torch.where(same, clip, old_clip)
    self.anchor_prev[env_ids] = torch.where(same[:, None], anchor, anchor_prev)
    self.frame_prev[env_ids] = torch.where(same, frame.float(), x)
    self.rate_prev[env_ids] = torch.where(same, torch.ones_like(rate), rate)
    self.blend_count[env_ids] = torch.where(
      same, self.blend_count[env_ids], torch.zeros_like(frame)
    )
    self.catchup[env_ids] = 0.0
    self._jump_step[env_ids] = self._env.common_step_counter

  def _clip_planar(self, clip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A clip's raw root planar pose at fractional frame ``x`` in [0, T), unrolled."""
    lo = x.floor().long()
    frac = x - lo.float()
    identity = torch.zeros(clip.shape[0], 3, device=self.device)
    a = self._gather(clip, lo, identity)
    b = self._gather(clip, lo + 1, identity)
    pa = planar_pose(a["root_pos"], a["root_quat"])
    pb = planar_pose(b["root_pos"], b["root_quat"])
    turn = torch.remainder(pb[:, 2] - pa[:, 2] + math.pi, 2 * math.pi) - math.pi
    out = torch.lerp(pa, pb, frac[:, None])
    out[:, 2] = pa[:, 2] + frac * turn
    return out

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None and self._twist_requests:
      # A pin change from set_fixed_twist, applied as a timer resample of every env.
      self._fixed_twist = self._twist_requests.popleft()
      self._switch(torch.arange(self.num_envs, device=self.device))
    if env_ids is None:
      step = (~self._just_reset).long()
      advanced = self.frame + step
      n = self.num_frames[self.clip]
      wrapped = advanced >= n
      self.frame = torch.where(wrapped, advanced - n, advanced)
      self.anchor = torch.where(
        wrapped[:, None], se2_compose(self.anchor, self.strides[self.clip]), self.anchor
      )
      # The outgoing clip advances on its own clock and loops at its own length.
      x = self.frame_prev + step.float() * self.rate_prev
      n_prev = self.num_frames[self.clip_prev].float()
      wrapped_prev = x >= n_prev
      self.frame_prev = torch.where(wrapped_prev, x - n_prev, x)
      self.anchor_prev = torch.where(
        wrapped_prev[:, None],
        se2_compose(self.anchor_prev, self.strides[self.clip_prev]),
        self.anchor_prev,
      )
      blending = (self.blend_count < self.blend_steps) & (step > 0)
      self.blend_count = self.blend_count + blending.long()
      gap = self._root_xy(self.clip, self.frame.float(), self.anchor) - self._root_xy(
        self.clip_prev, self.frame_prev, self.anchor_prev
      )
      self.catchup = (
        self.catchup
        + blending.float()[:, None] * (self._step_dt * self._alpha_rate) * gap
      )
      # Blend over: the served root is the new clip's, shifted by the catch-up.
      done = blending & (self.blend_count >= self.blend_steps)
      self.anchor[:, :2] -= done.float()[:, None] * self.catchup
      self.catchup = self.catchup * (~done).float()[:, None]
      # The served velocity loses its alpha_dot term at the next step's reward, when
      # the blend completes; the CLF must not difference across that.
      ending = self.blend_count == self.blend_steps - 1
      self._jump_step = torch.where(
        ending,
        torch.full_like(self._jump_step, self._env.common_step_counter),
        self._jump_step,
      )
      self._just_reset.zero_()
    else:
      self._just_reset[env_ids] = False
    self._cache.clear()

  ##
  # Viewer: the reference ghost and the twist controls.
  ##

  def _build_ghost(self) -> _Ghost:
    """The robot model at this entity's socket, recoloured, showing geom group 0.

    The musculoskeletal tasks get the robot model too: the same skeleton, socket
    and OSL, without the muscles. Joints map by name; joints the robot model lacks
    (kept fingers) are left out.
    """
    joints = list(self.robot.joint_names)
    socket: Socket = "compliant" if set(SOCKET_JOINTS) <= set(joints) else "rigid"
    spec = get_spec("robot", socket)
    model = spec.compile()
    shown = model.geom_group == GHOST_GEOM_GROUP
    model.geom_rgba[shown] = self.cfg.ghost_color
    model.geom_matid[shown] = -1  # Flat colour: no bone textures.
    # The native viewers draw ghosts with mjVIS_TRANSPARENT, which scales each
    # alpha by vis.map.alpha (0.3); 1 keeps ghost_color's alpha as drawn.
    model.vis.map.alpha = 1.0
    model.geom_rgba[~shown, 3] = 0.0
    model.site_rgba[:, 3] = 0.0
    model.tendon_rgba[:, 3] = 0.0
    free = mujoco.mjtJoint.mjJNT_FREE
    root = int(model.jnt_qposadr[np.flatnonzero(model.jnt_type == free)[0]])
    cols, adrs = [], []
    for col, name in enumerate(joints):
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if jid >= 0 and model.jnt_type[jid] != free:
        cols.append(col)
        adrs.append(int(model.jnt_qposadr[jid]))
    return _Ghost(
      model=model,
      root_adr=root,
      qpos_adr=np.asarray(adrs, dtype=np.int64),
      joint_cols=torch.tensor(cols, dtype=torch.long, device=self.device),
    )

  def ghost_qpos(self, env_ids: Sequence[int]) -> np.ndarray:
    """(len(env_ids), nq) the ghost model's qpos at the served reference."""
    if self._ghost is None:
      self._ghost = self._build_ghost()
    ghost = self._ghost
    ref = self._serve(0, torch.tensor(list(env_ids), device=self.device))
    qpos = np.tile(ghost.model.qpos0, (len(env_ids), 1))
    r = ghost.root_adr
    qpos[:, r : r + 3] = ref["root_pos"].cpu().numpy()
    qpos[:, r + 3 : r + 7] = ref["root_quat"].cpu().numpy()
    qpos[:, ghost.qpos_adr] = ref["joint_pos"][:, ghost.joint_cols].cpu().numpy()
    return qpos

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """The served reference (what the reward compares the robot with now) as a
    ghost, at the egocentric target rather than the clip's looping pose."""
    env_ids = list(visualizer.get_env_indices(self.num_envs))
    if not env_ids:
      return
    qpos = self.ghost_qpos(env_ids)
    assert self._ghost is not None
    for row, env in enumerate(env_ids):
      visualizer.add_ghost_mesh(
        qpos[row],
        model=self._ghost.model,
        alpha=self.cfg.ghost_color[3],  # Viser's opacity; native reads geom_rgba.
        label=f"ghost_{env}",
      )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """A "Twist command" folder: a "Pin twist" checkbox, one slider per axis over
    the library's labels and the clip they snap to. While pinned, every env plays
    that clip (``set_fixed_twist``); unpinning returns to the sampler. An axis the
    library never varies shows as a fixed number."""
    del name, get_env_idx, request_action
    lo, hi = self.twist_bounds
    values = list(self._fixed_twist or (0.0, 0.0, 0.0))
    sliders = []
    with server.gui.add_folder("Twist command"):
      pin = server.gui.add_checkbox(
        "Pin twist", initial_value=self._fixed_twist is not None
      )
      for axis, label in enumerate(("vx [m/s]", "vy [m/s]", "wz [rad/s]")):
        low, high = min(lo[axis], 0.0), max(hi[axis], 0.0)
        if high - low < 1e-9:
          values[axis] = low
          server.gui.add_number(label, initial_value=low, disabled=True)
          continue
        values[axis] = min(max(values[axis], low), high)
        slider = server.gui.add_slider(
          label,
          min=low,
          max=high,
          step=GUI_TWIST_STEP,
          initial_value=values[axis],
          disabled=not pin.value,
        )
        sliders.append((axis, slider))
      clip = self.clip_names[self.nearest_clip(values)]
      snapped = server.gui.add_text("Clip", initial_value=clip, disabled=True)

    def push() -> None:
      for axis, slider in sliders:
        values[axis] = float(slider.value)
      twist = (values[0], values[1], values[2])
      snapped.value = self.clip_names[self.nearest_clip(twist)]
      if pin.value:
        self.set_fixed_twist(twist)
      if on_change is not None:
        on_change()

    for _, slider in sliders:
      slider.on_update(lambda _: push())

    @pin.on_update
    def _(_) -> None:
      for _, slider in sliders:
        slider.disabled = not pin.value
      if pin.value:
        push()
      else:
        self.set_fixed_twist(None)
