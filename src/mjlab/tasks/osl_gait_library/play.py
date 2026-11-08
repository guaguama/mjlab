"""Keyboard twist control for the gait-library tasks in the native viewer.

sesteban's G1 unicycle ``play.py`` teleop, on :class:`GaitLibraryCommand`. The rest of
his play additions live elsewhere: ``--twist`` sets the command's ``fixed_twist``, the
viser sliders are the command's ``create_gui``, ``--no-disturbances`` is
:func:`~mjlab.tasks.osl_gait_library.config.dr.nominal_plant` and the reference ghost
is the command's debug visualization.
"""

from __future__ import annotations

from collections.abc import Callable

from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommand
from mjlab.viewer.native.keys import (
  KEY_DELETE,
  KEY_DOWN,
  KEY_HOME,
  KEY_PAGE_DOWN,
  KEY_PAGE_UP,
  KEY_UP,
)

VX_STEP = 0.05
"""Forward speed per key press (m/s)."""
WZ_STEP = 0.1
"""Yaw rate per key press (rad/s)."""

TELEOP_HELP = f"""[teleop] gait-library twist keys (native viewer; applied to ALL envs):
  Up / Down          forward speed +/- {VX_STEP} m/s (zeroes the yaw rate)
  PageUp / PageDown  yaw rate +/- {WZ_STEP} rad/s, left / right (zeroes the speed)
  Delete             stop: zero twist, the idle clip
  Home               release the pin: back to the task's own sampler
  Values clamp to the library's labels; the clip played is printed after each change.
  (Letters and [ ] are avoided: MuJoCo's viewer binds them to render toggles.)"""


def make_twist_teleop(cmd: GaitLibraryCommand) -> Callable[[int], None]:
  """A native-viewer key callback that pins the twist of every env (TELEOP_HELP).

  It runs on the viewer thread, so it only posts pins (``set_fixed_twist``) and
  snaps on the CPU (``nearest_clip``). Speed and yaw keys exclude each other, as in
  sesteban's teleop: arcs are reachable from ``--twist`` and the viser sliders.
  """
  lo, hi = cmd.twist_bounds
  start = cmd.fixed_twist or tuple(float(v) for v in cmd.twist_command[0].tolist())
  twist = [float(v) for v in start]

  def clamp(axis: int, value: float) -> float:
    return round(min(max(value, lo[axis]), hi[axis]), 3)

  def callback(key: int) -> None:
    if key == KEY_UP:
      twist[0], twist[2] = clamp(0, twist[0] + VX_STEP), 0.0
    elif key == KEY_DOWN:
      twist[0], twist[2] = clamp(0, twist[0] - VX_STEP), 0.0
    elif key == KEY_PAGE_UP:
      twist[0], twist[2] = 0.0, clamp(2, twist[2] + WZ_STEP)
    elif key == KEY_PAGE_DOWN:
      twist[0], twist[2] = 0.0, clamp(2, twist[2] - WZ_STEP)
    elif key == KEY_DELETE:
      twist[:] = [0.0, 0.0, 0.0]
    elif key == KEY_HOME:
      cmd.set_fixed_twist(None)
      print("[teleop] pin released: back to the task's twist sampler")
      return
    else:
      return
    pinned = (twist[0], twist[1], twist[2])
    cmd.set_fixed_twist(pinned)
    clip = cmd.nearest_clip(pinned)
    label = cmd.twist_labels[clip].tolist()
    print(
      f"[teleop] twist vx {pinned[0]:+.2f} vy {pinned[1]:+.2f} wz {pinned[2]:+.2f}"
      f"  -> {cmd.clip_names[clip]} (vx {label[0]:+.2f} vy {label[1]:+.2f}"
      f" wz {label[2]:+.2f})"
    )

  return callback
