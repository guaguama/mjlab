"""Torch ports of the loco-mujoco / musclemimic math the imitation terms use.

Every function reproduces its musclemimic source exactly, sign conventions and quirks
included, so that rewards, observations and terminations match number for number.
The source of each function is named in its docstring. Quaternions are (w, x, y, z),
as everywhere in mjlab.
"""

import math

import numpy as np
import torch

from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_from_matrix


def rotvec_from_matrix(mat: torch.Tensor) -> torch.Tensor:
  """scipy's ``Rotation.from_matrix(mat).as_rotvec()``: angle in [0, pi]."""
  return axis_angle_from_quat(quat_from_matrix(mat))


def quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Geodesic angle between two orientations, ``2 acos(|q1 . q2|)``.

  musclemimic's ``quaternion_angular_distance`` (``(r1.inv() * r2).magnitude()``) and
  the root-orientation termination both compute this. The former is fed scalar-first
  quaternions as if they were scalar-last; the angle is unchanged, because the dot
  product is invariant to permuting both quaternions' components the same way.
  """
  q1 = torch.nn.functional.normalize(q1, dim=-1)
  q2 = torch.nn.functional.normalize(q2, dim=-1)
  dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)
  return 2.0 * torch.acos(dot)


def relative_site_quantities(
  pos: torch.Tensor,
  mat: torch.Tensor,
  ang_vel: torch.Tensor,
  lin_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """``calculate_relative_site_quantities`` (``loco_mujoco/core/utils/math.py``).

  Args:
    pos: (..., S, 3) world site positions. Site 0 is the main site (the pelvis).
    mat: (..., S, 3, 3) world site orientations.
    ang_vel: (..., S, 3) world angular velocity of each site.
    lin_vel: (..., S, 3) world linear velocity of each site's origin. Together these
      are ``calc_site_velocities(..., flg_local=False)``.

  Returns:
    rpos: (..., K, 3) ``p_i - p_main``, in the world frame (K = S - 1).
    rangles: (..., K, 3) ``rotvec(R_main^T R_i)``.
    rvel: (..., K, 6) ``[R_rel_i^T w_i - w_main, R_main (v_main - v_i)]``, from
      ``calculate_relative_velocity_in_local_frame``. This is not the time
      derivative of ``rpos`` (it is negated and rotated by ``R_main``, not its
      transpose). MimicReward and the goal observation use it as is; the CLF does
      not (see :func:`relative_site_velocities`).
  """
  rpos = pos[..., 1:, :] - pos[..., :1, :]
  main_mat = mat[..., :1, :, :]
  rel_rot = main_mat.transpose(-1, -2) @ mat[..., 1:, :, :]
  rangles = rotvec_from_matrix(rel_rot)
  ang = (rel_rot.transpose(-1, -2) @ ang_vel[..., 1:, :, None]).squeeze(-1)
  ang = ang - ang_vel[..., :1, :]
  lin = (main_mat @ (lin_vel[..., :1, :] - lin_vel[..., 1:, :])[..., None]).squeeze(-1)
  return rpos, rangles, torch.cat([ang, lin], dim=-1)


def relative_site_velocities(
  mat: torch.Tensor, ang_vel: torch.Tensor, lin_vel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """``CLFReward._relative_site_velocities`` (``musclemimic/core/reward/clf.py``).

  Returns ``v_rel = v_i - v_main`` (world frame, exactly ``d/dt`` of ``rpos``) and
  ``omega_rel = R_main^T (w_i - w_main)`` (main-site frame, the frame of
  ``rangles``), each (..., K, 3).
  """
  v_rel = lin_vel[..., 1:, :] - lin_vel[..., :1, :]
  ang_rel_world = ang_vel[..., 1:, :] - ang_vel[..., :1, :]
  omega_rel = (mat[..., :1, :, :].transpose(-1, -2) @ ang_rel_world[..., None]).squeeze(
    -1
  )
  return v_rel, omega_rel


##
# CLF constants (``musclemimic/core/reward/clf_math.py``).
##


def di_care_block(q_pos: float, q_vel: float, r: float) -> tuple[float, float, float]:
  """Closed-form CARE solution ``P = [[p11, p12], [p12, p22]]`` of a double integrator.

  Solves ``A^T P + P A - P B R^-1 B^T P + Q = 0`` for ``A = [[0, 1], [0, 0]]``,
  ``B = [0, 1]^T``, ``Q = diag(q_pos, q_vel)``, ``R = [r]``.
  """
  if q_pos <= 0.0 or q_vel < 0.0 or r <= 0.0:
    raise ValueError(f"Need q_pos > 0, q_vel >= 0, r > 0; got {q_pos}, {q_vel}, {r}")
  p12 = math.sqrt(r * q_pos)
  p22 = math.sqrt(r * q_vel + 2.0 * (r**1.5) * math.sqrt(q_pos))
  p11 = p22 * math.sqrt(q_pos / r)
  return p11, p12, p22


def exact_decay_rate(q_pos: float, q_vel: float, r: float) -> float:
  """Largest ``alpha`` with ``Vdot + alpha V <= 0`` under the LQR controller.

  The smallest generalized eigenvalue of ``(Q_cl, P)``, where
  ``Q_cl = Q + K^T R K = [[2 q_pos, p11], [p11, 2 q_vel + 2 p12]]``.
  """
  p11, p12, p22 = di_care_block(q_pos, q_vel, r)
  q_cl = np.array([[2.0 * q_pos, p11], [p11, 2.0 * q_vel + 2.0 * p12]])
  p = np.array([[p11, p12], [p12, p22]])
  return float(np.min(np.linalg.eigvals(np.linalg.solve(p, q_cl)).real))


def v_max_unit(p11: float, p12: float, p22: float) -> float:
  """``V`` at unit normalized errors ``e = edot = 1``: ``p11 + 2 p12 + p22``."""
  return p11 + 2.0 * p12 + p22
