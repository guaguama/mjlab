"""Contracts for the musclemimic port (``tasks/osl_mimic``).

The pure-math tests pin the port to musclemimic's own formulas. Their references
were cross-checked against musclemimic and optax 0.2.8 directly: the site
quantities and CLF constants to 1e-7 and 1e-15 on real clip frames, the optimizer
to float precision. The simulation test needs a 100 Hz clipper clip
(``MJLAB_OSL_MIMIC_MOTION_FILE``) and is slow.
"""

import contextlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import mediapy as media
import numpy as np
import pytest
import torch
from conftest import get_test_device
from rsl_rl.utils.log_writer import LogWriter
from scipy.linalg import eigh, solve_continuous_are
from scipy.spatial.transform import Rotation

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonEffortActionCfg
from mjlab.tasks.osl_mimic.config.env_cfgs import (
  CLF_WEIGHTS,
  MIMIC_WEIGHTS,
  osl_mimic_env_cfg,
)
from mjlab.tasks.osl_mimic.config.rl_cfg import (
  HIDDEN_DIMS,
  NUM_UPDATES,
  MuscleMimicPpoCfg,
  OslMimicRunnerCfg,
)
from mjlab.tasks.osl_mimic.mdp import MimicMotionCommand, MimicMotionCommandCfg
from mjlab.tasks.osl_mimic.mdp import kinematics as K
from mjlab.tasks.osl_mimic.rl.models import RunningMeanStd
from mjlab.tasks.osl_mimic.rl.muon import MuonAdam
from mjlab.tasks.osl_mimic.rl.ppo import MuscleMimicPPO
from mjlab.tasks.osl_mimic.rl.runner import OslMimicOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASKS = {
  "Mjlab-Mimic-OSL-Fullbody-ROBOT": ("robot", "rigid", "mimic"),
  "Mjlab-Mimic-OSL-Fullbody-ROBOT-Compliant": ("robot", "compliant", "mimic"),
  "Mjlab-Mimic-OSL-Fullbody-ROBOT-CLF": ("robot", "rigid", "clf"),
  "Mjlab-Mimic-OSL-Fullbody-ROBOT-CLF-Compliant": ("robot", "compliant", "clf"),
  "Mjlab-Mimic-OSL-Fullbody-MSK": ("msk", "rigid", "mimic"),
  "Mjlab-Mimic-OSL-Fullbody-MSK-Compliant": ("msk", "compliant", "mimic"),
  "Mjlab-Mimic-OSL-Fullbody-MSK-CLF": ("msk", "rigid", "clf"),
  "Mjlab-Mimic-OSL-Fullbody-MSK-CLF-Compliant": ("msk", "compliant", "clf"),
}
MOTION = Path(os.environ.get("MJLAB_OSL_MIMIC_MOTION_FILE", "/nonexistent.npz"))


##
# Configuration.
##


@pytest.mark.parametrize("task", list(TASKS))
def test_task_contract(task: str) -> None:
  model, socket, reward = TASKS[task]
  cfg = load_env_cfg(task)
  play = load_env_cfg(task, play=True)
  assert load_runner_cls(task) is OslMimicOnPolicyRunner

  # 2 ms Euler, 5 substeps (100 Hz), 4/8/35 iterations, Euler damping off iff rigid.
  assert cfg.sim.mujoco.timestep == 0.002 and cfg.decimation == 5
  assert cfg.sim.mujoco.integrator == "euler"
  mj = cfg.sim.mujoco
  assert (mj.iterations, mj.ls_iterations, mj.ccd_iterations) == (4, 8, 35)
  assert ("eulerdamp" in cfg.sim.mujoco.disableflags) == (socket == "rigid")
  assert "multiccd" in cfg.sim.mujoco.disableflags
  assert not cfg.scale_rewards_by_dt
  assert cfg.scene.entities["robot"].collisions == ()  # The model's own contacts.

  # No randomization or noise anywhere.
  assert set(cfg.events) <= {"reset_scene_to_default"}
  assert not cfg.observations["actor"].enable_corruption
  assert list(cfg.observations["actor"].terms) == list(cfg.observations["critic"].terms)
  muscles = ["muscles"] if model == "msk" else []
  assert list(cfg.observations["actor"].terms) == [
    "root_pos",
    "joint_pos",
    "root_vel",
    "joint_vel",
    *muscles,
    "touch",
    "goal",
  ]

  if model == "robot":
    assert list(cfg.actions) == ["servos", "osl_motors"]
    servos = cfg.actions["servos"]
    assert isinstance(servos, JointPositionActionCfg)
    assert not servos.use_default_offset and servos.clip is not None
  else:
    assert list(cfg.actions) == ["osl_motors", "muscles"]
    assert isinstance(cfg.actions["muscles"], TendonEffortActionCfg)

  params = cfg.rewards["mimic"].params
  assert cfg.rewards["mimic"].weight == 1.0
  assert params["weights"] == (CLF_WEIGHTS if reward == "clf" else MIMIC_WEIGHTS)
  assert (params["clf"] is not None) == (reward == "clf")
  assert params["action_out_of_bounds_coeff"] == 0.01

  assert set(cfg.terminations) == {
    "time_out",
    "trajectory_end",
    "site_deviation",
    "root_deviation",
    "root_orientation",
  }
  assert set(play.terminations) == {"time_out", "trajectory_end", "site_deviation"}
  assert play.terminations["site_deviation"].params["threshold"] == 0.5
  motion, play_motion = cfg.commands["motion"], play.commands["motion"]
  assert isinstance(motion, MimicMotionCommandCfg)
  assert isinstance(play_motion, MimicMotionCommandCfg)
  assert motion.random_start and not play_motion.random_start
  assert (motion.lookahead_steps, motion.lookahead_stride) == (5, 20)

  rl = load_rl_cfg(task)
  assert isinstance(rl, OslMimicRunnerCfg)
  assert rl.num_steps_per_env == 80 and rl.max_iterations == NUM_UPDATES == 6103
  alg = rl.algorithm
  assert isinstance(alg, MuscleMimicPpoCfg)
  assert (alg.num_learning_epochs, alg.num_mini_batches) == (1, 128)
  assert alg.learning_rate == 4e-4 and alg.entropy_coef == 0.0
  assert alg.value_loss_coef == 0.25
  assert alg.actor_bias_init == ("qpos0" if model == "robot" else "none")
  assert rl.actor.hidden_dims == HIDDEN_DIMS
  assert rl.actor.distribution_cfg is not None
  assert rl.actor.distribution_cfg["init_std"] == (0.2 if model == "robot" else 3.0)
  assert rl.validation.interval == 61


##
# Math, against musclemimic's formulas.
##


def _reference_site_quantities(pos, mat, ang, lin):
  """``calculate_relative_site_quantities`` transcribed with numpy and scipy."""
  main_mat = mat[0]
  rel_rot = np.einsum("ik,nkj->nij", main_mat.T, mat[1:])
  rpos = pos[1:] - pos[0]
  rangles = Rotation.from_matrix(rel_rot).as_rotvec()
  lin_rel = np.einsum("jk,ik->ij", main_mat, lin[0] - lin[1:])
  ang_rel = np.einsum("ikj,ik->ij", rel_rot, ang[1:]) - ang[0]
  return rpos, rangles, np.hstack([ang_rel, lin_rel])


def test_relative_site_quantities() -> None:
  rng = np.random.default_rng(0)
  sites = 17
  pos = rng.normal(size=(sites, 3))
  mat = Rotation.random(sites, random_state=1).as_matrix()
  ang, lin = rng.normal(size=(sites, 3)), rng.normal(size=(sites, 3))
  expected = _reference_site_quantities(pos, mat, ang, lin)
  got = K.relative_site_quantities(*(torch.tensor(x) for x in (pos, mat, ang, lin)))
  for g, e in zip(got, expected, strict=True):
    np.testing.assert_allclose(g.numpy(), e, atol=1e-9)
  v_rel, omega_rel = K.relative_site_velocities(
    *(torch.tensor(x) for x in (mat, ang, lin))
  )
  np.testing.assert_allclose(v_rel.numpy(), lin[1:] - lin[0], atol=1e-12)
  np.testing.assert_allclose(omega_rel.numpy(), (ang[1:] - ang[0]) @ mat[0], atol=1e-12)


def test_quat_angle_ignores_component_order() -> None:
  q1 = torch.tensor(Rotation.random(50, random_state=2).as_quat(scalar_first=True))
  q2 = torch.tensor(Rotation.random(50, random_state=3).as_quat(scalar_first=True))
  expected = Rotation.from_quat(q1.numpy()).inv() * Rotation.from_quat(q2.numpy())
  np.testing.assert_allclose(
    K.quat_angle(q1, q2).numpy(), expected.magnitude(), atol=1e-6
  )


def test_clf_constants() -> None:
  p11, p12, p22 = K.di_care_block(1.0, 1.0, 0.1)
  a = np.array([[0.0, 1.0], [0.0, 0.0]])
  b = np.array([[0.0], [1.0]])
  care = solve_continuous_are(a, b, np.eye(2), np.array([[0.1]]))
  np.testing.assert_allclose([[p11, p12], [p12, p22]], care, atol=1e-12)
  q_cl = np.array([[2.0, p11], [p11, 2.0 + 2.0 * p12]])
  rate = eigh(q_cl, np.array([[p11, p12], [p12, p22]]), eigvals_only=True)[0]
  assert K.exact_decay_rate(1.0, 1.0, 0.1) == pytest.approx(rate, abs=1e-12)
  # musclemimic's values for its OSL config.
  assert (p11, p12, p22) == pytest.approx((1.27767583, 0.31622777, 0.40403657))
  assert K.v_max_unit(p11, p12, p22) == pytest.approx(2.31416794)
  assert rate == pytest.approx(1.28401503)


##
# Learning.
##


def test_muon_matches_optax() -> None:
  """Three steps of optax 0.2.8's clip_by_global_norm(1) + contrib.muon(lr=0.05).

  The second gradient exceeds the clip norm. The flax kernel is (in, out); the
  torch weight is its transpose.
  """
  kernel = (np.arange(6, dtype=np.float32).reshape(3, 2) - 2.5) / 4.0
  w = torch.nn.Parameter(torch.tensor(kernel.T.copy()))
  b = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
  opt = MuonAdam([w, b], lr=0.05, max_grad_norm=1.0)
  for t in range(3):
    gk = np.sin(np.arange(6, dtype=np.float32) * (t + 1)).reshape(3, 2)
    w.grad = torch.tensor((gk * (3.0 if t == 1 else 0.5)).T.copy())
    b.grad = torch.tensor(np.cos(np.arange(2, dtype=np.float32) + t) * 0.7)
    opt.step()
  np.testing.assert_allclose(
    w.detach().numpy(),
    [
      [-0.632701575756073, -0.10026022791862488, 0.3971562087535858],
      [-0.47923633456230164, 0.14172862470149994, 0.670891284942627],
    ],
    atol=2e-6,
  )
  np.testing.assert_allclose(
    b.detach().numpy(), [0.002908229362219572, -0.24031654000282288], atol=2e-6
  )


def test_muon_skips_non_finite_steps() -> None:
  w = torch.nn.Parameter(torch.ones(2, 2))
  opt = MuonAdam([w], lr=0.1)
  w.grad = torch.full((2, 2), float("nan"))
  opt.step()
  assert torch.equal(w.detach(), torch.ones(2, 2)) and opt.skipped_steps == 1


def test_running_mean_std_modes() -> None:
  rms = RunningMeanStd(3)
  x = torch.randn(64, 3) * 2.0 + 1.0
  # musclemimic's Welford update from count 1e-6, var 1.
  n, count = 64, 1e-6
  mean = (x.mean(0) - 0.0) * n / (count + n)
  m2 = 1.0 * count + (x.var(0, unbiased=False) + 1e-6) * n
  m2 = m2 + x.mean(0).square() * count * n / (count + n)
  var = m2 / (count + n)
  expected = (x - mean) / torch.sqrt(var + 1e-8)

  rms.mode = "transient"
  torch.testing.assert_close(rms(x), expected)
  assert torch.count_nonzero(rms.mean) == 0  # Not persisted.
  rms.mode = "update"
  torch.testing.assert_close(rms(x), expected)
  torch.testing.assert_close(rms.mean, mean)
  rms.mode = "frozen"
  torch.testing.assert_close(rms(x), (x - mean) / torch.sqrt(var + 1e-8))


def test_reward_normalization_matches_normalize_vec_reward() -> None:
  alg = SimpleNamespace(
    gamma=0.99,
    _return=torch.zeros(4),
    _return_mean=torch.zeros(()),
    _return_var=torch.ones(()),
    _return_count=torch.tensor(1e-4),
  )
  rng = np.random.default_rng(0)
  state = {"mean": 0.0, "var": 1.0, "count": 1e-4, "ret": np.zeros(4)}
  for _ in range(5):
    r = rng.random(4)
    d = (rng.random(4) < 0.3).astype(float)
    got = MuscleMimicPPO._normalize_rewards(
      cast(MuscleMimicPPO, alg), torch.tensor(r), torch.tensor(d)
    )
    ret = state["ret"] * 0.99 * (1 - d) + r
    delta = ret.mean() - state["mean"]
    total = state["count"] + 4
    state["mean"] += delta * 4 / total
    m2 = (
      state["var"] * state["count"]
      + ret.var() * 4
      + delta**2 * state["count"] * 4 / total
    )
    state.update(var=m2 / total, count=total, ret=ret)
    np.testing.assert_allclose(
      got.numpy(), r / math.sqrt(state["var"] + 1e-8), rtol=1e-6
    )


def test_linear_learning_rate() -> None:
  alg = SimpleNamespace(
    lr_schedule="linear",
    lr_schedule_updates=100,
    base_learning_rate=4e-4,
    num_updates_done=0,
  )
  schedule = MuscleMimicPPO.scheduled_learning_rate
  assert schedule(cast(MuscleMimicPPO, alg)) == pytest.approx(4e-4)
  alg.num_updates_done = 50
  assert schedule(cast(MuscleMimicPPO, alg)) == pytest.approx(2e-4)
  alg.num_updates_done = 100
  assert schedule(cast(MuscleMimicPPO, alg)) == 0.0


##
# Simulation (slow; needs the 100 Hz clip).
##


@pytest.mark.slow
@pytest.mark.parametrize("model", ["robot", "msk"])
def test_reference_start_and_first_step(model: str) -> None:
  """Episodes start on the exact reference frame and advance one frame per step."""
  if not MOTION.exists():
    pytest.skip("Set MJLAB_OSL_MIMIC_MOTION_FILE to a 100 Hz clipper mjlab clip")
  cfg = osl_mimic_env_cfg(model, "rigid", "mimic", mode="validation")  # type: ignore[arg-type]
  cfg.scene.num_envs = 2
  motion = cfg.commands["motion"]
  assert isinstance(motion, MimicMotionCommandCfg)
  motion.motion_file = str(MOTION)
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    cmd = cast(MimicMotionCommand, env.command_manager.get_term("motion"))
    assert cmd.frame.tolist() == [0, 0]
    torch.testing.assert_close(cmd.robot.data.joint_pos, cmd.ref_joint_pos[[0, 0]])
    goal = cast(torch.Tensor, obs["actor"])[:, -469:]
    torch.testing.assert_close(goal[:, -1], torch.zeros(2, device=env.device))
    zero = torch.zeros(2, env.action_manager.total_action_dim, device=env.device)
    _, reward, terminated, _, _ = env.step(zero)
    assert torch.isfinite(reward).all() and not terminated.any()
    assert cmd.frame.tolist() == [1, 1]
  finally:
    env.close()


##
# Checkpoint videos.
##


def _video_runner(tmp_path: Path, fail: bool = False) -> tuple[Any, list[float]]:
  """A runner whose video env renders numbered frames and records its reset draw."""
  runner = OslMimicOnPolicyRunner.__new__(OslMimicOnPolicyRunner)
  draws: list[float] = []
  count = iter(range(10_000))

  def render() -> np.ndarray:
    return np.full((48, 64, 3), next(count) % 256, dtype=np.uint8)

  def reset() -> tuple[torch.Tensor, dict]:
    draws.append(float(torch.rand(1)))
    return torch.zeros(1, 3), {}

  def step(
    actions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if fail:
      raise RuntimeError("renderer lost")
    return torch.zeros(1, 3), torch.zeros(1), torch.zeros(1), {}

  env = SimpleNamespace(device="cpu", step_dt=0.01, render=render)
  vec = SimpleNamespace(unwrapped=env, device="cpu", reset=reset, step=step)
  runner._video_env = cast(Any, vec)
  runner.video_cfg = {"enabled": True, "num_steps": 5, "seed": 3}
  runner.device = "cpu"
  runner.logger = cast(
    Any, SimpleNamespace(log_dir=str(tmp_path), writer=mock.Mock(spec=LogWriter))
  )
  setattr(  # noqa: B010  (an instance override of the method)
    runner,
    "_video_policy",
    lambda env: contextlib.nullcontext(lambda obs: torch.zeros(1, 2)),
  )
  return runner, draws


def test_record_video_writes_uploads_and_leaves_training_rng(tmp_path: Path) -> None:
  runner, draws = _video_runner(tmp_path)
  torch.manual_seed(123)
  state = torch.get_rng_state()
  path = runner.record_video(7)
  assert path == tmp_path / "videos" / "model_7.mp4"
  video = media.read_video(path)
  assert video.metadata is not None and video.metadata.fps == 100
  assert video.shape[:3] == (5, 48, 64)
  runner.logger.writer.save_video.assert_called_once_with(path, 7)
  # Training's random state is untouched, and the video's draws are seeded alike
  # at every checkpoint.
  assert torch.equal(torch.get_rng_state(), state)
  runner.record_video(8)
  torch.manual_seed(3)
  assert draws == [float(torch.rand(1))] * 2
  # A checkpoint saved twice (the final re-save) is recorded once.
  assert runner.record_video(7) == path and len(draws) == 2


def test_record_video_failure_never_raises(tmp_path: Path) -> None:
  runner, _ = _video_runner(tmp_path, fail=True)
  assert runner.record_video(3) is None
  assert not (tmp_path / "videos" / "model_3.mp4").exists()
  runner.logger.writer.save_video.assert_not_called()
  runner.video_cfg["enabled"] = False
  assert runner.record_video(4) is None
