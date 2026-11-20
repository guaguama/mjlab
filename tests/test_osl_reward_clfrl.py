"""Tests for the CLF-RL reward changes of 2026-11-18 (``osl_mimic.mdp.rewards``,
``osl_split.mdp.rewards``): the paper-form decrease condition, the regularizers,
their env stages, and the old reward reachable by flags.

CPU only. The terms run on fake env and command objects, except where MuJoCo's own
actuation is the reference (the servo terms) or where a real model's muscles are
read (the activation metric):

* the old flags (``decay_form="musclemimic"``, the old ``pen_w``, every regularizer
  at 0) reproduce the reward of commit 2d1e12a89 bit for bit, stage 1 and stage 2,
  with every task's own params (``osl_reward_head.py`` holds the frozen copies).
  That holds for the reward term given the same reference: the gait library's
  served ``rvel`` itself changed (it is now turned with the anchor);
* a golden fixture recorded from musclemimic's own environment
  (``fixtures/musclemimic_reward_golden.npz``) pins those flags to musclemimic;
* the paper form against a numpy transcription, and stage 2's per-group finite
  difference under a weight change;
* each regularizer: its rows or joints, its masks, its owner in stage 2.
"""

from __future__ import annotations

import copy
import math
import re
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
import tyro
from osl_reward_head import HeadMimicReward, HeadSplitMimicReward

import mjlab
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  OSL_TORQUE_JOINTS,
  SERVO_JOINTS,
  Model,
  get_spec,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_gait_library.evaluate import apply_run_reward
from mjlab.tasks.osl_mimic.config.env_cfgs import (
  CLF_WEIGHTS,
  MIMIC_WEIGHTS,
  OSL_TORQUE_WEIGHTS,
)
from mjlab.tasks.osl_mimic.config.rl_cfg import CLF_ENV_STAGES
from mjlab.tasks.osl_mimic.mdp.commands import MIMIC_BODIES
from mjlab.tasks.osl_mimic.mdp.kinematics import (
  di_care_block,
  relative_site_quantities,
  relative_site_rvel,
  relative_site_velocities,
  v_max_unit,
)
from mjlab.tasks.osl_mimic.mdp.reference import MimicReference
from mjlab.tasks.osl_mimic.mdp.rewards import (
  LIMIT_JOINTS,
  REGULARIZER_DIAGNOSTICS,
  REGULARIZERS,
  SERVO_LOG_DIAGNOSTICS,
  ClfCfg,
  RegularizerCfg,
  clf_cfg,
  clf_regularizers,
  mimic_reward,
)
from mjlab.tasks.osl_mimic.reward_settings import CHECKPOINT_KEY as REWARD_KEY
from mjlab.tasks.osl_mimic.reward_settings import (
  apply_reward_settings,
  check_resume_reward,
  env_reward_settings,
  resume_reward_mismatches,
  reward_flags,
  reward_mismatches,
  reward_record,
  reward_settings,
)
from mjlab.tasks.osl_mimic.rl.runner import OslMimicOnPolicyRunner, stage_accessors
from mjlab.tasks.osl_split.mdp.rewards import (
  CHANNELS,
  GROUPS,
  SPLIT_DIAGNOSTICS,
  SplitRewardCfg,
  split_mimic_reward,
  stage1_channel_weights,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.os import dump_yaml

NUM_ENVS = 6
DT = 0.01
JOINT_NAMES = (
  "hip_flexion_r",
  "hip_adduction_r",
  "OSL_KA_L1_osl_ankle_angle_r",
  "hip_rotation_r",
  "socket_tx",
  "OSL_KA_L1_osl_knee_angle_r",
  "hip_flexion_l",
  "hip_adduction_l",
  "hip_rotation_l",
  "knee_angle_l",
  "ankle_angle_l",
  "subtalar_angle_l",
  "lumbar_extension",
)
"""Every watched joint, the OSL's ankle before its knee and apart from it, and three
joints the limit penalty must ignore."""
LAYOUTS: dict[str, tuple[tuple[str, int], ...]] = {
  "msk": (("osl_motors", 2), ("muscles", 9)),
  "robot": (("servos", 7), ("osl_motors", 2)),
}
"""The stage-1 action orders (OSL rows 0:2 and 7:9), with few human rows."""
ERRORS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.2)
"""Per-env distance of the robot from the reference: rewards from ~1 to clipped."""
FIXTURE = Path(__file__).parent / "fixtures" / "musclemimic_reward_golden.npz"


##
# Fakes.
##


def _unit(x: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.normalize(x, dim=-1)


class _Command:
  """A :class:`MimicReferenceSource` over a random reference and a robot ``error``
  (per env) away from it along fixed random directions, so that a smoothly changing
  error moves the robot smoothly."""

  def __init__(self, seed: int = 0) -> None:
    self.gen = torch.Generator().manual_seed(seed)
    self.cfg = SimpleNamespace(site_bodies=MIMIC_BODIES)
    self.site_body_ids = list(range(len(MIMIC_BODIES)))
    v_adr = torch.arange(6, 6 + len(JOINT_NAMES), dtype=torch.int)
    self.robot: Any = SimpleNamespace(
      joint_names=JOINT_NAMES,
      indexing=SimpleNamespace(joint_v_adr=v_adr),
      data=SimpleNamespace(),
      actuators=[],
    )
    self.jumped = torch.zeros(NUM_ENVS, dtype=torch.bool)
    self.new_reference()
    self.set_error(ERRORS)

  def randn(self, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=self.gen)

  def new_reference(self) -> None:
    b, s, j = NUM_ENVS, len(MIMIC_BODIES), len(JOINT_NAMES)
    site_pos, site_ang, site_lin = (self.randn(b, s, 3) for _ in range(3))
    self._site_quat = _unit(self.randn(b, s, 4))
    site_mat = matrix_from_quat(self._site_quat)
    rpos, rangles, rvel = relative_site_quantities(
      site_pos, site_mat, site_ang, site_lin
    )
    v_rel, omega_rel = relative_site_velocities(site_mat, site_ang, site_lin)
    self.ref = MimicReference(
      root_pos_w=self.randn(b, 3),
      root_quat=_unit(self.randn(b, 4)),
      root_lin_vel_w=self.randn(b, 3),
      root_ang_vel_b=self.randn(b, 3),
      joint_pos=self.randn(b, j),
      joint_vel=self.randn(b, j),
      site_pos_w=site_pos,
      rpos=rpos,
      rangles=rangles,
      rvel=rvel,
      v_rel=v_rel,
      omega_rel=omega_rel,
    )
    self._ref_sites = (site_pos, site_ang, site_lin)
    shapes = {
      "root_pos": (b, 3),
      "root_quat": (b, 4),
      "root_qvel": (b, 6),
      "joint_pos": (b, j),
      "joint_vel": (b, j),
      "site_pos": (b, s, 3),
      "site_quat": (b, s, 4),
      "site_ang": (b, s, 3),
      "site_lin": (b, s, 3),
    }
    self._noise = {key: self.randn(*shape) for key, shape in shapes.items()}

  def set_error(self, error: float | tuple[float, ...] | torch.Tensor) -> None:
    """The robot ``error`` (per env) from the reference (0: exactly on it)."""
    scale = torch.as_tensor(error, dtype=torch.float32).expand(NUM_ENVS)
    noise = self._noise

    def near(x: torch.Tensor, key: str) -> torch.Tensor:
      shape = (NUM_ENVS,) + (1,) * (x.dim() - 1)
      return x + scale.reshape(shape) * noise[key]

    def near_quat(q: torch.Tensor, key: str) -> torch.Tensor:
      shape = (NUM_ENVS,) + (1,) * (q.dim() - 1)
      moved = _unit(q + scale.reshape(shape) * noise[key])
      return torch.where(scale.reshape(shape) > 0.0, moved, q)

    ref = self.ref
    self._root_q = torch.cat(
      [near(ref.root_pos_w, "root_pos"), near_quat(ref.root_quat, "root_quat")], -1
    )
    self._root_v = near(ref.root_qvel, "root_qvel")
    self.robot.data.joint_pos = near(ref.joint_pos, "joint_pos")
    self.robot.data.joint_vel = near(ref.joint_vel, "joint_vel")
    site_pos, site_ang, site_lin = self._ref_sites
    self._sites = (
      near(site_pos, "site_pos"),
      matrix_from_quat(near_quat(self._site_quat, "site_quat")),
      near(site_ang, "site_ang"),
      near(site_lin, "site_lin"),
    )

  @property
  def root_qpos(self) -> torch.Tensor:
    return self._root_q

  @property
  def root_qvel(self) -> torch.Tensor:
    return self._root_v

  @property
  def reference_jumped(self) -> torch.Tensor:
    return self.jumped

  def site_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return self._sites

  def reference(self, offset: int) -> MimicReference:
    assert offset == 1  # Rewards compare the post-step state with the next frame.
    return self.ref


def _env(cmd: _Command, model: str = "msk") -> Any:
  """A fake env over ``cmd`` with the model's action layout; the joints' hard
  limits are +-1 rad (wider on the last envs, as a randomized model's would be)."""
  dims = dict(LAYOUTS[model])
  terms = {name: SimpleNamespace(action_dim=dim) for name, dim in dims.items()}
  limits = torch.tensor([-1.0, 1.0]).expand(NUM_ENVS, len(JOINT_NAMES), 2).clone()
  limits[-2:] *= 1.5
  cmd.robot.data.joint_pos_limits = limits
  num_actions = sum(dims.values())
  return SimpleNamespace(
    num_envs=NUM_ENVS,
    device="cpu",
    step_dt=DT,
    episode_length_buf=torch.zeros(NUM_ENVS, dtype=torch.long),
    command_manager=SimpleNamespace(get_term=lambda name: cmd),
    action_manager=SimpleNamespace(
      active_terms=list(dims),
      get_term=lambda name: terms[name],
      action=torch.zeros(NUM_ENVS, num_actions),
      prev_action=torch.zeros(NUM_ENVS, num_actions),
    ),
    sim=SimpleNamespace(
      data=SimpleNamespace(qfrc_actuator=torch.zeros(NUM_ENVS, 6 + len(JOINT_NAMES)))
    ),
  )


def _advance(
  env: Any,
  cmd: _Command,
  *,
  error: float | tuple[float, ...] | torch.Tensor = ERRORS,
  new_reference: bool = True,
  reset: tuple[int, ...] = (),
  jumped: tuple[int, ...] = (),
) -> None:
  """One env step: a new state (a new reference unless ``new_reference`` is
  False), action and actuator force. ``reset`` envs are on their first episode
  step and ``jumped`` envs see a reference jump."""
  if new_reference:
    cmd.new_reference()
  cmd.set_error(error)
  env.episode_length_buf += 1
  env.episode_length_buf[list(reset)] = 1
  cmd.jumped = torch.zeros(NUM_ENVS, dtype=torch.bool)
  cmd.jumped[list(jumped)] = True
  manager = env.action_manager
  manager.prev_action = manager.action
  manager.action = 1.2 * cmd.randn(NUM_ENVS, manager.action.shape[-1])
  manager.prev_action[list(reset)] = 0.0  # mjlab zeroes it at reset.
  env.sim.data.qfrc_actuator = 40.0 * cmd.randn(NUM_ENVS, 6 + len(JOINT_NAMES))


def _params(
  reward: str,
  clf: ClfCfg | None = None,
  regularizers: RegularizerCfg | None = None,
) -> dict[str, Any]:
  """The stage-1 term's params (``osl_mimic_env_cfg``'s keys)."""
  is_clf = reward == "clf"
  return {
    "command_name": "motion",
    "weights": dict(CLF_WEIGHTS if is_clf else MIMIC_WEIGHTS),
    "action_out_of_bounds_coeff": 0.01,
    "joint_torque_weights": dict(OSL_TORQUE_WEIGHTS),
    "clf": (clf or ClfCfg()) if is_clf else None,
    "regularizers": regularizers,
  }


def _term(env: Any, params: dict[str, Any], split: bool = False) -> Any:
  func = split_mimic_reward if split else mimic_reward
  if split:
    params.setdefault("channels", SplitRewardCfg())
  return func(RewardTermCfg(func=func, weight=1.0, params=params), env)


def _no_contact(regularizers: RegularizerCfg) -> RegularizerCfg:
  """``regularizers`` without the leg-contact penalty, which needs the scene's contact
  sensor these fake envs lack (``test_osl_leg_contact`` covers it)."""
  return replace(regularizers, leg_contact=0.0)


def _v_max(clf: ClfCfg) -> float:
  return v_max_unit(*di_care_block(clf.q_pos, clf.q_vel, clf.r)) + 1e-8


def _assert_identical(got: torch.Tensor, want: torch.Tensor) -> None:
  assert got.dtype == want.dtype and got.shape == want.shape
  torch.testing.assert_close(got, want, rtol=0.0, atol=0.0, equal_nan=True)


def _legacy(params: dict[str, Any], pen_w: float | None) -> dict[str, Any]:
  """``params`` with the old flags: musclemimic's decrease condition at the old
  ``pen_w`` and every regularizer at 0."""
  params = copy.deepcopy(params)
  if params.get("clf") is not None:
    assert pen_w is not None
    params["clf"].decay_form = "musclemimic"
    params["clf"].pen_w = pen_w
  regularizers = params["regularizers"]
  for name in ("osl_action_rate", "servo_action_rate", "joint_limits", "leg_contact"):
    setattr(regularizers, name, 0.0)
  regularizers.servo_torque = regularizers.servo_saturation = 0.0
  return params


##
# The old flags: bit for bit the reward of commit 2d1e12a89.
##

# task -> (model, the old pen_w; None under MimicReward)
OLD_FLAG_TASKS: dict[str, tuple[str, float | None]] = {
  "Mjlab-Mimic-OSL-Fullbody-MSK-CLF": ("msk", 0.25),
  "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-MSK-CLF-Compliant": ("msk", 0.0),
  "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT-CLF": ("robot", 0.25),
  "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT": ("robot", None),
  "Mjlab-Stairs-OSL-Fullbody-MSK-CLF-Compliant": ("msk", 0.25),
  "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant": ("msk", 0.0),
  "Mjlab-Split-Unicycle-OSL-Fullbody-ROBOT-CLF": ("robot", 0.25),
  "Mjlab-Split-Unicycle-OSL-Fullbody-MSK": ("msk", None),
  "Mjlab-Split-Stairs-OSL-Fullbody-MSK-CLF-Compliant": ("msk", 0.0),
  "Mjlab-FSM-Unicycle-OSL-Fullbody-MSK-CLF-Compliant": ("msk", 0.0),
}
"""Unicycle, stairs and FSM, stage 1 and stage 2, both models, both rewards. The
old pen_w is each task's own at 2d1e12a89: 0 on the musculoskeletal unicycle tasks
and everything built on them, 0.25 elsewhere."""


@pytest.mark.parametrize("task", list(OLD_FLAG_TASKS))
def test_old_flags_reproduce_head_bit_for_bit(task: str) -> None:
  model, pen_w = OLD_FLAG_TASKS[task]
  term_cfg = load_env_cfg(task).rewards["mimic"]
  split = term_cfg.func is split_mimic_reward
  params = _legacy(dict(term_cfg.params), pen_w)
  head_params = {k: v for k, v in params.items() if k != "regularizers"}
  cmd = _Command(seed=1)
  env = _env(cmd, model)
  new = _term(env, params, split)
  head_cls = HeadSplitMimicReward if split else HeadMimicReward
  head_cfg = RewardTermCfg(func=head_cls, weight=1.0, params=head_params)
  head = cast(Any, head_cls(head_cfg, env))
  # Split terms: the stage-1 weights, then the default anneal's end on the OSL
  # channel.
  weights = stage1_channel_weights(params["clf"] is not None)
  end = {"human": 0.5, "osl_sites": 1.0, "osl_joints": 10.0}
  steps: list[dict[str, Any]] = [
    {},  # Every env on its first episode step, and the terms' first call.
    {},
    {"jumped": (1, 4)},
    {"reset": (2,), "jumped": (5,)},
    {"error": 0.0},  # Exactly on the reference.
    {"new_reference": False, "error": tuple(e * 1.1 for e in ERRORS)},
    {},  # Env 3 not finite.
    {},
    {"new_reference": False},  # A channel's floor binds on envs 4 and 5.
  ]
  osl = [7, 8] if model == "robot" else [0, 1]
  human = [r for r in range(env.action_manager.action.shape[-1]) if r not in osl]
  rewards = []
  for call, step in enumerate(steps):
    _advance(env, cmd, **step)
    if call == 6:
      cmd.robot.data.joint_pos[3, 0] = float("nan")
    if call == 8:  # Far from the reference: musclemimic's violation is positive.
      env.action_manager.action[4, human] = 40.0
      env.action_manager.action[5, osl] = -40.0
    if split and call == 4:
      for term in (new, head):
        term.set_group_weights({"human": weights["human"], "osl": end})
    got, want = new(env, **params), head(env, **head_params)
    _assert_identical(got, want)
    assert set(head.components) <= set(new.components)
    for key, value in head.components.items():
      _assert_identical(new.components[key], value)
    for key, value in head._cache.items():
      if value is None:
        assert new._cache[key] is None
      else:
        _assert_identical(cast(torch.Tensor, new._cache[key]), value)
    if params["clf"] is not None:
      _assert_identical(new._ed_prev, head._ed_prev)
      _assert_identical(new._clf_valid, head._clf_valid)
    if split:
      _assert_identical(new.channel_rewards, head.channel_rewards)
    if split and call == 8 and params["clf"] is not None:
      # The floor binds, and the violation sits outside it.
      assert pen_w is not None
      c = head.components
      for channel, env_id in (("human", 4), ("osl", 5)):
        floored = c[f"split_penalty_{channel}"][env_id]
        floored = floored + pen_w * c[f"split_violation_{channel}"][env_id]
        assert float(floored) == pytest.approx(-1.0)
        if pen_w > 0.0:
          assert float(c[f"split_violation_{channel}"][env_id]) > 0.0
    # The new diagnostics are there, and the regularizers did nothing.
    assert set(REGULARIZER_DIAGNOSTICS) <= set(new.components)
    assert not new._reg_cache and bool((new.components["reg_total"] == 0).all())
    rewards.append(want)
  # Not vacuous: rewards both clear of the clip and clipped (or zeroed when NaN).
  every = torch.cat(rewards)
  assert int((every > 0.05).sum()) >= 10 and bool((every == 0.0).any())


##
# The golden test: musclemimic's own environment.
##


def _golden_case(data: Any, name: str) -> tuple[Any, list[dict[str, Any]]]:
  """mjlab_eval.py's replay of a recorded case: a fake env and command per step,
  musclemimic's MjData and reference (its materialized trajectory)."""
  joint_names = [str(j) for j in data["joint_names"]][1:]  # Drop the free joint.
  nv = data[f"{name}__qvel"].shape[1]

  def t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)

  cmd: Any = SimpleNamespace(
    site_body_ids=list(range(17)),
    robot=SimpleNamespace(
      joint_names=joint_names,
      indexing=SimpleNamespace(joint_v_adr=torch.arange(6, nv)),
      data=SimpleNamespace(),
    ),
    reference_jumped=torch.zeros(1, dtype=torch.bool),
  )
  num_actions = data[f"{name}__action"].shape[1]
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=float(data["dt"]),
    command_manager=SimpleNamespace(get_term=lambda _: cmd),
    action_manager=SimpleNamespace(
      active_terms=["actions"],
      get_term=lambda _: SimpleNamespace(action_dim=num_actions),
    ),
    sim=SimpleNamespace(data=SimpleNamespace()),
  )
  steps = []
  for i in range(len(data[f"{name}__cur_step"])):

    def row(key: str, i: int = i) -> torch.Tensor:
      return t(data[f"{name}__{key}"][i])

    q, v = row("qpos")[None], row("qvel")[None]
    xpos, xmat = row("xpos"), row("xmat").reshape(17, 3, 3)
    cvel, com = row("cvel"), row("subtree_com_root")
    ang = cvel[:, :3]
    lin = cvel[:, 3:] + torch.cross(ang, xpos - com[None], dim=-1)
    # The reference frame, re-centred on the episode's first frame in xy.
    ref_q, ref_v = row("ref_qpos"), row("ref_qvel")
    spos, smat = row("ref_site_xpos"), row("ref_site_xmat").reshape(17, 3, 3)
    cvp, ref_com = row("ref_cvel_parent"), row("ref_subtree_com_root")
    s_ang = cvp[:, :3]
    s_lin = cvp[:, 3:] + torch.cross(s_ang, spos - ref_com[None], dim=-1)
    rpos, rangles, rvel = relative_site_quantities(spos, smat, s_ang, s_lin)
    v_rel, omega_rel = relative_site_velocities(smat, s_ang, s_lin)
    off = torch.zeros(1, 3, dtype=torch.float64)
    off[0, :2] = -row("ref_root_xy0")
    steps.append(
      {
        "root_qpos": q[:, :7],
        "root_qvel": v[:, :6],
        "joint_pos": q[:, 7:],
        "joint_vel": v[:, 6:],
        "sites": (xpos[None], xmat[None], ang[None], lin[None]),
        "reference": MimicReference(
          root_pos_w=ref_q[None, :3] + off,
          root_quat=ref_q[None, 3:7],
          root_lin_vel_w=ref_v[None, :3],
          root_ang_vel_b=ref_v[None, 3:6],
          joint_pos=ref_q[None, 7:],
          joint_vel=ref_v[None, 6:],
          site_pos_w=spos[None] + off[:, None, :],
          rpos=rpos[None],
          rangles=rangles[None],
          rvel=rvel[None],
          v_rel=v_rel[None],
          omega_rel=omega_rel[None],
        ),
        "action": row("action")[None],
        "qfrc_actuator": row("qfrc_actuator")[None],
        "cur_step": int(data[f"{name}__cur_step"][i]),
      }
    )
  return (cmd, env), steps


GOLDEN_PAIRS = (
  ("reward_preclip", "info_reward_preclip"),
  ("reward_qpos", "info_reward_qpos"),
  ("reward_qvel", "info_reward_qvel"),
  ("reward_root_pos", "info_reward_root_pos"),
  ("reward_root_vel", "info_reward_root_vel"),
  ("reward_rpos", "info_reward_rpos"),
  ("reward_rquat", "info_reward_rquat"),
  ("reward_rvel_rot", "info_reward_rvel_rot"),
  ("reward_rvel_lin", "info_reward_rvel_lin"),
  ("penalty_total", "info_penalty_total"),
  ("err_root_xyz", "info_err_root_xyz"),
  ("err_root_yaw", "info_err_root_yaw"),
  ("err_joint_pos", "info_err_joint_pos"),
  ("err_joint_vel", "info_err_joint_vel"),
  ("err_site_abs", "info_err_site_abs"),
  ("err_rpos", "info_err_rpos"),
)
GOLDEN_CLF_PAIRS = (
  ("clf_V", "info_clf_V"),
  ("clf_V_norm", "info_clf_V_norm"),
  ("clf_vdot", "info_clf_vdot"),
  ("clf_cross", "info_clf_cross"),
  ("clf_violation", "info_clf_violation"),
  ("clf_level", "info_reward_clf"),
)


@pytest.mark.parametrize(
  "name", ["msk_mimic", "robot_mimic", "msk_clf", "msk_clf_graded"]
)
def test_musclemimic_golden(name: str) -> None:
  """With the old flags the term computes musclemimic's own reward, recorded from
  its CPU environment (MimicReward on both models; CLFReward saturated far from the
  reference and graded near it), to 1e-6 in float64."""
  data = np.load(FIXTURE)
  (cmd, env), steps = _golden_case(data, name)
  clf = name.startswith("msk_clf")
  params = _params(
    "clf" if clf else "mimic",
    ClfCfg(decay_form="musclemimic", pen_w=0.25),
    RegularizerCfg(),  # Every coefficient 0.
  )
  term = _term(env, params)
  pairs = GOLDEN_PAIRS + (GOLDEN_CLF_PAIRS if clf else ())
  got: dict[str, list[float]] = {key: [] for key, _ in pairs}
  got["reward"] = []
  for step in steps:
    cmd.root_qpos, cmd.root_qvel = step["root_qpos"], step["root_qvel"]
    cmd.robot.data.joint_pos = step["joint_pos"]
    cmd.robot.data.joint_vel = step["joint_vel"]
    cmd.site_state = lambda step=step: step["sites"]
    cmd.reference = lambda offset, step=step: step["reference"]
    env.action_manager.action = step["action"]
    env.sim.data.qfrc_actuator = step["qfrc_actuator"]
    env.episode_length_buf = torch.tensor([step["cur_step"]])
    got["reward"].append(float(term(env, **params)[0]))
    for key, _ in pairs:
      got[key].append(float(term.components[key][0]))
  for key, recorded in (("reward", "reward"), *pairs):
    np.testing.assert_allclose(
      got[key], data[f"{name}__{recorded}"], rtol=1e-6, atol=1e-6, err_msg=key
    )
  np.testing.assert_allclose(
    got["reward"], data[f"{name}__info_reward_total"], atol=1e-6
  )
  # Not vacuous: each case holds what it is named for.
  rewards = np.asarray(got["reward"])
  if name == "msk_clf_graded":
    violation = np.asarray(got["clf_violation"])
    assert int(((violation > 0.0) & (violation < 1.0)).sum()) >= 10
  elif name == "msk_clf":
    assert int((np.asarray(got["clf_violation"]) == 1.0).sum()) >= 10
  else:
    assert bool((rewards == 0.0).any()) and int((rewards > 0.05).sum()) >= 10
    assert bool((np.asarray(got["penalty_total"]) == -1.0).any())


##
# The paper-form decrease condition.
##


def test_paper_form_matches_numpy() -> None:
  """``clip(max(Vdot_fd + 0.5 V, 0) / (4.5 V_max), 0, 1)`` with the finite
  difference of V over the step, 0 on an episode's first step, on a reference
  jump and on the term's first call; subtracted at ``pen_scale x pen_w`` inside
  the clip."""
  clf = ClfCfg()  # The default: the paper form, pen_w 0.5.
  assert (clf.decay_form, clf.pen_w, clf.decay_rate, clf.decay_norm_rate) == (
    "paper",
    0.5,
    0.5,
    4.5,
  )
  cmd = _Command(seed=2)
  env = _env(cmd)
  params = _params("clf", clf)
  term = _term(env, params)
  v_max = _v_max(clf)
  base = torch.tensor([0.002, 0.004, 0.008, 0.02, 0.05, 0.3])
  v_prev: torch.Tensor | None = None
  graded = saturated = recovering = 0
  for step in range(12):
    reset = (2,) if step == 6 else ()
    jumped = (4,) if step == 8 else ()
    if step == 9:
      clf.pen_scale = 0.5  # As an env stage sets it.
    error = base * (1.0 + 0.6 * math.sin(0.7 * step))
    _advance(env, cmd, error=error, new_reference=False, reset=reset, jumped=jumped)
    reward = term(env, **params)
    c = {k: v.double() for k, v in term.components.items()}
    v = c["clf_V"]
    first = (env.episode_length_buf <= 1) | cmd.jumped
    vdot = (torch.zeros_like(v) if v_prev is None else (v - v_prev) / DT).masked_fill(
      first, 0.0
    )
    violation = ((vdot + 0.5 * v).clamp(min=0.0) / (4.5 * v_max)).clamp(0.0, 1.0)
    violation = violation.masked_fill(first, 0.0) * float(v_prev is not None)
    torch.testing.assert_close(c["clf_vdot_fd"], vdot, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c["clf_violation"], violation, rtol=1e-5, atol=1e-6)
    assert torch.equal(c["clf_violation_saturated"], (violation >= 1.0).double())
    level = torch.exp(-(v / v_max).clamp(max=5.0))
    floored = _team_penalty(env).double()  # From the action and the OSL force.
    pen = clf.pen_scale * clf.pen_w
    torch.testing.assert_close(
      c["penalty_total"], floored - pen * violation, atol=1e-6, rtol=0
    )
    preclip = level + floored - pen * violation
    torch.testing.assert_close(
      reward.double(), preclip.clamp(min=0.0), atol=1e-5, rtol=0
    )
    expected = ((c["reward_preclip"] < 0) & (c["clf_vdot_fd"] < 0)).double()
    assert torch.equal(c["clf_clipped_recovering"], expected)
    graded += int(((violation > 0) & (violation < 1)).sum())
    saturated += int((violation == 1).sum())
    recovering += int(expected.sum())
    v_prev = v
  assert graded >= 10 and saturated >= 1 and recovering >= 1  # Not vacuous.


def _osl_dofs() -> list[int]:
  return [6 + JOINT_NAMES.index(j) for j in OSL_TORQUE_JOINTS]


def _team_penalty(env: Any) -> torch.Tensor:
  """Stage 1's floored action penalty, by hand: 0.01 x the mean squared excess of
  the action over [-1, 1] and 1e-5 x the OSL joints' squared force, floored at -1
  (no regularizers)."""
  action = env.action_manager.action
  oob = 0.01 * (action.abs() - 1.0).clamp(min=0.0).square().mean(-1)
  torque = (1e-5 * env.sim.data.qfrc_actuator[:, _osl_dofs()].square()).sum(-1)
  return (-oob - torque).clamp(min=-1.0)


def _owner_penalties(env: Any, model: str) -> dict[str, torch.Tensor]:
  """Each stage-2 channel's floored owner penalty, by hand: the out-of-bounds
  penalty over its own action rows (their mean), and on the OSL channel the OSL
  joints' force penalty, each floored at -1 (no regularizers)."""
  action = env.action_manager.action
  squared = (action.abs() - 1.0).clamp(min=0.0).square()
  osl = [7, 8] if model == "robot" else [0, 1]
  human = [r for r in range(action.shape[-1]) if r not in osl]
  torque = (1e-5 * env.sim.data.qfrc_actuator[:, _osl_dofs()].square()).sum(-1)
  return {
    "human": (-0.01 * squared[:, human].mean(-1)).clamp(min=-1.0),
    "osl": (-0.01 * squared[:, osl].mean(-1) - torque).clamp(min=-1.0),
  }


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_split_violation_scales_and_sits_outside_each_floor(model: str) -> None:
  """Each stage-2 channel subtracts ``pen_scale x pen_w x`` its violation after its
  own floored owner penalty (computed here from the actions and forces), inside the
  clip at 0: with ``pen_scale`` 0.25 and then 0, and with one channel's action rows
  far out of bounds, so its floor binds while its violation is positive."""
  cmd = _Command(seed=13)
  env = _env(cmd, model)
  clf = ClfCfg(pen_scale=0.25)  # The paper form at pen_w 0.5.
  params = _params("clf", clf)
  term = _term(env, params, split=True)
  osl = [7, 8] if model == "robot" else [0, 1]
  human = [r for r in range(env.action_manager.action.shape[-1]) if r not in osl]
  floored_with_violation = 0
  for step in range(5):
    if step == 3:
      clf.pen_scale = 0.0
    _advance(env, cmd, error=0.01 * (1.0 + 0.4 * step), new_reference=False)
    action = env.action_manager.action
    action[0, human] = 40.0  # The human channel's floor binds on env 0,
    action[1, osl] = -40.0  # the OSL channel's on env 1.
    term(env, **params)
    c = term.components
    own = _owner_penalties(env, model)
    assert float(own["human"][0]) == -1.0 and float(own["osl"][1]) == -1.0
    pen = clf.pen_scale * clf.pen_w
    for channel in CHANNELS:
      violation = c[f"split_violation_{channel}"]
      torch.testing.assert_close(
        c[f"split_penalty_{channel}"], own[channel] - pen * violation
      )
      preclip = c[f"split_preclip_{channel}"]
      torch.testing.assert_close(c[f"split_reward_{channel}"], preclip.clamp(min=0.0))
    if pen > 0.0:
      floored_with_violation += int(c["split_violation_human"][0] > 0.0)
      floored_with_violation += int(c["split_violation_osl"][1] > 0.0)
  assert floored_with_violation >= 2  # Not vacuous: a violation past a bound floor.


def test_split_fd_ignores_weight_changes() -> None:
  """Stage 2 differences each channel's V per group at the current weights: a
  weight change between two steps of the same state is no V-dot, where a stored
  channel V would show one."""
  cmd = _Command(seed=3)
  env = _env(cmd)
  params = _params("clf")
  term = _term(env, params, split=True)
  start = stage1_channel_weights(True)
  end = {"human": 0.5, "osl_sites": 1.0, "osl_joints": 10.0}
  _advance(env, cmd, error=0.01)
  _advance(env, cmd, error=0.01, new_reference=False)
  term.set_group_weights(start)
  term(env, **params)
  v_start = term.components["split_V_norm_osl"].clone()
  # The same state at the anneal's end weights.
  env.episode_length_buf += 1
  term.set_group_weights({"human": start["human"], "osl": end})
  term(env, **params)
  v_end = term.components["split_V_norm_osl"].clone()
  stored = (v_end - v_start) * _v_max(ClfCfg()) / DT  # A stored channel V's Vdot.
  assert float(stored.abs().max()) > 1.0
  for channel in CHANNELS:
    assert bool((term.components[f"split_vdot_fd_{channel}"].abs() < 1e-3).all())
  # The next step at those weights differences V at those weights.
  _advance(env, cmd, error=0.012, new_reference=False)
  term(env, **params)
  v_next = term.components["split_V_norm_osl"]
  expected = (v_next - v_end) * _v_max(ClfCfg()) / DT
  torch.testing.assert_close(
    term.components["split_vdot_fd_osl"], expected, rtol=1e-4, atol=1e-3
  )
  # At the stage-1 weights both channels are the team's.
  team, *_ = _split_at_stage1(seed=4)
  torch.testing.assert_close(
    team.components["split_vdot_fd_human"],
    team.components["clf_vdot_fd"],
    rtol=1e-4,
    atol=1e-3,
  )


def test_vdot_window_1_is_the_one_step_form() -> None:
  """``vdot_window`` 1 (the musculoskeletal default) is the one-step paper form, bit
  for bit: ``where(first or no history, 0, (V_t - V_{t-1}) / dt)``, through resets
  and reference jumps."""
  clf = ClfCfg()
  assert clf.vdot_window == 1 and clf_cfg("msk") == clf
  cmd = _Command(seed=15)
  env = _env(cmd)
  params = _params("clf", clf)
  term = _term(env, params)
  v_max = _v_max(clf)
  v_prev: torch.Tensor | None = None
  for step in range(10):
    reset = (2,) if step == 5 else ()
    jumped = (4,) if step == 7 else ()
    error = 0.01 * (1.0 + 0.5 * math.sin(0.9 * step))
    _advance(env, cmd, error=error, new_reference=False, reset=reset, jumped=jumped)
    term(env, **params)
    c = term.components
    v = c["clf_V"]
    first = (env.episode_length_buf <= 1) | cmd.jumped
    valid = torch.full_like(v, float(v_prev is not None))
    prev = torch.zeros_like(v) if v_prev is None else v_prev
    vdot = torch.where(first | (valid == 0.0), torch.zeros_like(v), (v - prev) / DT)
    norm = clf.decay_norm_rate * v_max
    violation = ((vdot + clf.decay_rate * v).clamp(min=0.0) / norm).clamp(0, 1)
    violation = violation * (~first).float() * valid
    _assert_identical(c["clf_vdot_fd"], vdot)
    _assert_identical(c["clf_violation"], violation)
    v_prev = v.clone()


def test_vdot_window_10_matches_numpy() -> None:
  """With ``vdot_window`` 10 (the robot's) the paper form differences V over 10
  steps, ``(V_t - V_{t-10}) / (10 dt)``, and its violation is 0 until 10 fresh steps
  follow an episode start or a reference jump."""
  window = 10
  clf = clf_cfg("robot")
  assert clf.vdot_window == window
  cmd = _Command(seed=16)
  env = _env(cmd, "robot")
  params = _params("clf", clf)
  term = _term(env, params)
  v_max = _v_max(clf)
  history: list[np.ndarray] = []
  since = np.zeros(NUM_ENVS, dtype=int)
  graded = masked = 0
  for step in range(32):
    reset = (2,) if step == 14 else ()
    jumped = (4,) if step == 21 else ()
    error = 0.01 * (1.0 + 0.6 * math.sin(0.4 * step))
    _advance(env, cmd, error=error, new_reference=False, reset=reset, jumped=jumped)
    term(env, **params)
    c = term.components
    v = c["clf_V"].double().numpy()
    first = ((env.episode_length_buf <= 1) | cmd.jumped).numpy() | (step == 0)
    since = np.where(first, 0, np.minimum(since + 1, window))
    fresh = since >= window
    old = history[-window] if len(history) >= window else np.zeros_like(v)
    vdot = np.where(fresh, (v - old) / (window * DT), 0.0)
    violation = np.clip(np.maximum(vdot + 0.5 * v, 0.0) / (4.5 * v_max), 0.0, 1.0)
    violation = np.where(fresh, violation, 0.0)
    np.testing.assert_allclose(c["clf_vdot_fd"].numpy(), vdot, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(c["clf_violation"].numpy(), violation, atol=1e-6)
    history.append(v)
    graded += int(((violation > 0) & (violation < 1)).sum())
    masked += int((~fresh & (0.5 * v > 0)).sum())
  # Env 2 reset at step 14 and env 4 jumped at 21: 10 masked steps each, then rates.
  assert since[2] == window and since[4] == window
  assert graded >= 10 and masked >= 20  # Not vacuous.


def test_split_windowed_fd_ignores_weight_changes() -> None:
  """Stage 2's windowed V-dot recombines the window's per-group sums at the current
  weights: a weight change inside a window of one state is no V-dot, and a moving
  state gives ``sum_g w_g (S_g,t - S_g,t-w) / (w dt sum_g w_g n_g)``."""
  window = 3
  cmd = _Command(seed=17)
  env = _env(cmd)
  params = _params("clf", ClfCfg(vdot_window=window))
  term = _term(env, params, split=True)
  counts = term._kernel_counts["clf"]
  v_max = _v_max(ClfCfg())
  end = {"human": 0.5, "osl_sites": 1.0, "osl_joints": 10.0}
  start = stage1_channel_weights(True)
  sums: list[dict[str, torch.Tensor]] = []
  for step in range(10):
    if step == 5:  # The anneal's end, in the middle of a window of one state.
      term.set_group_weights({"human": start["human"], "osl": end})
    moving = step >= 7
    error = 0.01 * (1.0 + 0.3 * (step - 6)) if moving else 0.01
    _advance(env, cmd, error=error, new_reference=step == 0)
    term(env, **params)
    c = term.components
    sums.append(
      {
        group: c[f"split_qform_{group}"].double() * counts[group] * v_max
        for group in GROUPS
      }
    )
    if 3 <= step <= 6:  # A full window of one state: no rate, before or after.
      for channel in CHANNELS:
        assert float(c[f"split_vdot_fd_{channel}"].abs().max()) < 1e-3
    if moving:
      weights = term.group_weights["osl"]
      den = sum(weights[g] * counts[g] for g in GROUPS)
      num = torch.stack(
        [weights[g] * (sums[-1][g] - sums[-1 - window][g]) for g in GROUPS]
      ).sum(0)
      expected = num / (window * DT * den)
      torch.testing.assert_close(
        c["split_vdot_fd_osl"].double(), expected, rtol=1e-3, atol=1e-2
      )
      assert float(expected.abs().max()) > 1.0  # Not vacuous: V moves.


CLF_TASK_FAMILIES = (
  "Mjlab-Mimic-OSL-Fullbody{}-CLF",
  "Mjlab-Gait-Library-Unicycle-OSL-Fullbody{}-CLF",
  "Mjlab-Split-Unicycle-OSL-Fullbody{}-CLF",
  "Mjlab-FSM-Unicycle-OSL-Fullbody{}-CLF",
  "Mjlab-Stairs-OSL-Fullbody{}-CLF",
  "Mjlab-Split-Stairs-OSL-Fullbody{}-CLF",
)


@pytest.mark.parametrize("family", CLF_TASK_FAMILIES)
def test_model_aware_clf_defaults(family: str) -> None:
  """Every CLF task family: the robot differences V over 10 steps and logs its servo
  costs; the musculoskeletal model over 1 step, with nothing to log."""
  for tag, window in (("-ROBOT", 10), ("-MSK", 1)):
    params = load_env_cfg(family.format(tag)).rewards["mimic"].params
    assert params["clf"].vdot_window == window
    assert params["regularizers"].log_servo_terms == (tag == "-ROBOT")
  mimic = load_env_cfg(family.format("-ROBOT").replace("-CLF", "")).rewards["mimic"]
  assert mimic.params["clf"] is None
  assert not mimic.params["regularizers"].log_servo_terms


def _split_at_stage1(seed: int) -> tuple[split_mimic_reward, Any, _Command]:
  cmd = _Command(seed=seed)
  env = _env(cmd)
  params = _params("clf")
  term = _term(env, params, split=True)
  for step in range(3):
    _advance(env, cmd, error=0.01 * (1.0 + 0.1 * step), new_reference=step == 0)
    term(env, **params)
  return term, env, cmd


##
# Regularizers.
##


def _clipped_rate(env: Any, rows: list[int]) -> torch.Tensor:
  manager = env.action_manager
  now = manager.action[:, rows].clamp(-1.0, 1.0)
  before = manager.prev_action[:, rows].clamp(-1.0, 1.0)
  return (now - before).square().mean(-1)


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_action_rates(model: str) -> None:
  """The OSL rows (and on the robot the servo rows) by term name, on the clipped
  action, 0 on an episode's first step; inside the -1 floor at ``scale``. Stage 2
  charges the OSL rows to the OSL channel and the servo rows to the human's."""
  cmd = _Command(seed=5)
  env = _env(cmd, model)
  regularizers = RegularizerCfg(scale=0.5, osl_action_rate=2.0, servo_action_rate=3.0)
  params = _params("mimic", regularizers=regularizers)
  stage1 = _term(env, copy.deepcopy(params))
  split_params = copy.deepcopy(params)
  stage2 = _term(env, split_params, split=True)
  _advance(env, cmd)
  _advance(env, cmd, reset=(1,))
  env.action_manager.action[0] = 3.0  # Clipped: no change from a clipped 2.0.
  env.action_manager.prev_action[0] = 2.0
  stage1(env, **params)
  stage2(env, **split_params)

  osl = list(range(7, 9)) if model == "robot" else [0, 1]
  servos = list(range(7)) if model == "robot" else []
  first = env.episode_length_buf <= 1
  osl_cost = (2.0 * _clipped_rate(env, osl)).masked_fill(first, 0.0)
  assert float(osl_cost[0]) == 0.0 and float(osl_cost[1]) == 0.0
  assert int((osl_cost > 0).sum()) >= 3
  c = stage1.components
  torch.testing.assert_close(c["reg_osl_action_rate"], osl_cost)
  if servos:
    servo_cost = (3.0 * _clipped_rate(env, servos)).masked_fill(first, 0.0)
  else:
    servo_cost = torch.zeros(NUM_ENVS)  # The musculoskeletal model has none.
  torch.testing.assert_close(c["reg_servo_action_rate"], servo_cost)
  torch.testing.assert_close(c["reg_total"], 0.5 * (osl_cost + servo_cost))
  action = env.action_manager.action
  oob = -0.01 * (action.abs() - 1.0).clamp(min=0.0).square().mean(-1)
  qfrc = env.sim.data.qfrc_actuator[
    :, [6 + JOINT_NAMES.index(j) for j in OSL_TORQUE_JOINTS]
  ]
  torque = (1e-5 * qfrc.square()).sum(-1)
  expected = (oob - torque - 0.5 * (osl_cost + servo_cost)).clamp(min=-1.0)
  torch.testing.assert_close(c["penalty_total"], expected)

  s = stage2.components
  torch.testing.assert_close(s["split_regularizers_osl"], 0.5 * osl_cost)
  torch.testing.assert_close(s["split_regularizers_human"], 0.5 * servo_cost)
  human = [r for r in range(action.shape[-1]) if r not in osl]
  for channel, rows, cost, owned in (
    ("osl", osl, osl_cost, torque),
    ("human", human, servo_cost, torch.zeros(NUM_ENVS)),
  ):
    own_oob = -0.01 * (action[:, rows].abs() - 1.0).clamp(min=0.0).square().mean(-1)
    penalty = (own_oob - owned - 0.5 * cost).clamp(min=-1.0)
    torch.testing.assert_close(s[f"split_penalty_{channel}"], penalty)


def test_joint_limits() -> None:
  """The watched joints' excess over their hard limits (per env), summed in rad;
  stage 2 charges the OSL knee and ankle to the OSL channel, the rest to the
  human's. An unknown joint is refused."""
  cmd = _Command(seed=6)
  env = _env(cmd)
  regularizers = RegularizerCfg(joint_limits=1.5)
  assert regularizers.joint_limit_joints == LIMIT_JOINTS
  params = _params("clf", regularizers=regularizers)
  stage1 = _term(env, copy.deepcopy(params))
  split_params = copy.deepcopy(params)
  stage2 = _term(env, split_params, split=True)
  _advance(env, cmd)
  q = torch.zeros(NUM_ENVS, len(JOINT_NAMES))
  excess = {
    "hip_flexion_r": 0.1,
    "knee_angle_l": -0.2,
    "OSL_KA_L1_osl_knee_angle_r": -0.05,
    "OSL_KA_L1_osl_ankle_angle_r": 0.3,
    "subtalar_angle_l": 0.4,  # Not watched.
    "lumbar_extension": -0.4,  # Not watched.
  }
  for joint, over in excess.items():
    q[:, JOINT_NAMES.index(joint)] = math.copysign(1.0, over) + over
  q[0] = 0.0  # Inside every limit.
  cmd.robot.data.joint_pos = q
  stage1(env, **params)
  stage2(env, **split_params)

  limits = cmd.robot.data.joint_pos_limits  # +-1, +-1.5 on the last two envs.
  per_joint = (limits[..., 0] - q).clamp(min=0.0) + (q - limits[..., 1]).clamp(min=0.0)
  watched = [JOINT_NAMES.index(j) for j in LIMIT_JOINTS]
  osl = [JOINT_NAMES.index(j) for j in OSL_TORQUE_JOINTS]
  human = [j for j in watched if j not in osl]
  total = 1.5 * per_joint[:, watched].sum(-1)
  assert float(total[0]) == 0.0 and float(total[1]) == pytest.approx(1.5 * 0.65)
  assert float(total[-1]) == 0.0  # The wider limits hold every joint.
  torch.testing.assert_close(stage1.components["reg_joint_limits"], total)
  s = stage2.components
  torch.testing.assert_close(
    s["split_reg_joint_limits_osl"], 1.5 * per_joint[:, osl].sum(-1)
  )
  torch.testing.assert_close(
    s["split_reg_joint_limits_human"], 1.5 * per_joint[:, human].sum(-1)
  )
  torch.testing.assert_close(
    s["split_regularizers_osl"] + s["split_regularizers_human"], total
  )

  bad = _params("clf", regularizers=RegularizerCfg(joint_limit_joints=("wrist",)))
  with pytest.raises(ValueError, match="unknown"):
    _term(env, bad)
  negative = _params("clf", regularizers=RegularizerCfg(joint_limits=-1.0))
  with pytest.raises(ValueError, match="joint_limits"):
    _term(env, negative)


def _servo_actuator(ids: list[int], field: str) -> XmlActuator:
  act = XmlActuator.__new__(XmlActuator)
  act._command_field = field  # pyright: ignore[reportAttributeAccessIssue]
  act._global_ctrl_ids = torch.tensor(ids)  # pyright: ignore[reportAttributeAccessIssue]
  return act


def _mujoco_servos() -> SimpleNamespace:
  """The robot model in a state where some servos are force-limited: MuJoCo's own
  actuation, with its unclamped demand (the same state, force limits off), and a
  fake robot env reading that state, whose OSL rows show a huge force (control
  units on the robot) that no servo term may read. The expected unweighted
  ``torque`` and ``saturation`` costs come from MuJoCo."""
  model = get_spec("robot", "rigid").compile()
  names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
  servo = [i for i, name in enumerate(names) if name in SERVO_JOINTS]
  osl = [i for i, name in enumerate(names) if name in OSL_TORQUE_JOINTS]
  assert len(servo) == 27 and len(osl) == 2
  data = mujoco.MjData(model)
  rng = np.random.default_rng(0)
  mujoco.mj_forward(model, data)
  # A delayed target: half the servos hold targets close to where they are, the
  # other half targets far from it.
  lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
  far = np.where(np.arange(model.nu) % 2 == 0, 0.01, 1.5)
  offset = far * rng.choice([-1.0, 1.0], model.nu)
  data.ctrl[:] = np.clip(data.actuator_length + offset, lo, hi)
  data.qvel[6:] = rng.normal(0.0, 0.05, model.nv - 6)
  mujoco.mj_forward(model, data)
  # MuJoCo's demand before the force limit: the same state, limits off.
  free = mujoco.MjData(model)
  free.qpos[:], free.qvel[:], free.ctrl[:] = data.qpos, data.qvel, data.ctrl
  unlimited = copy.deepcopy(model)
  unlimited.actuator_forcelimited[:] = 0
  mujoco.mj_forward(unlimited, free)

  def batch(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)[None].repeat(
      NUM_ENVS, *([1] * np.ndim(x))
    )

  cmd = _Command(seed=7)
  cmd.robot.actuators = [
    _servo_actuator(servo, "position"),
    _servo_actuator(osl, "effort"),
  ]
  env = _env(cmd, "robot")
  env.sim.mj_model = model
  env.sim.data = SimpleNamespace(
    qfrc_actuator=torch.zeros(NUM_ENVS, 6 + len(JOINT_NAMES)),
    actuator_force=batch(data.actuator_force),
    actuator_length=batch(data.actuator_length),
    actuator_velocity=batch(data.actuator_velocity),
    ctrl=batch(data.ctrl),
  )
  env.sim.model = SimpleNamespace(
    actuator_forcerange=batch(model.actuator_forcerange),
    actuator_gainprm=batch(model.actuator_gainprm),
    actuator_biasprm=batch(model.actuator_biasprm),
  )
  # The OSL rows' force is in control units on the robot; it must not count.
  env.sim.data.actuator_force[:, osl] = 1e4
  force_range = model.actuator_forcerange[servo, 1]
  demand = free.actuator_force[servo]
  saturated = np.abs(demand) > force_range
  assert 3 <= int(saturated.sum()) < 27  # Some servos at their limit, not all.
  np.testing.assert_allclose(
    np.abs(data.actuator_force[servo][saturated]), force_range[saturated]
  )
  return SimpleNamespace(
    env=env,
    cmd=cmd,
    servo=servo,
    torque=float(np.mean((data.actuator_force[servo] / force_range) ** 2)),
    saturation=float(
      np.mean(np.maximum(np.abs(demand) - force_range, 0.0) / force_range)
    ),
  )


def test_servo_terms_against_mujoco() -> None:
  """The robot's servo torque and saturation from MuJoCo's own actuation: torque
  from the servos' actuator force (gear 1) over its range, the PD demand from
  ``data.ctrl``, never from the OSL motors (geared on the robot); per-world gains."""
  scene = _mujoco_servos()
  env, cmd, servo = scene.env, scene.cmd, scene.servo
  torque, saturation = scene.torque, scene.saturation
  regularizers = RegularizerCfg(servo_torque=2.0, servo_saturation=3.0)
  params = _params("mimic", regularizers=regularizers)
  term = _term(env, params)
  _advance(env, cmd)
  term(env, **params)
  c = term.components
  torch.testing.assert_close(
    c["reg_servo_torque"], torch.full((NUM_ENVS,), 2.0 * torque, dtype=torch.float64)
  )
  torch.testing.assert_close(
    c["reg_servo_saturation"],
    torch.full((NUM_ENVS,), 3.0 * saturation, dtype=torch.float64),
  )
  # Per-world gains: a stiffer servo on env 0 demands more (Kp scales the target
  # error); the demand reads data.ctrl, whatever the action.
  model_t = env.sim.model
  model_t.actuator_gainprm[0, servo, 0] *= 2.0
  model_t.actuator_biasprm[0, servo, 1] *= 2.0
  env.action_manager.action = torch.zeros_like(env.action_manager.action)
  term(env, **params)
  stiff = term.components
  assert float(stiff["reg_servo_saturation"][0]) > 3.0 * saturation
  torch.testing.assert_close(
    stiff["reg_servo_saturation"][1:],
    torch.full((NUM_ENVS - 1,), 3.0 * saturation, dtype=torch.float64),
  )
  # A model without servos (the musculoskeletal one): both terms are 0.
  bare = _Command(seed=8)
  msk_env = _env(bare)
  msk_term = _term(msk_env, _params("mimic", regularizers=regularizers))
  _advance(msk_env, bare)
  msk_term(msk_env, **_params("mimic", regularizers=regularizers))
  for name in ("reg_servo_torque", "reg_servo_saturation", "reg_total"):
    assert bool((msk_term.components[name] == 0.0).all())


def test_servo_costs_are_logged_but_never_rewarded() -> None:
  """``log_servo_terms`` (on in the robot's CLF defaults, whose servo torque terms
  are 0) reports the servos' torque and saturation costs, unweighted, on every step,
  and leaves the reward and every other component bit for bit what they are without
  the log, a NaN in the log included. A model without servos logs nothing."""
  scene = _mujoco_servos()
  env, cmd = scene.env, scene.cmd
  regularizers = _no_contact(clf_regularizers("robot"))
  assert regularizers.log_servo_terms and regularizers.joint_limits > 0.0
  assert regularizers.servo_torque == regularizers.servo_saturation == 0.0
  silent = copy.deepcopy(regularizers)
  silent.log_servo_terms = False
  logged = _params("clf", clf_cfg("robot"), regularizers)
  plain = _params("clf", clf_cfg("robot"), silent)
  log_term, plain_term = _term(env, logged), _term(env, plain)
  for step in range(3):
    _advance(env, cmd, error=0.02 * (1 + step), new_reference=step == 0)
    if step == 2:
      env.sim.data.actuator_force[0, scene.servo[0]] = float("nan")
    reward = log_term(env, **logged)
    _assert_identical(reward, plain_term(env, **plain))
    c = log_term.components
    assert set(c) - set(plain_term.components) == set(SERVO_LOG_DIAGNOSTICS)
    for key, value in plain_term.components.items():
      _assert_identical(c[key], value)
    if step < 2:
      torch.testing.assert_close(
        c["servo_torque_cost"], torch.full((NUM_ENVS,), scene.torque).double()
      )
      torch.testing.assert_close(
        c["servo_saturation_cost"],
        torch.full((NUM_ENVS,), scene.saturation).double(),
      )
      assert scene.torque > 0.0 and scene.saturation > 0.0
    else:  # A NaN servo force: in the log only.
      assert bool(torch.isnan(c["servo_torque_cost"][0]))
      assert bool(torch.isfinite(reward).all())
      assert float(c["reg_total"].abs().max()) > 0.0  # The other terms act.
  # The musculoskeletal model has no servos: nothing logged, nothing changed.
  bare = _Command(seed=14)
  msk_env = _env(bare)
  msk = _params("clf", clf_cfg("msk"), RegularizerCfg(log_servo_terms=True))
  msk_plain = _params("clf", clf_cfg("msk"), RegularizerCfg())
  msk_term, msk_plain_term = _term(msk_env, msk), _term(msk_env, msk_plain)
  _advance(msk_env, bare)
  _assert_identical(msk_term(msk_env, **msk), msk_plain_term(msk_env, **msk_plain))
  assert not set(SERVO_LOG_DIAGNOSTICS) & set(msk_term.components)


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_activation_energy_metric(model: Model) -> None:
  """``mean(act^2)`` over the muscles, a metric only; the robot (na = 0) has none,
  and its reward stays finite."""
  mj_model = get_spec(model, "rigid").compile()
  cmd = _Command(seed=9)
  env = _env(cmd)
  params = _params("clf", regularizers=_no_contact(clf_regularizers(model)))
  plain = _term(env, copy.deepcopy(params))  # Built without a model: no metric.
  env.sim.mj_model = mj_model
  term = _term(env, params)
  act = torch.rand(NUM_ENVS, mj_model.na, generator=cmd.gen)
  env.sim.data.act = act
  _advance(env, cmd)
  reward = term(env, **params)
  assert bool(torch.isfinite(reward).all())
  # Never part of the reward.
  _assert_identical(plain(env, **params), reward)
  if model == "robot":
    assert mj_model.na == 0 and "activation_energy" not in term.components
    return
  muscles = [
    int(mj_model.actuator_actadr[i])
    for i in range(mj_model.nu)
    if mj_model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
  ]
  assert len(muscles) == 339
  torch.testing.assert_close(
    term.components["activation_energy"], act[:, muscles].square().mean(-1)
  )
  assert "activation_energy" not in plain.components


##
# Env stages.
##


def _staged_env(term_cfg: RewardTermCfg) -> Any:
  return SimpleNamespace(
    reward_manager=SimpleNamespace(get_term_cfg=lambda name: term_cfg),
    termination_manager=SimpleNamespace(get_term_cfg=lambda name: None),
  )


def test_reward_param_stages_act_on_the_live_term() -> None:
  """``rewards.<term>.params.<param>[.<field>]`` set the config objects the term
  reads on every step, so a staged ``pen_scale`` or regularizer ``scale`` holds
  from the next step on."""
  cmd = _Command(seed=10)
  env = _env(cmd)
  params = _params("clf", regularizers=RegularizerCfg(osl_action_rate=1.0))
  term_cfg = RewardTermCfg(func=mimic_reward, weight=1.0, params=params)
  term = mimic_reward(term_cfg, env)
  staged = _staged_env(term_cfg)
  pen_get, pen_set = stage_accessors(staged, "rewards.mimic.params.clf.pen_scale")
  reg_get, reg_set = stage_accessors(staged, "rewards.mimic.params.regularizers.scale")
  assert pen_get() == reg_get() == 1.0
  for step in range(4):
    if step == 3:
      pen_set(0.25)
      reg_set(2.0)
    _advance(env, cmd, error=0.02 * (1.0 + 0.3 * step), new_reference=False)
    term(env, **params)
  assert params["clf"].pen_scale == 0.25 and params["regularizers"].scale == 2.0
  c = term.components
  rate = c["reg_osl_action_rate"]
  assert float(rate.min()) > 0.0 and float(c["clf_violation"].max()) > 0.0
  torch.testing.assert_close(c["reg_total"], 2.0 * rate)
  action = env.action_manager.action
  oob = -0.01 * (action.abs() - 1.0).clamp(min=0.0).square().mean(-1)
  dofs = [6 + JOINT_NAMES.index(j) for j in OSL_TORQUE_JOINTS]
  torque = (1e-5 * env.sim.data.qfrc_actuator[:, dofs].square()).sum(-1)
  floored = (oob - torque - 2.0 * rate).clamp(min=-1.0)
  torch.testing.assert_close(
    c["penalty_total"], floored - 0.25 * 0.5 * c["clf_violation"]
  )
  # Only what the term reads on every step can be staged: every other param is read
  # once, at build, and a staged value would silently do nothing.
  assert "clf.c_viol" not in mimic_reward.LIVE_PARAMS
  assert "clf.vdot_window" not in mimic_reward.LIVE_PARAMS  # Sizes the history.
  for name in mimic_reward.LIVE_PARAMS:
    get, _ = stage_accessors(staged, f"rewards.mimic.params.{name}")
    assert isinstance(get(), float)
  for path in (
    "rewards.mimic.params.action_out_of_bounds_coeff",
    "rewards.mimic.params.weights",
    "rewards.mimic.params.clf.c_viol",
    "rewards.mimic.params.clf.eps_dead",
    "rewards.mimic.params.clf.alpha_safety",
    "rewards.mimic.params.clf.decay_form",
    "rewards.mimic.params.clf.vdot_window",
    "rewards.mimic.params.regularizers.joint_limit_joints",
    "rewards.mimic.params.regularizers.log_servo_terms",
    "rewards.mimic.params.channels",
    "rewards.mimic.params.nothing",
    "rewards.mimic.params.clf.pen_scale.x",
  ):
    with pytest.raises(ValueError):
      stage_accessors(staged, path)
  with pytest.raises(ValueError, match="once, when it is built"):
    stage_accessors(staged, "rewards.mimic.params.clf.c_viol")
  # A live param a MimicReward term does not have (no CLF).
  mimic_cfg = RewardTermCfg(func=mimic_reward, weight=1.0, params=_params("mimic"))
  with pytest.raises(KeyError, match="no field"):
    stage_accessors(_staged_env(mimic_cfg), "rewards.mimic.params.clf.pen_scale")
  # A function term reads every param on every call: any number may be staged.
  twist = RewardTermCfg(
    func=_twist_like, weight=0.2, params={"command_name": "motion", "std": 0.1}
  )
  std_get, std_set = stage_accessors(_staged_env(twist), "rewards.twist.params.std")
  std_set(0.3)
  assert std_get() == 0.3 and twist.params["std"] == 0.3
  with pytest.raises(TypeError, match="not a number"):
    stage_accessors(_staged_env(twist), "rewards.twist.params.command_name")


def _twist_like(env: Any, command_name: str, std: float) -> torch.Tensor:
  del command_name
  return torch.full((env.num_envs,), std)


@pytest.mark.parametrize(
  "update, value", [(0, 0.0), (499, 0.0), (500, 0.5), (999, 0.5), (1000, 1.0)]
)
def test_clf_ramps(update: int, value: float) -> None:
  """Every CLF task's stage-1 ramps: off until update 500, half until 1000."""
  params = _params("clf", regularizers=clf_regularizers("msk"))
  term_cfg = RewardTermCfg(func=mimic_reward, weight=1.0, params=params)
  runner = SimpleNamespace(
    env=SimpleNamespace(unwrapped=_staged_env(term_cfg)),
    alg=SimpleNamespace(num_updates_done=update),
    env_stages=CLF_ENV_STAGES,
    _stage_defaults={},
  )
  applied = OslMimicOnPolicyRunner.apply_env_stages(cast(Any, runner))
  assert applied == {
    "rewards.mimic.params.clf.pen_scale": value,
    "rewards.mimic.params.regularizers.scale": value,
  }
  assert params["clf"].pen_scale == params["regularizers"].scale == value
  # An explicit update count (stage 2's k) overrides the algorithm's.
  applied = OslMimicOnPolicyRunner.apply_env_stages(cast(Any, runner), 0)
  assert set(applied.values()) == {0.0}


def test_diagnostic_keys() -> None:
  """Every split diagnostic is reported, the new ones included."""
  cmd = _Command(seed=11)
  env = _env(cmd)
  params = _params("clf", regularizers=_no_contact(clf_regularizers("msk")))
  term = _term(env, params, split=True)
  _advance(env, cmd)
  _advance(env, cmd, new_reference=False)
  term(env, **params)
  assert set(SPLIT_DIAGNOSTICS) <= set(term.components)
  for key in SPLIT_DIAGNOSTICS:
    assert bool(torch.isfinite(term.components[key]).all()), key


def test_relative_site_rvel_is_bit_identical() -> None:
  """``relative_site_rvel`` (the gait library's served rvel) is
  ``relative_site_quantities``' rvel (the reward's), bit for bit."""
  cmd = _Command(seed=12)
  pos, mat, ang, lin = cmd.site_state()
  *_, rvel = relative_site_quantities(pos, mat, ang, lin)
  _assert_identical(relative_site_rvel(mat, ang, lin), rvel)


##
# Across launches: the servo-cost metrics, the resume check, an evaluation's reward.
##

SERVO_LOG_FLAG = "--env.rewards.mimic.params.regularizers.log-servo-terms"
MSK_UNICYCLE = "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-MSK-Compliant"


def _train_cfg(task: str, *argv: str) -> TrainConfig:
  """The config ``train.py`` builds: the task's, with ``argv`` applied by tyro."""
  return tyro.cli(
    TrainConfig,
    args=list(argv),
    default=TrainConfig.from_task(task),
    config=mjlab.TYRO_FLAGS,
  )


@pytest.mark.parametrize(
  ("task", "value", "logged"),
  [
    ("Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT-CLF", None, True),
    ("Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT-CLF", "False", False),
    ("Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT", None, False),
    ("Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT", "True", True),
    ("Mjlab-FSM-Unicycle-OSL-Fullbody-ROBOT", "True", True),
    ("Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant", "True", False),
  ],
)
def test_servo_cost_metrics_follow_the_final_config(
  task: str, value: str | None, logged: bool
) -> None:
  """The servo-cost metrics are registered from the config ``train.py`` ends with:
  with ``log_servo_terms`` on a robot (the CLF tasks' default, or the flag on a
  MimicReward task), not with it off, and never on the musculoskeletal model, which
  has no servos to log."""
  cfg = _train_cfg(task, *(() if value is None else (SERVO_LOG_FLAG, value))).env
  if value is not None:
    log = cfg.rewards["mimic"].params["regularizers"].log_servo_terms
    assert log == (value == "True")
  registered = {name for name in SERVO_LOG_DIAGNOSTICS if name in cfg.metrics}
  assert registered == (set(SERVO_LOG_DIAGNOSTICS) if logged else set())


def test_rebuild_flags_parse_back_exactly() -> None:
  """``reward_flags`` writes numbers exactly: settings with more than 6 significant
  digits parse back to themselves through the command line, so the resume check
  accepts the flags it suggests."""
  task = "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT-CLF-Compliant"
  base = env_reward_settings(TrainConfig.from_task(task).env)
  assert base is not None
  fine = dict(
    base,
    clf_pen_w=0.1234567,
    clf_decay_rate=1.0 / 3.0,
    reg_osl_action_rate=1.0 / 3.0,
    reg_scale=0.7654321,
  )
  stages = [[123, "rewards.mimic.params.clf.pen_scale", 1.0 / 7.0]]
  flags = reward_flags(fine, stages)
  parsed = _train_cfg(task, *shlex.split(" ".join(flags)))
  live = env_reward_settings(parsed.env)
  assert live is not None
  assert (live["clf_pen_w"], live["reg_osl_action_rate"]) == (0.1234567, 1.0 / 3.0)
  assert live["clf_vdot_window"] == base["clf_vdot_window"] == 10
  record = {"config": fine, "env_stages": stages}
  resumed_stages = cast(Any, parsed.agent).env_stages
  assert resume_reward_mismatches(record, live, resumed_stages, 0) == []


def test_the_resume_check_compares_clf_presence() -> None:
  """A MimicReward run under a CLF task (or the reverse) differs whatever the
  regularizers: the check says so and names the task to resume it under, which no
  flag switches. The warm start, whose reward switch handles that case, leaves the
  line out."""
  mimic = load_env_cfg(MSK_UNICYCLE)
  clf = load_env_cfg(MSK_UNICYCLE.replace("-MSK-", "-MSK-CLF-"))
  mimic_record, clf_record = reward_record(mimic, ()), reward_record(clf, ())
  assert mimic_record is not None and clf_record is not None
  clf_live = env_reward_settings(clf)
  assert clf_live is not None
  presence = "reward MimicReward (no CLF) vs CLF (paper decrease condition)"
  assert resume_reward_mismatches(mimic_record, clf_live, (), 0)[0] == presence
  with pytest.raises(ValueError, match=re.escape(presence)) as refused:
    check_resume_reward("run/model_9.pt", {REWARD_KEY: mimic_record}, clf, (), 0, False)
  assert "under its MimicReward task (this task without -CLF)" in str(refused.value)
  with pytest.raises(ValueError, match="resume it under its CLF task"):
    check_resume_reward("run/model_9.pt", {REWARD_KEY: clf_record}, mimic, (), 0, False)
  # Equal regularizers: only the CLF differs, and only the resume check says so.
  mimic_settings = dict(mimic_record["config"])
  same_regs = {**clf_live, **{k: v for k, v in mimic_settings.items() if "reg_" in k}}
  assert reward_mismatches(mimic_settings, same_regs) == [presence]
  assert reward_mismatches(mimic_settings, same_regs, clf_presence=False) == []


def test_an_unreadable_run_reward_is_said_so(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """A checkpoint from before the record whose run's params/ is not beside it (a
  W&B download): the check says its run's reward cannot be read, not that it
  differs, and how to go on; allowed, it trains on with a warning."""
  cfg = load_env_cfg(MSK_UNICYCLE.replace("-MSK-", "-MSK-CLF-"))
  path = tmp_path / "wandb_checkpoints" / "abc123" / "model_5998.pt"
  with pytest.raises(ValueError, match="cannot be read") as refused:
    check_resume_reward(path, {}, cfg, (), 5999, False)
  message = str(refused.value)
  params = path.parent / "params"
  assert f"Copy the run's params/ folder (env.yaml and agent.yaml) to {params}" in (
    message
  )
  assert "--agent.allow-reward-mismatch True" in message
  assert "another reward" not in message and "its run's params (" not in message
  check_resume_reward(path, {}, cfg, (), 5999, True)
  out = capsys.readouterr().out
  assert "[WARN]" in out and "cannot be read" in out and "unchecked" in out


def test_an_evaluation_scores_under_the_runs_reward(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """``evaluate.py`` gives the task's config its run's reward: a checkpoint's own
  record, else (a run from before 2026-11-18) its params', which then rebuild that
  run's reward as the old flags do, in new config objects."""
  task = MSK_UNICYCLE.replace("-MSK-", "-MSK-CLF-")
  run = tmp_path / "r6"
  clf = {"pen_w": 0.0, "tau": 0.1}
  dump_yaml(
    run / "params" / "env.yaml", {"rewards": {"mimic": {"params": {"clf": clf}}}}
  )
  torch.save({"infos": {}}, run / "model_5998.pt")
  cfg = load_env_cfg(task)
  before = cfg.rewards["mimic"].params["clf"]
  apply_run_reward(cfg, run / "model_5998.pt")
  params = cfg.rewards["mimic"].params
  assert params["clf"] is not before and before.decay_form == "paper"
  scored = params["clf"]
  assert (scored.decay_form, scored.pen_w, scored.pen_scale, scored.vdot_window) == (
    "musclemimic",
    0.0,
    1.0,
    1,
  )
  assert all(getattr(params["regularizers"], name) == 0.0 for name in REGULARIZERS)
  assert "(from its run's params/ (it predates the record))" in capsys.readouterr().out
  # A checkpoint's record wins; log_servo_terms, a diagnostic, stays the config's.
  robot = "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT-CLF"
  recorded = load_env_cfg(robot)
  recorded.rewards["mimic"].params["clf"].pen_w = 0.3
  recorded.rewards["mimic"].params["regularizers"].joint_limits = 2.0
  record = reward_record(recorded, CLF_ENV_STAGES)
  torch.save({"infos": {REWARD_KEY: record}}, run / "model_6000.pt")
  cfg = load_env_cfg(robot)
  apply_run_reward(cfg, run / "model_6000.pt")
  assert record is not None and env_reward_settings(cfg) == record["config"]
  assert cfg.rewards["mimic"].params["regularizers"].log_servo_terms
  assert "(from its record)" in capsys.readouterr().out
  # A MimicReward run's settings do not fit a CLF task.
  with pytest.raises(ValueError, match="MimicReward"):
    apply_reward_settings(load_env_cfg(task), reward_settings(None, RegularizerCfg()))
