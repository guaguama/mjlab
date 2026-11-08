"""musclemimic's ``osl_fullbody`` imitation configs, ported to mjlab.

Sources (musclemimic ``dev``): ``fullbody/conf_osl_fullbody.yaml`` (musculoskeletal
model, MimicReward), ``conf_osl_fullbody_clf.yaml`` (the same with CLFReward) and
``conf_osl_fullbody_robot.yaml`` (the position-servo twin, MimicReward). The robot
CLF variant has no musclemimic config: it applies ``conf_osl_fullbody_clf.yaml``'s
change to the robot config, as musclemimic's v2 robot CLF run did on the CLI.

What carries over unchanged:

* simulation: 2 ms Euler steps, 5 per action (100 Hz), 4 solver, 8 line-search and
  35 CCD iterations, Euler damping disabled on the rigid socket only (the compliant
  socket needs it implicit), and the model's own contact settings (87 colliding geoms, 46
  explicit pairs, 1 mm margins) with multi-contact CCD off, as in MuJoCo 3.4;
* starts: the exact reference state at a uniformly random frame, with no noise,
  pushes or randomization; episodes end at the clip's last frame;
* observations, actions, reward and terminations, term for term (see ``mdp``).
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Literal

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  CONTACT_CONE,
  CONTACT_IMPRATIO,
  MUSCLE_TENDONS_EXPR,
  OSL_TORQUE_JOINTS,
  SERVO_JOINTS,
  Model,
  Socket,
  get_osl_msk_cfg,
  get_osl_robot_cfg,
  get_spec,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import (
  JointEffortActionCfg,
  JointPositionActionCfg,
  TendonEffortActionCfg,
)
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.osl_mimic import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

Reward = Literal["mimic", "clf"]
Mode = Literal["train", "validation", "play"]

TIMESTEP = 0.002
DECIMATION = 5
"""100 Hz control, so motion clips must be 100 Hz."""
EPISODE_LENGTH_S = 300.0
"""musclemimic's ``horizon: 30000`` steps. Clip ends terminate episodes long before."""
NUM_ENVS = 4096

# conf_osl_fullbody.yaml reward_params (the seven *_w_sum weights).
MIMIC_WEIGHTS: dict[str, float] = {
  "qpos": 0.2,
  "qvel": 0.1,
  "root_pos": 0.1,
  "root_vel": 0.1,
  "rpos": 0.6,
  "rquat": 0.01,
  "rvel": 0.1,
}
# conf_osl_fullbody_clf.yaml: the CLF term is the only tracking signal.
CLF_WEIGHTS: dict[str, float] = dict.fromkeys(MIMIC_WEIGHTS, 0.0)
# Keeps the geared OSL motors from driving the model into the floor early on.
OSL_TORQUE_WEIGHTS: dict[str, float] = dict.fromkeys(OSL_TORQUE_JOINTS, 1e-5)
OSL_MOTOR_CTRL = 2.88
"""The OSL motors' ctrlrange half-width; DefaultControl maps [-1, 1] onto it."""

# Training terminates on all three; validation and play on the site check alone, at
# 0.5 m (musclemimic's validation handler).
TRAIN_THRESHOLDS = {"site": 1.0, "root": 1.0, "root_ori": 1.0}
VALIDATION_SITE_THRESHOLD = 0.5

DIAGNOSTICS = (
  "reward_qpos",
  "reward_qvel",
  "reward_root_pos",
  "reward_root_vel",
  "reward_rpos",
  "reward_rquat",
  "reward_rvel_rot",
  "reward_rvel_lin",
  "penalty_total",
  "reward_preclip",
  "reward_clipped",
  "err_root_xyz",
  "err_root_yaw",
  "err_joint_pos",
  "err_joint_vel",
  "err_site_abs",
  "err_rpos",
)
CLF_DIAGNOSTICS = ("clf_V", "clf_V_norm", "clf_violation", "clf_level", "clf_vdot")


@lru_cache(maxsize=1)
def _servo_ranges() -> dict[str, tuple[float, float]]:
  """Each servo's joint range: the span musclemimic's normalized command covers.

  One servo table serves both sockets, so the rigid model is read.
  """
  spec = get_spec("robot", "rigid")
  return {
    str(a.target): (float(a.ctrlrange[0]), float(a.ctrlrange[1]))
    for a in spec.actuators
    if a.target in SERVO_JOINTS
  }


def _actions(model: Model) -> dict[str, ActionTermCfg]:
  """``DefaultControl``: actions in [-1, 1] mapped affinely onto each ctrlrange.

  The result is clipped to the range, as musclemimic clips ``ctrl``.
  """
  osl = JointEffortActionCfg(
    entity_name="robot",
    actuator_names=OSL_TORQUE_JOINTS,
    preserve_order=True,
    scale=OSL_MOTOR_CTRL,
    offset=0.0,
    clip={".*": (-OSL_MOTOR_CTRL, OSL_MOTOR_CTRL)},
  )
  if model == "robot":
    ranges = _servo_ranges()
    servos = JointPositionActionCfg(
      entity_name="robot",
      actuator_names=SERVO_JOINTS,
      preserve_order=True,
      scale={j: (hi - lo) / 2.0 for j, (lo, hi) in ranges.items()},
      offset={j: (hi + lo) / 2.0 for j, (lo, hi) in ranges.items()},
      use_default_offset=False,
      clip=dict(ranges),
    )
    # musclemimic's action order: the 27 servos, then the OSL knee and ankle.
    return {"servos": servos, "osl_motors": osl}
  # musclemimic rewrites the muscles' ctrlrange to [-1, 1]; MuJoCo clamps to [0, 1]
  # inside the activation dynamics either way, so negative commands are zero drive.
  muscles = TendonEffortActionCfg(
    entity_name="robot",
    actuator_names=(MUSCLE_TENDONS_EXPR,),
    scale=1.0,
    offset=0.0,
    clip={".*": (-1.0, 1.0)},
  )
  return {"osl_motors": osl, "muscles": muscles}


def _observations(model: Model) -> dict[str, ObservationGroupCfg]:
  motion = {"command_name": "motion"}
  terms: dict[str, ObservationTermCfg] = {
    "root_pos": ObservationTermCfg(func=mdp.root_pos_no_xy, params=motion),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos, params=motion),
    "root_vel": ObservationTermCfg(func=mdp.root_qvel, params=motion),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel, params=motion),
  }
  if model == "msk":
    terms["muscles"] = ObservationTermCfg(
      func=mdp.muscle_state, params={"entity_name": "robot"}
    )
  terms["touch"] = ObservationTermCfg(func=mdp.touch, params={"entity_name": "robot"})
  terms["goal"] = ObservationTermCfg(func=mdp.mimic_goal, params=motion)
  actor = ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=False
  )
  # musclemimic's critic sees exactly the actor's observation.
  return {"actor": actor, "critic": deepcopy(actor)}


def _terminations(mode: Mode) -> dict[str, TerminationTermCfg]:
  motion = {"command_name": "motion"}
  terms = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "trajectory_end": TerminationTermCfg(
      func=mdp.trajectory_end, params=motion, time_out=True
    ),
  }
  if mode == "train":
    terms["site_deviation"] = TerminationTermCfg(
      func=mdp.mean_site_deviation,
      params={**motion, "threshold": TRAIN_THRESHOLDS["site"]},
    )
    terms["root_deviation"] = TerminationTermCfg(
      func=mdp.root_site_deviation,
      params={**motion, "threshold": TRAIN_THRESHOLDS["root"]},
    )
    terms["root_orientation"] = TerminationTermCfg(
      func=mdp.root_orientation_deviation,
      params={**motion, "threshold": TRAIN_THRESHOLDS["root_ori"]},
    )
  else:
    terms["site_deviation"] = TerminationTermCfg(
      func=mdp.mean_site_deviation,
      params={**motion, "threshold": VALIDATION_SITE_THRESHOLD},
    )
  return terms


def stress_env_cfg(
  cfg: ManagerBasedRlEnvCfg, num_steps: int, num_envs: int
) -> ManagerBasedRlEnvCfg:
  """Turn a training-mode config into a stress-validation one, in place, and return
  it: the training distribution (starts, commands and whatever disturbances the task
  trains with), but validation's 0.5 m site check as the only failure, and episodes
  of ``num_steps`` control steps on ``num_envs`` envs."""
  for name in ("root_deviation", "root_orientation"):
    cfg.terminations.pop(name, None)
  site = cfg.terminations["site_deviation"]
  site.params = {**site.params, "threshold": VALIDATION_SITE_THRESHOLD}
  cfg.episode_length_s = num_steps * cfg.decimation * cfg.sim.mujoco.timestep
  cfg.scene.num_envs = num_envs
  return cfg


def osl_mimic_env_cfg(
  model: Model = "robot",
  socket: Socket = "rigid",
  reward: Reward = "mimic",
  mode: Mode = "train",
) -> ManagerBasedRlEnvCfg:
  """Create a musclemimic-style OSL full-body imitation environment.

  Args:
    model: ``"robot"`` (27 position servos + 2 OSL motors) or ``"msk"`` (339
      muscles + 2 OSL motors, fingerless).
    socket: ``"rigid"`` or ``"compliant"``.
    reward: ``"mimic"`` (MimicReward) or ``"clf"`` (CLFReward).
    mode: ``"train"`` starts at random frames with musclemimic's training
      terminations; ``"validation"`` and ``"play"`` start at frame 0 and end only
      on the 0.5 m site check or the clip's end.
  """
  entity = (
    get_osl_robot_cfg(socket, collision="xml")
    if model == "robot"
    else get_osl_msk_cfg(socket, collision="xml")
  )
  clf = reward == "clf"
  rewards = {
    "mimic": RewardTermCfg(
      func=mdp.mimic_reward,
      weight=1.0,
      params={
        "command_name": "motion",
        "weights": CLF_WEIGHTS if clf else MIMIC_WEIGHTS,
        "action_out_of_bounds_coeff": 0.01,
        "joint_torque_weights": OSL_TORQUE_WEIGHTS,
        "clf": mdp.ClfCfg() if clf else None,
      },
    )
  }
  names = DIAGNOSTICS + (CLF_DIAGNOSTICS if clf else ())
  metrics = {
    name: MetricsTermCfg(func=mdp.mimic_component, params={"name": name})
    for name in names
  }

  return ManagerBasedRlEnvCfg(
    decimation=DECIMATION,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": entity},
      num_envs=NUM_ENVS if mode == "train" else 1,
    ),
    observations=_observations(model),
    actions=_actions(model),
    commands={
      "motion": mdp.MimicMotionCommandCfg(random_start=mode == "train"),
    },
    rewards=rewards,
    terminations=_terminations(mode),
    metrics=metrics,
    sim=SimulationCfg(
      # musclemimic's per-world Warp buffers.
      nconmax=96,
      njmax=768,
      mujoco=MujocoCfg(
        timestep=TIMESTEP,
        # Euler on both sockets, as in musclemimic; only Euler's implicit damping
        # follows the socket. The rigid weld is stable with explicit damping (plain
        # MyoFullBody's setting). The compliant socket reaches dof_damping * dt /
        # M_ii = 15.68 against explicit Euler's bound of 2 and turns NaN under Warp
        # unless its damping stays implicit.
        integrator="euler",
        iterations=4,
        ls_iterations=8,
        ccd_iterations=35,  # MuJoCo's default, which musclemimic runs; mjlab's is 50.
        # musclemimic runs MuJoCo 3.4, where multi-contact CCD was opt-in and off.
        # mujoco_warp 3.11 turns it on by default and then rejects the model's
        # 1 mm geom margins on mesh pairs; turning it off restores 3.4's contacts.
        disableflags=("multiccd", "eulerdamp") if socket == "rigid" else ("multiccd",),
        # Not musclemimic's (MuJoCo's default pyramidal cone, impratio 1): mj-nlp's,
        # which solves the gait library.
        cone=CONTACT_CONE,
        impratio=CONTACT_IMPRATIO,
      ),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="pelvis",
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    episode_length_s=EPISODE_LENGTH_S if mode != "play" else 1.0e9,
    scale_rewards_by_dt=False,
  )
