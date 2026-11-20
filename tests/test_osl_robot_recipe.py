"""The robot recipe (round 7) and what it must leave alone (CPU).

* Every OSL config outside the robot unicycle recipe is the digest fixture's, bit
  for bit (``fixtures/osl_cfg_digests.json``, see ``osl_cfg_snapshot.py``; its
  ``source`` names the commit): every musculoskeletal task, and the robot's stairs,
  FSM runner and split-stairs.
* The robot unicycle stage 1 and stage 2 defaults, value by value.
* The robot's networks (2026-11-19): every robot task's actor and critic, stage 2's
  included, at ``ROBOT_HIDDEN_DIMS``; a stage-1 run with musclemimic's networks
  still resumes, starts stage 2 and plays through the flags or its params/.
* The gait library a process's envs share: one copy, never written, serving what a
  fresh load serves.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shlex
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
import tyro
from osl_cfg_snapshot import FIXTURE, KINDS, osl_tasks, task_digests

import mjlab
import mjlab.envs
import mjlab.rl
from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Socket, get_spec
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scripts import play
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_gait_library.config.env_cfgs import (
  UNICYCLE_LIBRARY_ENV_VAR,
  UNICYCLE_ROBOT_NUM_ENVS,
  UNICYCLE_ROBOT_ROOT_DEVIATION,
  UNICYCLE_ROBOT_TWIST_WEIGHT,
  osl_gait_library_env_cfg,
)
from mjlab.tasks.osl_gait_library.config.rl_cfg import (
  UNICYCLE_ROBOT_ENV_STAGES,
  UNICYCLE_ROBOT_ITERATIONS,
  UNICYCLE_ROBOT_PHASE_A_ITERATIONS,
  GaitLibraryRunnerCfg,
  osl_gait_library_runner_cfg,
)
from mjlab.tasks.osl_gait_library.evaluate import EvaluateConfig, evaluate
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_gait_library.mdp import commands as library_commands
from mjlab.tasks.osl_gait_library.mdp.commands import (
  GaitLibraryCommand,
  clear_library_cache,
)
from mjlab.tasks.osl_gait_library.rl import runner as library_runner
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.config.rl_cfg import (
  CLF_ENV_STAGES,
  HIDDEN_DIMS,
  ROBOT_HIDDEN_DIMS,
  MuscleMimicPpoCfg,
  OslMimicRunnerCfg,
)
from mjlab.tasks.osl_mimic.reward_settings import env_stages_in_force
from mjlab.tasks.osl_mimic.rl.runner import (
  UNQUOTED_FLAGS_NOTE,
  configure_networks_for_checkpoint,
  hidden_dims_flags,
  run_hidden_dims,
)
from mjlab.tasks.osl_mimic.touch_sites import CHECKPOINT_KEY as TOUCH_SITES_KEY
from mjlab.tasks.osl_split.config.rl_cfg import (
  OSL_HIDDEN_DIMS,
  SPLIT_CLF_ENV_STAGES,
  SplitRunnerCfg,
  osl_split_runner_cfg,
)
from mjlab.tasks.osl_split.rl.runner import _check_architecture
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml

SOCKETS: tuple[Socket, ...] = ("rigid", "compliant")
REWARDS: tuple[Reward, ...] = ("mimic", "clf")
TAGS: dict[tuple[Socket, Reward], str] = {
  (socket, reward): ("-CLF" if reward == "clf" else "")
  + ("-Compliant" if socket == "compliant" else "")
  for socket in SOCKETS
  for reward in REWARDS
}
STAGE1 = "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-ROBOT"
STAGE2 = "Mjlab-Split-Unicycle-OSL-Fullbody-ROBOT"
FSM = "Mjlab-FSM-Unicycle-OSL-Fullbody-ROBOT"
SPLIT_STAIRS = "Mjlab-Split-Stairs-OSL-Fullbody-ROBOT"
ENV_FIELDS = frozenset({"scene", "rewards", "terminations"})
"""The env fields the robot recipe sets: the env count, the twist weight and the
drift limit."""
RECIPE_CHANGES: dict[str, dict[str, frozenset[str]]] = {
  **{
    STAGE1 + tag: {
      "env": ENV_FIELDS,
      "play_env": frozenset({"rewards"}),
      "rl": frozenset(
        {"algorithm", "env_stages", "max_iterations", "save_interval", "validation"}
      ),
    }
    for tag in TAGS.values()
  },
  **{
    STAGE2 + tag: {
      "env": ENV_FIELDS,
      "play_env": frozenset({"rewards"}),
      "rl": frozenset({"algorithm", "max_iterations", "validation"}),
    }
    for tag in TAGS.values()
  },
  # Stage 1.5's env is the split env (test_osl_fsm_env); its runner is unchanged.
  **{
    FSM + tag: {"env": ENV_FIELDS, "play_env": frozenset({"rewards"})}
    for tag in TAGS.values()
  },
}
"""Config fields the robot recipe may change, per task and config (every other one
must be the fixture's)."""
DT = 0.01


def _fixture() -> dict[str, Any]:
  return json.loads(FIXTURE.read_text())


##
# What the recipe leaves alone.
##


def test_the_digests_cover_every_osl_task() -> None:
  assert sorted(_fixture()["tasks"]) == osl_tasks()


@pytest.mark.parametrize("task", osl_tasks())
def test_configs_outside_the_recipe_are_the_reward_commits(task: str) -> None:
  """Every musculoskeletal task, and every robot config the recipe does not set,
  equals the fixture's field for field; the recipe's own changes stay in their
  fields."""
  want = _fixture()["tasks"][task]
  got = task_digests(task)
  allowed = RECIPE_CHANGES.get(task, {})
  for kind in KINDS:
    fields = set(want[kind]) | set(got[kind])
    changed = sorted(f for f in fields if want[kind].get(f) != got[kind].get(f))
    unexpected = sorted(set(changed) - allowed.get(kind, frozenset()))
    assert not unexpected, (
      f"{task} {kind}: {unexpected} differ from the fixture's configs. If the "
      "base commit changed them, regenerate the digests from it (osl_cfg_snapshot)."
    )
    if "-MSK" in task:
      assert not changed


@pytest.mark.parametrize("tag", list(TAGS.values()))
def test_fsm_and_split_stairs_keep_the_robots_former_runner(tag: str) -> None:
  """Stage 1.5 and stairs stage 2 do not build on the robot's unicycle recipe: they
  keep the robot's settings from before it, float32 Newton-Schulz included (it
  would otherwise reach them through the inherited PPO fields)."""
  for task in (FSM + tag, SPLIT_STAIRS + tag):
    rl = load_rl_cfg(task)
    assert isinstance(rl, SplitRunnerCfg)
    assert rl.algorithm.muon_ns_dtype == "float32"
    assert rl.algorithm.schedule_iterations == 0
    assert rl.validation.interval == 30
  assert load_rl_cfg(SPLIT_STAIRS + tag).max_iterations == 1000
  # The flag itself: the unicycle split config without the recipe is the former one.
  socket, reward = next(k for k, v in TAGS.items() if v == tag)
  former = osl_split_runner_cfg("robot", socket, reward, "x", robot_recipe=False)
  assert former.algorithm.muon_ns_dtype == "float32"
  assert former.max_iterations == 1000 and former.validation.interval == 30
  msk = osl_split_runner_cfg("msk", socket, reward, "x")
  assert osl_split_runner_cfg("msk", socket, reward, "x", robot_recipe=False) == msk


##
# The recipe.
##


@pytest.mark.parametrize("socket", SOCKETS)
@pytest.mark.parametrize("reward", REWARDS)
def test_robot_stage1_recipe(socket: Socket, reward: Reward) -> None:
  task = STAGE1 + TAGS[(socket, reward)]
  env = load_env_cfg(task)
  # 8192 envs in training only; the twist weight in every mode (validation scores
  # with it); the 2 m drift limit where there is one (training).
  assert env.scene.num_envs == UNICYCLE_ROBOT_NUM_ENVS == 8192
  assert env.rewards["twist"].weight == UNICYCLE_ROBOT_TWIST_WEIGHT == 0.2
  root = env.terminations["root_deviation"].params["threshold"]
  assert root == UNICYCLE_ROBOT_ROOT_DEVIATION == 2.0
  for mode in ("validation", "play"):
    other = osl_gait_library_env_cfg("robot", socket, reward, mode, "unicycle")
    assert other.scene.num_envs == 1
    assert other.rewards["twist"].weight == 0.2
    assert "root_deviation" not in other.terminations
  motion = env.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg) and motion.sampling == "library"

  rl = load_rl_cfg(task)
  assert isinstance(rl, GaitLibraryRunnerCfg)
  alg = rl.algorithm
  assert isinstance(alg, MuscleMimicPpoCfg) and rl.actor.distribution_cfg is not None
  assert rl.max_iterations == UNICYCLE_ROBOT_ITERATIONS == 2000
  assert (alg.learning_rate, alg.lr_schedule) == (4e-4, "linear")
  assert (alg.lr_schedule_updates, alg.lr_final_ratio) == (1500, 0.1)
  assert alg.lr_schedule_offset == 0
  assert rl.actor.distribution_cfg["init_std"] == 0.2 and alg.init_std_terms == {}
  assert (alg.std_cap_start, alg.std_cap_end) == (0.2, 0.07)
  assert (alg.std_cap_begin_update, alg.std_cap_end_update) == (200, 1500)
  # Phase A (the chain's first launch) ends where the rate and the cap reach their
  # ends; phase B trains on at them.
  assert UNICYCLE_ROBOT_PHASE_A_ITERATIONS == 1500 == alg.lr_schedule_updates
  assert alg.std_floor == 0.03 <= alg.std_cap_end
  assert alg.actor_bias_init == "qpos0" and alg.muon_ns_dtype == "bfloat16"
  assert (alg.num_mini_batches, rl.num_steps_per_env) == (128, 80)
  assert rl.actor.hidden_dims == rl.critic.hidden_dims == ROBOT_HIDDEN_DIMS
  v = rl.validation
  assert (v.interval, v.inference) == (125, ("mean",))
  assert (v.stress_interval, v.stress_inference) == (250, ("mean",))
  assert rl.save_interval == 250 and rl.video.enabled

  # The recipe's stages come after the CLF ramps (none under MimicReward).
  ramps = CLF_ENV_STAGES if reward == "clf" else ()
  assert rl.env_stages == (*ramps, *UNICYCLE_ROBOT_ENV_STAGES)
  assert UNICYCLE_ROBOT_ENV_STAGES == (
    (400, "terminations.root_deviation.threshold", 1.0),
    (1200, "rewards.twist.weight", 0.06),
  )
  # No two settings change at one update.
  changes = [u for u, _, _ in rl.env_stages]
  owners = {u: {p for uu, p, _ in rl.env_stages if uu == u} for u in changes}
  if reward == "clf":
    assert owners == {
      500: {
        "rewards.mimic.params.clf.pen_scale",
        "rewards.mimic.params.regularizers.scale",
      },
      1000: {
        "rewards.mimic.params.clf.pen_scale",
        "rewards.mimic.params.regularizers.scale",
      },
      400: {"terminations.root_deviation.threshold"},
      1200: {"rewards.twist.weight"},
    }
  # What each update trains with (the config's own value once no stage holds).
  force = {u: env_stages_in_force(rl.env_stages, u) for u in (0, 399, 400, 1199, 1200)}
  assert force[0]["terminations.root_deviation.threshold"] == 1.0
  assert force[0]["rewards.twist.weight"] == 0.06
  assert "terminations.root_deviation.threshold" not in force[400]
  assert force[1199]["rewards.twist.weight"] == 0.06
  assert "rewards.twist.weight" not in force[1200]


@pytest.mark.parametrize("socket", SOCKETS)
@pytest.mark.parametrize("reward", REWARDS)
def test_robot_stage2_recipe(socket: Socket, reward: Reward) -> None:
  """Round 6's tested stage-2 design on the robot: 500 iterations at schedule 600,
  validation every 60, MSK's curriculum and ramps; the env count and bfloat16
  Newton-Schulz come from the robot's stage-1 recipe; sampling stays the env's (the
  launchers pass stage 1's last phase's flags)."""
  task = STAGE2 + TAGS[(socket, reward)]
  env = load_env_cfg(task)
  assert env.scene.num_envs == 8192
  assert env.rewards["twist"].weight == 0.2
  assert env.terminations["root_deviation"].params["threshold"] == 2.0
  motion = env.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg) and motion.sampling == "library"

  rl = load_rl_cfg(task)
  assert isinstance(rl, SplitRunnerCfg)
  assert (rl.max_iterations, rl.algorithm.schedule_iterations) == (500, 600)
  assert (rl.validation.interval, rl.validation.stress_interval) == (60, 120)
  assert rl.save_interval == 250
  stage1 = osl_gait_library_runner_cfg("robot", socket, reward, "x", "unicycle")
  assert isinstance(stage1.algorithm, MuscleMimicPpoCfg)
  assert rl.algorithm.muon_ns_dtype == stage1.algorithm.muon_ns_dtype == "bfloat16"
  assert rl.algorithm.num_mini_batches == 128
  assert rl.env_stages == (SPLIT_CLF_ENV_STAGES if reward == "clf" else ())
  # The curriculum is MSK's, unchanged.
  msk = osl_split_runner_cfg("msk", socket, reward, "x")
  assert (rl.freeze, rl.blend, rl.channels) == (msk.freeze, msk.blend, msk.channels)
  assert rl.osl_actor == msk.osl_actor
  assert replace(rl.algorithm, schedule_iterations=0) == msk.algorithm


##
# The robot's networks.
##

MUSCLEMIMIC_MODEL = "mjlab.tasks.osl_mimic.rl.models:MuscleMimicMLPModel"
STAIRS = "Mjlab-Stairs-OSL-Fullbody-ROBOT"
OLD_DIMS = "'(" + ",".join(str(dim) for dim in HIDDEN_DIMS) + ")'"
"""musclemimic's dims as the refusals print them: no spaces, quoted for the shell."""


@pytest.mark.parametrize("task", osl_tasks())
def test_robot_tasks_train_the_small_networks(task: str) -> None:
  """Every robot task on musclemimic's networks trains its actor and critic at
  ``ROBOT_HIDDEN_DIMS``, stage 2's and stage 1.5's OSL critic too, the OSL actor
  unchanged; every musculoskeletal task keeps musclemimic's 16 layers. The G1-style
  tracking tasks build rsl_rl's MLP, not these networks."""
  assert ROBOT_HIDDEN_DIMS == (1024, 1024, 512, 512)
  assert HIDDEN_DIMS == (1024,) * 12 + (2048,) * 3 + (1024,)
  rl = load_rl_cfg(task)
  assert isinstance(rl, RslRlOnPolicyRunnerCfg)
  if rl.actor.class_name != MUSCLEMIMIC_MODEL:
    assert "-Tracking-" in task
    return
  want = ROBOT_HIDDEN_DIMS if "-ROBOT" in task else HIDDEN_DIMS
  networks = [rl.actor, rl.critic]
  if isinstance(rl, SplitRunnerCfg):
    networks.append(rl.osl_critic)
    assert rl.osl_actor.hidden_dims == OSL_HIDDEN_DIMS
  assert [tuple(n.hidden_dims) for n in networks] == [want] * len(networks)


def _train_cfg(task: str, *argv: str) -> TrainConfig:
  """The config ``train.py`` builds: the task's, with ``argv`` applied by tyro."""
  return tyro.cli(
    TrainConfig,
    args=list(argv),
    default=TrainConfig.from_task(task),
    config=mjlab.TYRO_FLAGS,
  )


def _lineage(stage1: Any, hidden_dims: tuple[int, ...]) -> dict[str, Any]:
  """What a warm start records of a stage-1 run with these networks: its
  agent.yaml's model blocks, as plain data."""
  blocks = {
    key: dict(asdict(getattr(stage1, name)), hidden_dims=list(hidden_dims))
    for key, name in (("actor_cfg", "actor"), ("critic_cfg", "critic"))
  }
  return {"stage1_checkpoint": "/runs/stage1/model_1000.pt", **blocks}


@pytest.mark.parametrize(
  ("stage2", "stage1"), [(STAGE2, STAGE1), (FSM, STAGE1), (SPLIT_STAIRS, STAIRS)]
)
def test_robot_stage2_builds_its_teachers_networks(stage2: str, stage1: str) -> None:
  """Stage 2 (and stage 1.5, stairs stage 2) builds the networks it copies from
  stage 1 at today's robot stage-1 size: a teacher of that size passes the
  architecture check with no flag. A teacher with musclemimic's networks (every
  robot run before 2026-11-19) is refused, naming the flags that build them, and
  those flags, parsed as ``train`` parses them (pasted into a shell, or unquoted
  inside R7_EXTRA), pass."""
  tag = "-CLF-Compliant"
  teacher = load_rl_cfg(stage1 + tag)
  assert isinstance(teacher, OslMimicRunnerCfg)
  assert teacher.actor.hidden_dims == teacher.critic.hidden_dims == ROBOT_HIDDEN_DIMS
  task = stage2 + tag
  rl = asdict(load_rl_cfg(task))
  _check_architecture(_lineage(teacher, ROBOT_HIDDEN_DIMS), rl, reward_switch=False)
  old = _lineage(teacher, HIDDEN_DIMS)
  with pytest.raises(ValueError, match="another architecture") as refused:
    _check_architecture(old, rl, reward_switch=False)
  message = str(refused.value)
  flags = message.split("Build stage 1's architecture: ", 1)[1].splitlines()[0]
  for module in ("actor", "critic", "osl-critic"):
    assert f"--agent.{module}.hidden-dims {OLD_DIMS}" in flags
  assert UNQUOTED_FLAGS_NOTE in message
  for argv in (shlex.split(flags), flags.replace("'", "").split()):
    parsed = cast(Any, _train_cfg(task, *argv).agent)
    assert parsed.osl_actor.hidden_dims == OSL_HIDDEN_DIMS
    _check_architecture(old, asdict(parsed), reward_switch=False)


def _stage1(task: str) -> GaitLibraryRunnerCfg:
  cfg = load_rl_cfg(task)
  assert isinstance(cfg, GaitLibraryRunnerCfg)
  return cfg


def test_an_old_robot_run_is_rebuilt_from_its_params(tmp_path: Path) -> None:
  """A robot run from before 2026-11-19 records musclemimic's networks in its
  params/agent.yaml. The resume error's flags (quoted for the shell, with no
  spaces: the round-7 launchers split R7_EXTRA on whitespace and take them
  unquoted) parse back to them either way, and play's and the evaluation's hook
  sizes today's config from that file; a run without params/ changes nothing."""
  task = STAGE1 + "-CLF-Compliant"
  old = _stage1(task)
  old.actor.hidden_dims = old.critic.hidden_dims = HIDDEN_DIMS
  run = tmp_path / "2026-11-19_11-00-43_r7_robot_s1_seed42_A"
  dump_yaml(run / "params" / "agent.yaml", asdict(old))  # As train.py writes it.
  checkpoint = run / "model_1000.pt"
  assert run_hidden_dims(checkpoint) == {"actor": HIDDEN_DIMS, "critic": HIDDEN_DIMS}

  today = _stage1(task)
  flags = hidden_dims_flags(checkpoint, asdict(today))
  assert flags == [
    f"--agent.actor.hidden-dims {OLD_DIMS}",
    f"--agent.critic.hidden-dims {OLD_DIMS}",
  ]
  text = " ".join(flags)
  for argv in (shlex.split(text), text.replace("'", "").split()):
    parsed = cast(Any, _train_cfg(task, *argv).agent)
    assert parsed.actor.hidden_dims == parsed.critic.hidden_dims == HIDDEN_DIMS
    assert hidden_dims_flags(checkpoint, asdict(parsed)) == []

  changed = configure_networks_for_checkpoint(today, checkpoint)
  assert changed == {"actor": HIDDEN_DIMS, "critic": HIDDEN_DIMS}
  assert today.actor.hidden_dims == today.critic.hidden_dims == HIDDEN_DIMS
  bare = _stage1(task)
  assert configure_networks_for_checkpoint(bare, tmp_path / "model_0.pt") == {}
  assert bare.actor.hidden_dims == ROBOT_HIDDEN_DIMS


class _Built(Exception):
  """Raised where the tool would build its env or runner: what it configured
  before is checked then."""


def _built(*args: Any, **kwargs: Any) -> Any:
  raise _Built


def _old_robot_run(tmp_path: Path, task: str) -> Path:
  """A checkpoint of a robot run from before 2026-11-19: musclemimic's networks in
  its params/agent.yaml, the task's env in its params/env.yaml, and nothing in
  the checkpoint but the touch sites (neither tool here gets to load it)."""
  run = tmp_path / "2026-11-19_11-00-43_r7_robot_s1_seed42_A"
  old = _stage1(task)
  old.actor.hidden_dims = old.critic.hidden_dims = HIDDEN_DIMS
  env = load_env_cfg(task)
  dump_yaml(run / "params" / "agent.yaml", asdict(old))
  dump_yaml(run / "params" / "env.yaml", asdict(env))
  checkpoint = run / "model_1000.pt"
  payload = {"actor_state_dict": {}, "iter": 1000, "infos": None}
  torch.save({**payload, TOUCH_SITES_KEY: cast(Any, env).osl_touch_sites}, checkpoint)
  return checkpoint


def test_play_builds_the_networks_of_the_checkpoints_run(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """play.py takes no --agent overrides: before it builds anything, the task's
  runner sizes the networks from the checkpoint's run
  (``configure_agent_for_checkpoint``)."""
  task = STAGE1 + "-CLF-Compliant"
  checkpoint = _old_robot_run(tmp_path, task)
  loaded: list[Any] = []

  def load(task_id: str) -> Any:
    loaded.append(load_rl_cfg(task_id))
    return loaded[-1]

  monkeypatch.setattr(play, "load_rl_cfg", load)
  monkeypatch.setattr(play, "ManagerBasedRlEnv", _built)
  cfg = play.PlayConfig(checkpoint_file=str(checkpoint), device="cpu")
  with pytest.raises(_Built):
    play.run_play(task, cfg)
  (agent,) = loaded
  assert agent.actor.hidden_dims == agent.critic.hidden_dims == HIDDEN_DIMS


def test_evaluation_builds_the_networks_of_the_checkpoints_run(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The gait-library evaluation hands its runner the networks the checkpoint's
  run trained, read from its params/agent.yaml."""
  task = STAGE1 + "-CLF-Compliant"
  checkpoint = _old_robot_run(tmp_path, task)
  monkeypatch.setenv(UNICYCLE_LIBRARY_ENV_VAR, "")  # The evaluation sets it.
  agents: list[dict[str, Any]] = []

  def runner(env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
    agents.append(train_cfg)
    raise _Built

  monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", lambda **kwargs: None)
  monkeypatch.setattr(mjlab.rl, "RslRlVecEnvWrapper", lambda env: env)
  monkeypatch.setattr(library_runner, "GaitLibraryOnPolicyRunner", runner)
  with pytest.raises(_Built):
    evaluate(EvaluateConfig(checkpoint, library=tmp_path, device="cpu"))
  (agent,) = agents
  assert agent["actor"]["hidden_dims"] == agent["critic"]["hidden_dims"] == HIDDEN_DIMS


def test_evaluation_gives_its_envs_the_runs_sampler(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The gait-library evaluation builds its training env (which the stress pass
  copies, and whose command the validation env follows) with the twist sampler of
  the checkpoint's run, read from its params/env.yaml: a run on a forward-only
  library, its pivot and backward fractions at 0, is evaluated as it trained."""
  task = STAGE1 + "-CLF-Compliant"
  checkpoint = _old_robot_run(tmp_path, task)
  saved = load_env_cfg(task)
  motion = cast(GaitLibraryCommandCfg, saved.commands["motion"])
  motion.sampling = "continuous"
  motion.rel_turn_envs = motion.rel_back_envs = 0.0
  motion.fwd_arc_regions = (((0.5, 1.5), (0.0, 0.5), 1.0),)
  motion.twist_metric_weights, motion.blend_time_s = (1.0, 1.0, 0.25), 0.6
  dump_yaml(checkpoint.parent / "params" / "env.yaml", asdict(saved))
  monkeypatch.setenv(UNICYCLE_LIBRARY_ENV_VAR, "")  # The evaluation sets it.
  built: list[Any] = []

  def env(cfg: Any, **kwargs: Any) -> Any:
    built.append(cfg)
    raise _Built

  monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", env)
  with pytest.raises(_Built):
    evaluate(EvaluateConfig(checkpoint, library=tmp_path, device="cpu"))
  (cfg,) = built
  rebuilt = cfg.commands["motion"]
  assert isinstance(rebuilt, GaitLibraryCommandCfg)
  assert rebuilt.library_dir == str(tmp_path)
  assert (rebuilt.sampling, rebuilt.rel_turn_envs, rebuilt.rel_back_envs) == (
    "continuous",
    0.0,
    0.0,
  )
  assert rebuilt.fwd_arc_regions == (((0.5, 1.5), (0.0, 0.5), 1.0),)
  assert rebuilt.bck_arc_regions == motion.bck_arc_regions  # Unchanged by the run.
  # The snap metric and the blend time of the run's clip switches.
  assert rebuilt.twist_metric_weights == (1.0, 1.0, 0.25)
  assert rebuilt.blend_time_s == 0.6


##
# The shared library.
##


def _clip(model: Any, data: Any, frames: int, vel: tuple, wz: float) -> dict:
  """``frames`` of the start pose moving at world velocity ``vel`` and yaw rate
  ``wz``, as clipper's mjlab format stores them (tests/test_osl_split_export.py)."""
  base = model.qpos0.copy()
  pos, quat, joints = [], [], []
  for t in range(frames):
    q = base.copy()
    q[:2] = np.asarray(vel[:2]) * t * DT
    yaw = wz * t * DT
    q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    pos.append(data.xpos[1:].copy())
    quat.append(data.xquat[1:].copy())
    joints.append(q[7:].copy())
  body_pos = np.stack(pos)
  ang = np.zeros_like(body_pos)
  ang[..., 2] = wz
  return {
    "joint_pos": np.stack(joints),
    "joint_vel": np.zeros((frames, model.nq - 7)),
    "body_pos_w": body_pos,
    "body_quat_w": np.stack(quat),
    "body_lin_vel_w": np.gradient(body_pos, DT, axis=0),
    "body_ang_vel_w": ang,
    "fps": np.array([1.0 / DT]),
  }


def _se2(x: float, y: float, yaw: float) -> np.ndarray:
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def robot_model() -> Any:
  return get_spec("robot", "rigid").compile()


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory, robot_model: Any) -> Path:
  """Idle, a forward and a backward walk and a pivot: every unicycle mode."""
  folder = tmp_path_factory.mktemp("library")
  data = mujoco.MjData(robot_model)
  frames, v, wz = 50, 0.8, 0.6
  # Forward is the root's body -y, which is world -y at zero yaw.
  clips = {
    "idle": ((0, 0, 0), (0.0, 0.0, 0.0), 0.0, _se2(0, 0, 0)),
    "walk_fwd": ((0, -v, 0), (v, 0.0, 0.0), 0.0, _se2(0, -v * frames * DT, 0)),
    "walk_bck": ((0, v, 0), (-v, 0.0, 0.0), 0.0, _se2(0, v * frames * DT, 0)),
    "walk_turn": ((0, 0, 0), (0.0, 0.0, wz), wz, _se2(0, 0, wz * frames * DT)),
  }
  for name, (vel, twist, rate, stride) in clips.items():
    clip = _clip(robot_model, data, frames, vel, rate)
    np.savez(folder / f"{name}.npz", twist=np.array(twist), stride_se2=stride, **clip)
  return folder


@pytest.fixture
def fresh_cache() -> Iterator[None]:
  clear_library_cache()
  yield
  clear_library_cache()


def _command(
  library: Path, model: Any, site_bodies: tuple[str, ...] | None = None
) -> GaitLibraryCommand:
  """A real command over ``library`` on a stand-in env and entity (the robot's body
  and joint names); only the per-step update needs a simulated robot."""
  bodies = [model.body(i).name for i in range(1, model.nbody)]
  joints = [model.joint(j).name for j in range(1, model.njnt)]
  robot = SimpleNamespace(body_names=bodies, joint_names=joints)
  env = SimpleNamespace(num_envs=4, device="cpu", step_dt=DT, scene={"robot": robot})
  cfg = GaitLibraryCommandCfg(library_dir=str(library), resampling_time_range=(3, 8))
  cfg.sampling = "library"
  if site_bodies is not None:
    cfg.site_bodies = site_bodies
  return GaitLibraryCommand(cfg, cast(Any, env))


def _library_tensors(cmd: GaitLibraryCommand) -> dict[str, torch.Tensor]:
  return {
    **{f"data.{k}": v for k, v in cmd.data.items()},
    "num_frames": cmd.num_frames,
    "twists": cmd.twists,
    "strides": cmd.strides,
  }


def _checksums(cmd: GaitLibraryCommand) -> dict[str, str]:
  return {
    k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    for k, v in _library_tensors(cmd).items()
  }


def test_commands_over_one_library_share_one_copy(
  library: Path, robot_model: Any, fresh_cache: None
) -> None:
  a = _command(library, robot_model)
  b = _command(library, robot_model)
  # One copy of every tensor, under a dict and a name list of each command's own.
  for key, tensor in _library_tensors(a).items():
    assert tensor.data_ptr() == _library_tensors(b)[key].data_ptr(), key
  assert a.data is not b.data and a.clip_names is not b.clip_names
  assert a.clip_names == b.clip_names == ["idle", "walk_bck", "walk_fwd", "walk_turn"]
  assert a.num_clips == b.num_clips == 4
  b.data["mine"] = torch.zeros(1)  # A command's own entry stays its own.
  assert "mine" not in a.data
  # Other reference sites load another copy.
  sites = a.cfg.site_bodies[:3]
  c = _command(library, robot_model, site_bodies=sites)
  assert c.data["site_pos"].shape[2] == 3
  assert c.data["joint_pos"].data_ptr() != a.data["joint_pos"].data_ptr()
  # Once no command holds a library, it is freed.
  assert len(library_commands._LIBRARY_CACHE) == 2
  del a, b, c
  gc.collect()
  assert len(library_commands._LIBRARY_CACHE) == 0


def test_a_rewritten_library_is_read_again(
  tmp_path: Path, library: Path, robot_model: Any, fresh_cache: None
) -> None:
  folder = tmp_path / "lib"
  folder.mkdir()
  for clip in library.glob("*.npz"):
    (folder / clip.name).write_bytes(clip.read_bytes())
  before = _command(folder, robot_model)
  with np.load(folder / "walk_fwd.npz") as data:
    arrays = dict(data)
  arrays["joint_pos"] = arrays["joint_pos"] + 0.01
  np.savez(folder / "walk_fwd.npz", **arrays)
  stat = (folder / "walk_fwd.npz").stat()
  os.utime(folder / "walk_fwd.npz", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
  after = _command(folder, robot_model)
  assert after.data["joint_pos"].data_ptr() != before.data["joint_pos"].data_ptr()
  clip = after.clip_names.index("walk_fwd")
  shift = after.data["joint_pos"][clip] - before.data["joint_pos"][clip]
  torch.testing.assert_close(shift, torch.full_like(shift, 0.01), rtol=0, atol=1e-6)


def test_a_shared_library_serves_what_a_fresh_load_serves(
  library: Path, robot_model: Any, fresh_cache: None
) -> None:
  """A command on the shared copy serves, bit for bit, what a command that loaded
  the library itself serves, and serving never writes into the shared tensors."""
  shared = _command(library, robot_model)
  other = _command(library, robot_model)  # Holds the shared copy too.
  clear_library_cache()
  fresh = _command(library, robot_model)
  assert fresh.data["joint_pos"].data_ptr() != shared.data["joint_pos"].data_ptr()
  sums = _checksums(shared)
  assert sums == _checksums(fresh)
  g = torch.Generator().manual_seed(0)
  for _ in range(50):
    clip = torch.randint(0, 4, (4,), generator=g)
    frame = torch.randint(0, 400, (4,), generator=g)
    anchor = torch.rand(4, 3, generator=g) * 4 - 2
    z = torch.rand(4, generator=g) * 0.1
    got, want = (
      shared._gather(clip, frame, anchor, z),
      fresh._gather(clip, frame, anchor, z),
    )
    assert got.keys() == want.keys()
    for key in want:
      assert torch.equal(got[key], want[key]), key
  assert _checksums(shared) == _checksums(other) == sums


@pytest.mark.slow
def test_rollouts_on_the_shared_library_are_bit_identical(
  library: Path, fresh_cache: None
) -> None:
  """The real robot CLF env, three times: the first load, an env sharing its copy
  (as the validation, stress and video envs do), and an env that loads its own;
  identical seeds and actions give identical rewards, and no rollout writes into
  the shared library."""

  def make_env() -> ManagerBasedRlEnv:
    cfg = osl_gait_library_env_cfg("robot", "compliant", "clf", "train", "unicycle")
    motion = cfg.commands["motion"]
    assert isinstance(motion, GaitLibraryCommandCfg)
    motion.library_dir = str(library)
    cfg.scene.num_envs = 4
    cfg.seed = 0
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")

  def rollout(env: ManagerBasedRlEnv) -> torch.Tensor:
    # The env draws commands, starts and pushes from torch's global generator.
    torch.manual_seed(1)
    env.reset()
    g = torch.Generator().manual_seed(1)
    rewards = []
    for _ in range(12):
      action = torch.rand(4, env.action_manager.total_action_dim, generator=g) * 2 - 1
      rewards.append(env.step(action)[1].clone())
    return torch.stack(rewards)

  def command(env: ManagerBasedRlEnv) -> GaitLibraryCommand:
    return cast(GaitLibraryCommand, env.command_manager.get_term("motion"))

  first = make_env()
  sums = _checksums(command(first))
  shared = make_env()
  assert (
    command(shared).data["joint_pos"].data_ptr()
    == command(first).data["joint_pos"].data_ptr()
  )
  reference = rollout(first)
  assert torch.equal(rollout(shared), reference)
  clear_library_cache()
  own = make_env()
  assert (
    command(own).data["joint_pos"].data_ptr()
    != command(first).data["joint_pos"].data_ptr()
  )
  assert torch.equal(rollout(own), reference)
  assert reference.abs().sum() > 0
  assert _checksums(command(first)) == _checksums(command(own)) == sums
  for env in (first, shared, own):
    env.close()
