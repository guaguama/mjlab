"""Contracts for stairs stage 2 (``tasks/osl_split_stairs``).

CPU only and without an mjlab env:

* the registration, and the agent settings against unicycle stage 2's;
* the env config against stairs stage 1's and unicycle stage 2's: the human's
  observation is stairs stage 1's, the OSL actor's is unicycle stage 2's for any
  sensor set, the terrain follows the command line;
* the runner's evaluation configs and one-clip validation, the terrain copy, the
  clip and terrain in checkpoints, and the warm start from a synthetic stairs
  stage-1 run with its provenance checks.

The clip test builds the env on a clipper stairs clip (``MJLAB_OSL_STAIRS_MOTION``).
A slow test also trains two iterations from a stairs stage-1 run
(``MJLAB_OSL_SPLIT_STAIRS_STAGE1_RUN``) on the test device.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import asdict, replace
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import tyro
from conftest import get_test_device
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensordict import TensorDict
from test_osl_split_runner import (
  ACTOR_DIM,
  HIDDEN,
  HUMAN_DIM,
  _FakeEnv,
  _train_cfg,
)
from test_osl_stairs import BOXES, MOTION, _write_clip, needs_clip

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_gait_library.config.dr import OslDrCfg
from mjlab.tasks.osl_mimic.config.env_cfgs import osl_mimic_env_cfg
from mjlab.tasks.osl_mimic.config.rl_cfg import MuscleMimicPpoCfg
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_mimic.rl.muon import MuonAdam
from mjlab.tasks.osl_mimic.touch_sites import CHECKPOINT_KEY, set_touch_sites
from mjlab.tasks.osl_split.config.env_cfgs import (
  PERCEPTION_DR_REFUSAL,
  osl_split_env_cfg,
)
from mjlab.tasks.osl_split.config.rl_cfg import SplitRunnerCfg, osl_split_runner_cfg
from mjlab.tasks.osl_split.mdp import (
  OslObsCfg,
  SplitRewardCfg,
  osl_obs_dim,
  osl_validation_group,
  osl_validation_lags,
  split_mimic_reward,
)
from mjlab.tasks.osl_split.rl.layout import ActionLayout
from mjlab.tasks.osl_split.rl.runner import SplitOnPolicyRunner
from mjlab.tasks.osl_split.rl.storage import obs_group
from mjlab.tasks.osl_split.rl.warmstart import (
  LOG_STD,
  build_warm_start,
  lineage_touch_sites,
  load_warm_start,
)
from mjlab.tasks.osl_split_stairs.config import (
  STAIRS_SPLIT_TASKS,
  TASK_PREFIX,
  stairs_split_experiment_name,
  stairs_split_task_id,
)
from mjlab.tasks.osl_split_stairs.config.env_cfgs import (
  OslSplitStairsEnvCfg,
  osl_split_stairs_env_cfg,
)
from mjlab.tasks.osl_split_stairs.config.rl_cfg import VIDEO_STEPS
from mjlab.tasks.osl_split_stairs.rl.runner import (
  STAIRS_INFO_KEY,
  StairsSplitRunner,
  check_stairs_warm_start,
  stairs_identity,
)
from mjlab.tasks.osl_stairs.config.env_cfgs import osl_stairs_env_cfg, sync_terrain
from mjlab.tasks.osl_stairs.config.rl_cfg import osl_stairs_runner_cfg
from mjlab.tasks.osl_stairs.mdp import StairsMotionCommandCfg
from mjlab.tasks.osl_stairs.terrain import TerrainBoxes, resolve_terrain_file
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml

TASK = "Mjlab-Split-Stairs-OSL-Fullbody-MSK-CLF-Compliant"
UNICYCLE_TASK = "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
MODES = ("train", "validation", "play")
OSL_FLAG = "--env.observations.osl"
SENSOR_SETS = (
  OslObsCfg(),
  OslObsCfg(socket_load_cell=True, heel_contact=False, toe_contact=False),
  OslObsCfg(heel_contact=False, toe_contact=False),
  OslObsCfg(history_length=4, socket_load_cell=True),
)
"""The default (the real leg's), R2's two cells, contacts off, and two cells plus
contacts over a shorter history."""
STAGE1_EXPERIMENT = "osl_fullbody_stairs_msk_clf_compliant"
STAGE1_ITER = 40
STAGE1_RUN_ENV_VAR = "MJLAB_OSL_SPLIT_STAIRS_STAGE1_RUN"


def _training_cfg(folder: Path, **kwargs: Any) -> OslSplitStairsEnvCfg:
  """The MSK-CLF-compliant training config on a synthetic clip in ``folder``."""
  cfg = osl_split_stairs_env_cfg("msk", "compliant", "clf", "train", **kwargs)
  motion = cfg.commands["motion"]
  assert isinstance(motion, StairsMotionCommandCfg)
  motion.motion_file = str(_write_clip(folder))
  sync_terrain(cfg)
  return cfg


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


##
# Registration and settings.
##


def test_tasks_are_registered_with_unicycle_stage2_settings() -> None:
  """Eight tasks, the stairs runner, and unicycle stage 2's agent settings except
  stairs stage 1's episode handling and validation, the variant, the videos and
  the tags."""
  ids = sorted(t for t in list_tasks() if t.startswith(TASK_PREFIX))
  tags = [
    m + r + s
    for m in ("-ROBOT", "-MSK")
    for r in ("", "-CLF")
    for s in ("", "-Compliant")
  ]
  assert ids == sorted(STAIRS_SPLIT_TASKS) == sorted(TASK_PREFIX + t for t in tags)
  for task, (model, socket, reward) in STAIRS_SPLIT_TASKS.items():
    assert stairs_split_task_id(model, socket, reward) == task
    assert load_runner_cls(task) is StairsSplitRunner
    suffix = task.removeprefix(TASK_PREFIX).removeprefix("-ROBOT")
    experiment = "osl_fullbody_split_stairs" + suffix.lower().replace("-", "_")
    assert stairs_split_experiment_name(model, socket, reward) == experiment

    # Stairs stage 1 is not the robot's unicycle recipe: the robot keeps its former
    # stage-2 settings (float32 Newton-Schulz, 1000 iterations).
    expected = osl_split_runner_cfg(
      model, socket, reward, experiment, robot_recipe=False
    )
    assert expected.algorithm.muon_ns_dtype == (
      "float32" if model == "robot" else "bfloat16"
    )
    # Unicycle stage 2 bootstraps; stairs stage 1, whose critics stage 2 copies,
    # does not.
    assert expected.algorithm.bootstrap_time_outs
    stage1 = osl_stairs_runner_cfg(model, socket, reward, experiment)
    assert isinstance(stage1.algorithm, MuscleMimicPpoCfg)
    assert not stage1.algorithm.bootstrap_time_outs
    assert (stage1.validation.num_envs, stage1.validation.num_steps) == (100, 0)
    expected.algorithm.bootstrap_time_outs = False
    expected.validation.num_envs = 100
    expected.validation.num_steps = 0
    expected.validation.variant = "stairs"
    expected.video.num_steps = VIDEO_STEPS
    expected.wandb_tags = ("osl_fullbody", model, reward, "stairs", "split")
    assert load_rl_cfg(task) == expected
    for play in (False, True):
      assert isinstance(load_env_cfg(task, play=play), OslSplitStairsEnvCfg)


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_env_is_stairs_stage1_through_the_split_stage(model: str) -> None:
  """Per mode: stairs stage 1's scene terrain, events, terminations, episode and
  human observation; unicycle stage 2's OSL group, OSL sensors, leg-contact
  sensor, split reward and metrics; no critic group. The musculoskeletal CLF
  penalty weight is unicycle stage 2's 0, the robot's stays osl_mimic's."""
  m = cast(Any, model)
  for mode in MODES:
    md = cast(Any, mode)
    cfg = osl_split_stairs_env_cfg(m, "compliant", "clf", md)
    stage1 = osl_stairs_env_cfg(m, "compliant", "clf", md)
    unicycle = osl_split_env_cfg(m, "compliant", "clf", md)
    assert set(cfg.observations) == {"actor", "osl"}
    assert cfg.observations["actor"] == stage1.observations["actor"]
    assert cfg.observations["osl"] == unicycle.observations["osl"]

    assert isinstance(cfg.scene.spec_fn, TerrainBoxes)
    assert cfg.scene.spec_fn == stage1.scene.spec_fn
    assert cfg.scene.env_spacing == 0.0
    robot = cfg.scene.entities["robot"]
    uni_robot = unicycle.scene.entities["robot"]
    qualname = cast(Any, robot.spec_fn).__qualname__
    assert qualname == cast(Any, uni_robot.spec_fn).__qualname__
    assert "with_osl_sensors" in qualname
    # Stage 1's leg-contact sensor, once: unicycle stage 2's scene sensor.
    assert cfg.scene.sensors == stage1.scene.sensors == unicycle.scene.sensors

    assert cfg.events == stage1.events
    assert cfg.terminations == stage1.terminations
    assert "trajectory_end" in cfg.terminations
    assert cfg.episode_length_s == stage1.episode_length_s
    if mode == "train":
      assert cfg.episode_length_s == 300.0  # The clip's end ends an episode.
      root = cfg.terminations["root_deviation"]
      assert root.params["threshold"] == 1.0

    assert set(cfg.rewards) == {"mimic"}
    mimic, stage1_mimic = cfg.rewards["mimic"], stage1.rewards["mimic"]
    assert mimic.func is split_mimic_reward and mimic.weight == 1.0
    params = dict(mimic.params)
    assert params.pop("channels") == SplitRewardCfg()
    # Stairs stage 1's reward, the CLF and its regularizers included.
    assert params == dict(stage1_mimic.params)

    split = {name for name in unicycle.metrics if name.startswith("split_")}
    assert split and set(cfg.metrics) == set(stage1.metrics) | split
    for name in split:
      assert cfg.metrics[name] == unicycle.metrics[name]


@pytest.mark.parametrize("osl_obs", SENSOR_SETS)
def test_osl_group_is_unicycle_stage2s(osl_obs: OslObsCfg) -> None:
  """For every sensor set and mode, the OSL actor reads what unicycle stage 2's
  reads (the drop rule included: no contact term once both contacts are off)."""
  for mode in MODES:
    md = cast(Any, mode)
    cfg = osl_split_stairs_env_cfg("msk", "compliant", "clf", md, osl_obs=osl_obs)
    unicycle = osl_split_env_cfg("msk", "compliant", "clf", md, osl_obs=osl_obs)
    group = cfg.observations["osl"]
    assert group == unicycle.observations["osl"]
    contacts = osl_obs.heel_contact or osl_obs.toe_contact
    assert ("contact" in group.terms) is contacts


def test_refusals() -> None:
  """Stage 1's perception switches (they would corrupt the human's observation),
  and the heel contact on the original touch sites, also after construction."""
  for dr_cfg in (OslDrCfg(encoder_bias=True), OslDrCfg(obs_noise=True)):
    with pytest.raises(ValueError, match=re.escape(PERCEPTION_DR_REFUSAL)):
      osl_split_stairs_env_cfg(dr_cfg=dr_cfg)
  with pytest.raises(ValueError, match="heel contact"):
    replace(osl_split_stairs_env_cfg(), osl_touch_sites="original")
  toe_only = osl_split_stairs_env_cfg(osl_obs=OslObsCfg(heel_contact=False))
  assert replace(toe_only, osl_touch_sites="original").osl_touch_sites == "original"


def test_cli_keeps_the_terrain_and_the_sensor_set(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """tyro rebuilds the config after the overrides: the terrain follows the clip,
  the OSL sensors and leg-contact sensor stay, the sensor flags give unicycle
  stage 2's group, and the heel contact on the original zones is refused."""
  motion = _write_clip(tmp_path)
  legacy = (
    f"{OSL_FLAG}.terms.loadcell.params.socket True "
    f"{OSL_FLAG}.terms.contact.params.heel False "
    f"{OSL_FLAG}.terms.contact.params.toe False "
    "--env.osl-touch-sites original"
  ).split()

  def parse(task: str, *flags: str) -> Any:
    clip = ["--env.commands.motion.motion-file", str(motion)]
    return tyro.cli(
      TrainConfig,
      args=[*clip, *flags] if task == TASK else list(flags),
      default=TrainConfig.from_task(task),
      config=mjlab.TYRO_FLAGS,
    ).env

  env = parse(TASK, *legacy)
  assert isinstance(env, OslSplitStairsEnvCfg)
  assert env.scene.spec_fn == TerrainBoxes(str(motion), "")
  assert "with_osl_sensors" in env.scene.entities["robot"].spec_fn.__qualname__
  assert "osl_leg_contact" in [sensor.name for sensor in env.scene.sensors]
  assert env.osl_touch_sites == "original"
  group = env.observations["osl"]
  assert group == parse(UNICYCLE_TASK, *legacy).observations["osl"]
  assert "contact" not in group.terms and group.terms["loadcell"].params["socket"]
  capsys.readouterr()
  with pytest.raises(SystemExit):  # tyro reports the config's refusal.
    parse(TASK, "--env.osl-touch-sites", "original")
  assert "heel contact" in capsys.readouterr().err


##
# Runner: evaluation, terrain, checkpoints.
##


def _bare_runner(train: Any) -> Any:
  runner: Any = object.__new__(StairsSplitRunner)
  runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=train))
  runner.validation_cfg = dict(asdict(load_rl_cfg(TASK))["validation"])
  return runner


def test_evaluation_envs_play_the_training_clip(tmp_path: Path) -> None:
  """Validation and video configs are the task's own modes (videos on the nominal
  plant) on the training clip and terrain, scoring the trained reward and
  reading the training OSL group exactly, each bus at the middle of its lag;
  stress copies the training config without its seed."""
  train = _training_cfg(tmp_path, osl_obs=OslObsCfg(history_length=4))
  train.rewards["mimic"].params["channels"] = SplitRewardCfg(owner_penalties=False)
  train_osl = train.observations["osl"]
  train_osl.terms["contact"].delay_max_lag = 4
  train.seed = 7
  runner = _bare_runner(train)
  train_motion = cast(StairsMotionCommandCfg, train.commands["motion"])
  for purpose in ("validation", "video", "stress"):
    cfg = runner._eval_env_cfg(purpose)
    assert isinstance(cfg, OslSplitStairsEnvCfg), purpose
    motion = cast(StairsMotionCommandCfg, cfg.commands["motion"])
    assert (motion.motion_file, motion.terrain_file) == (
      train_motion.motion_file,
      train_motion.terrain_file,
    )
    assert cfg.scene.spec_fn == train.scene.spec_fn
    assert motion.random_start is (purpose == "stress")
    assert cfg.rewards["mimic"].params == train.rewards["mimic"].params
    if purpose == "stress":
      assert cfg.seed is None and replace(cfg, seed=7) == train
      continue
    assert set(cfg.events) == {"reset_scene_to_default"}
    assert "root_deviation" not in cfg.terminations
    group = cfg.observations["osl"]
    assert group == osl_validation_group(train_osl, osl_validation_lags(train_osl))
    assert not group.enable_corruption
    contact = group.terms["contact"]
    assert (contact.delay_min_lag, contact.delay_max_lag) == (2, 2)
    assert group.terms["actpack"].history_length == 4
  # A training env on no stairs clip is not this runner's.
  runner.env.unwrapped.cfg = osl_split_env_cfg("msk", "compliant", "clf", "train")
  with pytest.raises(TypeError, match="Not a stairs training config"):
    runner._eval_env_cfg("validation")


def test_validation_plays_one_clip_from_100_envs() -> None:
  """Stairs stage 1's validation: ``validation.num_envs`` envs, one clip length
  (``validation.num_steps`` 0), no clip groups."""
  runner = _bare_runner(None)
  assert runner._validation_num_envs() == 100
  assert runner.validation_cfg["num_steps"] == 0
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(num_frames=731)),
    max_episode_length=30000,
    num_envs=100,
    device="cpu",
  )
  assert runner._default_validation_steps(env) == 731
  assert runner._validation_groups(env) == {}


def test_learn_copies_the_terrain_first(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  source = resolve_terrain_file(str(_write_clip(tmp_path)))
  copy = tmp_path / "run" / "params" / "terrain.json"
  calls: list[tuple[int, bool, str]] = []

  def learn(self: Any, num: int, init_at_random_ep_len: bool = False) -> None:
    calls.append((num, init_at_random_ep_len, copy.read_text()))

  monkeypatch.setattr(SplitOnPolicyRunner, "learn", learn)
  runner = object.__new__(StairsSplitRunner)
  cmd = SimpleNamespace(terrain_path=source)
  runner.env = cast(
    Any,
    SimpleNamespace(
      unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd))
    ),
  )
  runner.logger = cast(
    Any, SimpleNamespace(log_dir=str(tmp_path / "run"), disable_logs=False)
  )
  runner.learn(3)
  assert calls == [(3, False, source.read_text())]


def test_checkpoints_record_the_clip_and_terrain(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Every checkpoint records the clip and the terrain's digest; loading one that
  trained on others warns."""
  train = _training_cfg(tmp_path)
  identity = stairs_identity(train)
  assert identity == {
    "clip": "clip.npz",
    "terrain_sha256": _sha256(tmp_path / "clip_terrain.json"),
  }
  saved: list[tuple[str, Any]] = []
  monkeypatch.setattr(
    SplitOnPolicyRunner,
    "save",
    lambda self, path, infos=None: saved.append((path, infos)),
  )
  runner = _bare_runner(train)
  runner.save("model_5.pt", {"note": 1})
  assert saved == [("model_5.pt", {"note": 1, STAIRS_INFO_KEY: identity})]

  loaded: dict[str, Any] = {STAIRS_INFO_KEY: identity}
  monkeypatch.setattr(
    SplitOnPolicyRunner, "load", lambda self, *args, **kwargs: dict(loaded)
  )
  capsys.readouterr()
  assert runner.load("model_5.pt") == loaded
  assert "[WARN]" not in capsys.readouterr().out
  loaded[STAIRS_INFO_KEY] = {"clip": "other.npz", "terrain_sha256": "0" * 64}
  runner.load("model_5.pt")
  out = capsys.readouterr().out
  assert "trained on the clip other.npz; the env plays clip.npz" in out
  assert "another terrain (SHA-256 000000000000)" in out


##
# Warm start from a stairs stage-1 run.
##


def _stage1_checkpoint() -> dict[str, Any]:
  """A stairs stage-1 checkpoint of the fake env's size, as stage 1's runner saves
  it (its touch-site key included)."""
  torch.manual_seed(0)
  obs = TensorDict(
    {"actor": torch.zeros(1, ACTOR_DIM), "critic": torch.zeros(1, ACTOR_DIM)},
    batch_size=[1],
  )
  groups = {"actor": ["actor"], "critic": ["critic"]}
  dist = {"class_name": "GaussianDistribution", "init_std": 0.07, "std_type": "log"}
  actor = MuscleMimicMLPModel(
    obs, groups, "actor", HUMAN_DIM + 2, hidden_dims=HIDDEN, distribution_cfg=dist
  )
  critic = MuscleMimicMLPModel(obs, groups, "critic", 1, hidden_dims=HIDDEN)
  optimizer = MuonAdam(chain(actor.parameters(), critic.parameters()), lr=4e-4)
  updates = STAGE1_ITER + 1
  actor_state = actor.state_dict()
  assert LOG_STD in actor_state
  return {
    "actor_state_dict": actor_state,
    "critic_state_dict": critic.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "muscle_mimic_state": {
      "num_updates_done": updates,
      "return_mean": torch.tensor(10.0),
      "return_var": torch.tensor(50.0),
      "return_count": torch.tensor(updates * 80 * 4096 + 1e-4),
    },
    "iter": STAGE1_ITER,
    "infos": {"env_state": {"common_step_counter": updates * 80}},
    CHECKPOINT_KEY: "feet_extended",
  }


def _stage1_run(
  root: Path, clip: Path, kind: str = "stairs", pen_w: float = 0.5
) -> Path:
  """A stage-1 run as ``train`` writes it: ``kind`` ``"stairs"`` (stairs stage
  1 on ``clip``, its terrain copied), ``"flat"`` (osl_mimic on ``clip``, flat
  ground) or ``"library"`` (a unicycle run's env.yaml, as text)."""
  run_dir = root / kind / "logs" / "rsl_rl" / STAGE1_EXPERIMENT / "2026-11-14_00-00-00"
  agent = osl_stairs_runner_cfg("msk", "compliant", "clf", STAGE1_EXPERIMENT)
  agent.actor.hidden_dims = HIDDEN
  agent.critic.hidden_dims = HIDDEN
  # A short run standing for a finished one: no reward ramp still in force at its
  # end (the lineage records the ramps' values there).
  agent.env_stages = ()
  dump_yaml(run_dir / "params" / "agent.yaml", asdict(agent))
  env_yaml = run_dir / "params" / "env.yaml"
  if kind == "library":
    env_yaml.write_text("commands:\n  motion:\n    library_dir: /data/unicycle_v001\n")
  else:
    env = (osl_stairs_env_cfg if kind == "stairs" else osl_mimic_env_cfg)(
      "msk", "compliant", "clf", "train"
    )
    env.commands["motion"].motion_file = str(clip)  # type: ignore[attr-defined]
    if kind == "stairs":
      sync_terrain(env)
      shutil.copyfile(
        resolve_terrain_file(str(clip)), run_dir / "params" / "terrain.json"
      )
    env.rewards["mimic"].params["clf"].pen_w = pen_w
    dump_yaml(env_yaml, asdict(env))
  torch.save(_stage1_checkpoint(), run_dir / f"model_{STAGE1_ITER}.pt")
  return run_dir


def _runner(train: OslSplitStairsEnvCfg) -> StairsSplitRunner:
  """The stairs runner over the fake env, whose config is ``train``."""
  env = _FakeEnv()
  env.cfg = train  # The real training config: its clip, terrain and pen_w.
  cfg = _train_cfg(None)
  cfg["validation"]["variant"] = "stairs"
  return StairsSplitRunner(cast(Any, env), cfg, None, "cpu")


def test_warm_start_from_a_stairs_stage1_run(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """The converter takes a stairs stage-1 run as it is (no variant, no library,
  stage 1's pen_w and root limit, its touch sites), and the runner starts from it
  without the gait-library variant warning."""
  train = _training_cfg(tmp_path)
  ws = build_warm_start(_stage1_run(tmp_path, tmp_path / "clip.npz"))
  lineage = ws.lineage
  assert lineage["variant"] is None and lineage["library_dir"] is None
  assert lineage["library_release"] is None
  assert lineage["experiment_name"] == STAGE1_EXPERIMENT
  assert lineage["stage1_env"]["clf_pen_w"] == 0.5
  assert lineage["stage1_env"]["root_deviation"] == 1.0
  assert lineage["stage1_env"]["twist_weight"] is None
  assert lineage_touch_sites(lineage) == "feet_extended"
  assert ActionLayout.from_list(lineage["action_terms"]).terms == (
    ("osl_motors", 2),
    ("muscles", HUMAN_DIM),
  )
  assert check_stairs_warm_start(lineage, train) == []
  runner = _runner(train)
  capsys.readouterr()
  runner.warm_start(ws)
  out = capsys.readouterr().out
  assert runner._ready and runner.stage_start == STAGE1_ITER + 1
  assert "variant" not in out and "Stage 2 starts at iteration 41" in out


def test_warm_start_refuses_another_pen_w(tmp_path: Path) -> None:
  train = _training_cfg(tmp_path)
  run = _stage1_run(tmp_path, tmp_path / "clip.npz", pen_w=0.25)
  runner = _runner(train)
  with pytest.raises(ValueError, match="pen_w x pen_scale 0.25 vs 0.5"):
    runner.warm_start(build_warm_start(run))
  assert not runner._ready


def test_warm_start_provenance(tmp_path: Path) -> None:
  """A run that trained without a terrain, or that has no terrain copy, is
  refused before anything is applied; another clip or terrain is reported."""
  train = _training_cfg(tmp_path)
  clip = tmp_path / "clip.npz"
  flat = build_warm_start(_stage1_run(tmp_path, clip, kind="flat"))
  with pytest.raises(ValueError, match="on flat ground .* no terrain"):
    check_stairs_warm_start(flat.lineage, train)
  runner = _runner(train)
  with pytest.raises(ValueError, match="stairs stage-1 run"):
    runner.warm_start(flat)
  assert not runner._ready and runner.stage_start == 0
  library = build_warm_start(_stage1_run(tmp_path, clip, kind="library"))
  with pytest.raises(ValueError, match="on the motion library /data/unicycle_v001"):
    check_stairs_warm_start(library.lineage, train)

  run = _stage1_run(tmp_path, clip)
  lineage = build_warm_start(run).lineage
  (run / "params" / "terrain.json").unlink()
  with pytest.raises(ValueError, match="has no params/terrain.json"):
    check_stairs_warm_start(lineage, train)

  other = tmp_path / "other"
  other.mkdir()
  boxes = [{**box, "yaw": 0.0} for box in BOXES]
  other_clip = _write_clip(other, boxes).rename(other / "other_clip.npz")
  (other / "clip_terrain.json").rename(other / "other_clip_terrain.json")
  lineage = build_warm_start(_stage1_run(other, other_clip)).lineage
  warnings = check_stairs_warm_start(lineage, train)
  assert (
    warnings[0] == "Stage 1 trained on the clip other_clip.npz; stage 2 plays clip.npz."
  )
  assert warnings[1].startswith("Stage 1 trained on another terrain")
  assert len(warnings) == 2


##
# The real clip.
##


@needs_clip
def test_env_builds_on_the_clip() -> None:
  """The MSK-CLF-compliant env on the clip: stairs stage 1's human input (2333
  observations, the warm start's), unicycle stage 2's OSL input, finite reward
  channels."""
  cfg = osl_split_stairs_env_cfg("msk", "compliant", "clf", "train")
  motion = cast(StairsMotionCommandCfg, cfg.commands["motion"])
  motion.motion_file = str(MOTION)
  sync_terrain(cfg)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    dims = env.observation_manager.group_obs_dim
    assert dims["actor"] == (2333,)
    assert dims["osl"] == (osl_obs_dim(OslObsCfg()),)
    layout = ActionLayout.from_action_manager(env.action_manager)
    assert layout.terms == (("osl_motors", 2), ("muscles", 339))
    env.reset()
    term = cast(Any, env.reward_manager.get_term_cfg("mimic").func)
    actions = torch.zeros(2, env.action_manager.total_action_dim, device=env.device)
    for _ in range(10):
      _, reward, *_ = env.step(actions)
      assert bool(torch.isfinite(reward).all())
      assert bool(torch.isfinite(term.channel_rewards).all())
  finally:
    env.close()


@pytest.mark.slow
@pytest.mark.skipif(
  not (os.environ.get(STAGE1_RUN_ENV_VAR) and MOTION.is_file()),
  reason=f"Set {STAGE1_RUN_ENV_VAR} (a stairs stage-1 run, checkpoint or converted "
  "warm start) and MJLAB_OSL_STAIRS_MOTION (its clip) to run the smoke test.",
)
def test_stairs_stage2_smoke(tmp_path: Path) -> None:
  """Two stairs stage-2 iterations of the real task on the test device: warm start,
  the terrain copy, zero-shot and regular validations over one clip length and
  stress validations, checkpoints with the clip and their exports, then play's
  load."""
  device = get_test_device()
  ws_path = tmp_path / "ws.pt"
  ws = load_warm_start(os.environ[STAGE1_RUN_ENV_VAR])
  ws.save(ws_path)
  lineage = ws.lineage
  task = stairs_split_task_id(lineage["model"], lineage["socket"], lineage["reward"])

  def make_env(play: bool, num_envs: int) -> RslRlVecEnvWrapper:
    env_cfg = load_env_cfg(task, play=play)
    set_touch_sites(env_cfg, lineage_touch_sites(lineage))
    env_cfg.scene.num_envs = num_envs
    motion = env_cfg.commands["motion"]
    assert isinstance(motion, StairsMotionCommandCfg)
    motion.motion_file = str(MOTION)
    sync_terrain(env_cfg)
    return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  agent = load_rl_cfg(task)
  assert isinstance(agent, SplitRunnerCfg)
  agent.num_steps_per_env = 8
  agent.algorithm.num_mini_batches = 2
  if device == "cpu":  # bfloat16 Newton-Schulz is ~50x slower on most CPUs.
    agent.algorithm.muon_ns_dtype = "float32"
  agent.max_iterations = 2
  agent.save_interval = 1
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.video.enabled = False
  agent.freeze.human_actor_iterations = 1
  agent.freeze.critic_warmup_iterations = 0
  agent.blend.ramp_iterations = 1
  agent.validation.interval = 1
  agent.validation.num_envs = 4  # num_steps 0: one clip length.
  agent.validation.stress_interval = 1
  agent.validation.stress_num_envs = 4
  agent.validation.stress_num_steps = 20
  agent.warm_start.path = str(ws_path)
  # Any stairs stage-1 run will do for plumbing, whatever its reward settings.
  agent.warm_start.allow_reward_mismatch = True

  env = make_env(play=False, num_envs=8)
  log_dir = tmp_path / "run"
  runner = StairsSplitRunner(env, asdict(agent), str(log_dir), device)
  runner.learn(agent.max_iterations)
  runner.close()
  env.close()
  start = int(lineage["iter"]) + 1
  last = log_dir / f"model_{start + 1}.pt"
  assert not (log_dir / f"model_{start}.pt").exists()
  assert last.is_file()
  assert (log_dir / "exported" / f"osl_policy_{start + 1}.onnx").is_file()
  terrain = resolve_terrain_file(str(MOTION))
  assert (log_dir / "params" / "terrain.json").read_bytes() == terrain.read_bytes()
  infos = torch.load(last, map_location="cpu", weights_only=False)["infos"]
  assert infos[STAIRS_INFO_KEY] == {
    "clip": MOTION.name,
    "terrain_sha256": _sha256(terrain),
  }
  events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  events.Reload()

  def steps(tag: str) -> set[int]:
    return {e.step for e in events.Scalars(tag)}

  for tag in ("Validation/early_termination_rate", "Stress/survival_rate"):
    assert steps(tag) == {start, start + 1}, tag
  assert steps("ValidationTeacher/early_termination_rate") == {start}

  play_env = make_env(play=True, num_envs=2)
  played = StairsSplitRunner(play_env, asdict(load_rl_cfg(task)), device=device)
  played.load(str(last), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = played.get_inference_policy(device=device)
  obs = play_env.get_observations()
  with torch.inference_mode():
    for _ in range(5):
      obs, _, _, _ = play_env.step(policy(obs))
  for group in ("actor", "osl"):
    assert bool(torch.isfinite(obs_group(obs, group)).all())
  play_env.close()
