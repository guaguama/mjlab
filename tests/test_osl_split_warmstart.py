"""Tests for the stage-1 to stage-2 warm start (``osl_split.rl.warmstart``) and its
command line (``osl_split.scripts.split``).

Everything runs on CPU on synthetic stage-1 runs: an ``agent.yaml`` written the way
``train.py`` writes it (from the real stage-1 runner config, with tiny hidden dims),
checkpoints with the real key structure, and a W&B log naming the run, for both
action layouts (musculoskeletal: OSL rows first; robot: OSL rows last).
``MJLAB_OSL_SPLIT_STAGE1_RUN`` names a real stage-1 run directory to convert too
(read-only, CPU, a few seconds).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from tensordict import TensorDict

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_gait_library.config.rl_cfg import osl_gait_library_runner_cfg
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward, _actions
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_mimic.rl.muon import MuonAdam
from mjlab.tasks.osl_split.config.rl_cfg import osl_split_runner_cfg
from mjlab.tasks.osl_split.rl.layout import ActionLayout
from mjlab.tasks.osl_split.rl.warmstart import (
  LINEAGE_KEYS,
  LOG_STD,
  WARM_START_FORMAT,
  WarmStart,
  architecture_mismatches,
  build_warm_start,
  check_compatible,
  default_output_path,
  find_wandb_run_id,
  lineage_yaml_path,
  load_warm_start,
  resolve_stage1_checkpoint,
  verify_warm_start,
)
from mjlab.tasks.osl_split.scripts.split import main
from mjlab.utils.os import dump_yaml

OBS_DIM = 11
HIDDEN = (8, 6)
LAST = "mlp.6"
"""The output layer: Linear, LayerNorm and SiLU per hidden layer, then a Linear."""
DISTRIBUTION = {
  "class_name": "GaussianDistribution",
  "init_std": 3.0,
  "std_type": "log",
}
OUT_DIM: dict[str, int] = {"msk": 341, "robot": 29}
OSL_ROWS: dict[str, list[int]] = {"msk": [0, 1], "robot": [27, 28]}
NUM_ENVS, NUM_STEPS = 4096, 80
WANDB_ID = "abcd1234"
DEBUG_LOG, CONFIG_YAML = "logs/debug.log", "files/config.yaml"
LIBRARY = "/data/clipper/unicycle_v001"
STAGE1_RUN = os.environ.get("MJLAB_OSL_SPLIT_STAGE1_RUN", "")


##
# A synthetic stage-1 run.
##


def _write_agent_yaml(
  run_dir: Path, model: Model, socket: Socket, reward: Reward, experiment: str
) -> None:
  """As ``train.py`` writes it: ``dump_yaml(asdict(runner_cfg))``."""
  cfg = osl_gait_library_runner_cfg(
    model, socket, reward, experiment, variant="unicycle"
  )
  cfg.actor.hidden_dims = HIDDEN
  cfg.critic.hidden_dims = HIDDEN
  dump_yaml(run_dir / "params" / "agent.yaml", asdict(cfg))


def _stage1_checkpoint(model: Model, iteration: int) -> dict[str, Any]:
  """The keys ``MuscleMimicPPO.save`` and ``MjlabOnPolicyRunner.save`` write, with
  every tensor random (exact copies and exact slices then differ from mistakes)."""
  generator = torch.Generator().manual_seed(iteration)
  obs = TensorDict(
    {"actor": torch.zeros(1, OBS_DIM), "critic": torch.zeros(1, OBS_DIM)},
    batch_size=[1],
  )
  groups = {"actor": ["actor"], "critic": ["critic"]}
  actor = MuscleMimicMLPModel(
    obs,
    groups,
    "actor",
    OUT_DIM[model],
    hidden_dims=HIDDEN,
    distribution_cfg=DISTRIBUTION,
  )
  critic = MuscleMimicMLPModel(obs, groups, "critic", 1, hidden_dims=HIDDEN)

  def randomized(net: MuscleMimicMLPModel) -> dict[str, torch.Tensor]:
    state = {
      key: 0.5 * torch.randn(value.shape, generator=generator)
      for key, value in net.state_dict().items()
    }
    state["obs_normalizer.var"] = 0.5 + torch.rand(OBS_DIM, generator=generator)
    state["obs_normalizer.count"] = torch.tensor(2.0e8)
    return state

  actor_state = randomized(actor)
  actor_state[LOG_STD] = 4.0 * torch.rand(OUT_DIM[model], generator=generator) - 3.0
  updates = iteration + 1
  optimizer = MuonAdam(chain(actor.parameters(), critic.parameters()), lr=4e-4)
  return {
    "actor_state_dict": actor_state,
    "critic_state_dict": randomized(critic),
    "optimizer_state_dict": optimizer.state_dict(),
    "muscle_mimic_state": {
      "num_updates_done": updates,
      "return_mean": torch.tensor(11.9552),
      "return_var": torch.tensor(97.9229),
      "return_count": torch.tensor(updates * NUM_STEPS * NUM_ENVS + 1e-4),
    },
    "iter": iteration,
    "infos": {"env_state": {"common_step_counter": updates * NUM_STEPS}},
  }


def _stage1_run(
  root: Path,
  model: Model,
  socket: Socket = "compliant",
  reward: Reward = "clf",
  iterations: tuple[int, ...] = (9, 12),
) -> Path:
  """``<root>/logs/rsl_rl/<experiment>/<run>`` plus ``<root>/wandb`` naming it."""
  experiment = f"osl_fullbody_unicycle_{model}_{reward}_{socket}"
  run_dir = root / "logs" / "rsl_rl" / experiment / "2026-11-06_18-40-18"
  _write_agent_yaml(run_dir, model, socket, reward, experiment)
  (run_dir / "params" / "env.yaml").write_text(
    "scene:\n  entities:\n    robot:\n      spec_fn: "
    "!!python/name:mjlab.tasks.osl_gait_library.config.dr.fn ''\n"
    f"commands:\n  motion:\n    library_dir: {LIBRARY}\n"
  )
  for iteration in iterations:
    torch.save(_stage1_checkpoint(model, iteration), run_dir / f"model_{iteration}.pt")
  _wandb_run(root / "wandb", f"run-20261106_184026-{WANDB_ID}", run_dir.resolve())
  return run_dir


def _wandb_run(
  root: Path,
  run: str,
  log_dir: Path,
  warm_start: Path | None = None,
  files: tuple[str, ...] = (DEBUG_LOG,),
) -> None:
  """A W&B run directory as rsl_rl's ``WandbLogWriter`` leaves it (wandb 0.22 and
  0.30): ``debug.log`` echoes ``wandb.init``'s config, then the train config
  (``config_cb``); a finished run's ``config.yaml`` holds every key under ``value``
  and the command line under ``_wandb``. A stage-2 run started from ``warm_start``
  names it in both places, and so in both files."""
  train_cfg: dict[str, Any] = {"seed": 42}
  args: list[str] = []
  if warm_start is not None:
    train_cfg["warm_start"] = {"path": str(warm_start), "checkpoint": ""}
    args = ["--agent.warm-start.path", str(warm_start)]
  init = {"log_dir": str(log_dir), "_wandb": {}}
  callback = {"train_cfg": train_cfg}
  text = {
    DEBUG_LOG: (
      "2026-11-08 10:00:05,131 INFO    MainThread:1 [wandb_init.py:init():788] "
      f"wandb.init called with sweep_config: {{}}\nconfig: {init}\n"
      "2026-11-08 10:00:05,563 INFO    MainThread:1 [wandb_run.py:"
      f"_config_callback():1283] config_cb None None {callback}\n"
    ),
    CONFIG_YAML: yaml.safe_dump(
      {
        "_wandb": {"value": {"e": {"host": {"args": args}}}},
        "log_dir": {"value": str(log_dir)},
        "train_cfg": {"value": train_cfg},
      },
      indent=4,
      sort_keys=False,
    ),
  }
  for name in files:
    _wandb_file(root, run, name, text[name])


def _wandb_file(root: Path, run: str, name: str, text: str) -> None:
  path = root / run / name
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text)


def _model(out_dim: int, state: dict[str, torch.Tensor]) -> MuscleMimicMLPModel:
  obs = TensorDict({"actor": torch.zeros(1, OBS_DIM)}, batch_size=[1])
  net = MuscleMimicMLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    out_dim,
    hidden_dims=HIDDEN,
    distribution_cfg=DISTRIBUTION,
  )
  net.load_state_dict(state, strict=True)
  return net


def _human_rows(model: Model) -> list[int]:
  return [row for row in range(OUT_DIM[model]) if row not in OSL_ROWS[model]]


##
# Conversion.
##


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_stage1_layout_matches_osl_mimic(model: Model) -> None:
  layout = ActionLayout.for_model(model, OUT_DIM[model])
  assert [name for name, _ in layout.terms] == list(_actions(model))
  assert layout.rows("osl") == OSL_ROWS[model]


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_split_is_exact(tmp_path: Path, model: Model) -> None:
  run_dir = _stage1_run(tmp_path, model)
  ws = build_warm_start(run_dir)
  stage1 = torch.load(run_dir / "model_12.pt", weights_only=False)
  actor, critic = stage1["actor_state_dict"], stage1["critic_state_dict"]
  human, osl = _human_rows(model), OSL_ROWS[model]

  # Every actor tensor is copied, the output rows and log-stds restricted to the
  # human's rows; the critic is verbatim; the teacher head is the OSL rows.
  assert list(ws.human_actor) == list(actor)
  for key, value in actor.items():
    sliced = key in {f"{LAST}.weight", f"{LAST}.bias", LOG_STD}
    assert torch.equal(ws.human_actor[key], value[human] if sliced else value), key
  assert list(ws.human_critic) == list(critic)
  assert all(torch.equal(ws.human_critic[k], v) for k, v in critic.items())
  assert torch.equal(ws.teacher_head["weight"], actor[f"{LAST}.weight"][osl])
  assert torch.equal(ws.teacher_head["bias"], actor[f"{LAST}.bias"][osl])
  assert torch.equal(ws.teacher_head["log_std"], actor[LOG_STD][osl])
  for name in ("mean", "var", "count"):
    expected = stage1["muscle_mimic_state"][f"return_{name}"]
    assert ws.return_stats[name].shape == () and torch.equal(
      ws.return_stats[name], expected
    )
  assert ws.layout.rows("osl") == osl and ws.layout.rows("human") == human

  # The split networks reproduce the unified actor's rows.
  unified = _model(OUT_DIM[model], actor)
  split = _model(len(human), ws.human_actor)
  x = TensorDict({"actor": 3.0 * torch.randn(32, OBS_DIM)}, batch_size=[32])
  with torch.no_grad():
    full = unified(x)
    features = split.mlp[:-1](split.get_latent(x))
    teacher = features @ ws.teacher_head["weight"].T + ws.teacher_head["bias"]
    torch.testing.assert_close(split(x), full[:, human], rtol=0.0, atol=1e-6)
  torch.testing.assert_close(teacher, full[:, osl], rtol=0.0, atol=1e-6)
  assert 0.0 <= ws.lineage["verified_max_abs_diff"] <= 1e-5


@pytest.mark.parametrize("model", ["msk", "robot"])
def test_lineage(tmp_path: Path, model: Model) -> None:
  run_dir = _stage1_run(tmp_path, model)
  ws = build_warm_start(run_dir, verify=False)
  lineage = ws.lineage
  assert set(LINEAGE_KEYS) <= set(lineage)
  assert lineage["format"] == WARM_START_FORMAT
  assert lineage["stage1_run_dir"] == str(run_dir.resolve())
  assert lineage["stage1_checkpoint"] == str((run_dir / "model_12.pt").resolve())
  assert lineage["iter"] == 12 and lineage["num_updates_done"] == 13
  assert lineage["common_step_counter"] == 13 * NUM_STEPS
  assert lineage["env_steps"] == 13 * NUM_STEPS * NUM_ENVS
  task = ("model", "socket", "reward", "variant")
  assert tuple(lineage[k] for k in task) == (model, "compliant", "clf", "unicycle")
  assert lineage["experiment_name"] == run_dir.parent.name
  assert lineage["obs_dim"] == OBS_DIM
  layout = ActionLayout.for_model(model, OUT_DIM[model])
  assert lineage["action_terms"] == layout.to_list()
  assert lineage["human_rows"] == _human_rows(model)
  assert lineage["osl_rows"] == OSL_ROWS[model]
  # The architectures as agent.yaml holds them, tuples as lists.
  agent = yaml.load(
    (run_dir / "params" / "agent.yaml").read_text(), Loader=yaml.FullLoader
  )
  assert lineage["actor_cfg"] == json.loads(json.dumps(agent["actor"]))
  assert lineage["critic_cfg"] == json.loads(json.dumps(agent["critic"]))
  assert lineage["actor_cfg"]["hidden_dims"] == list(HIDDEN)
  std = torch.load(run_dir / "model_12.pt")["actor_state_dict"][LOG_STD].exp()
  human_std = std[_human_rows(model)]
  assert lineage["human_std"] == pytest.approx(
    {
      "min": float(human_std.min()),
      "median": float(human_std.median()),
      "max": float(human_std.max()),
    }
  )
  assert lineage["osl_std"] == pytest.approx(std[OSL_ROWS[model]].tolist())
  assert lineage["wandb_run_id"] == WANDB_ID
  assert datetime.fromisoformat(lineage["created"]).tzinfo is not None
  assert lineage["verified_max_abs_diff"] is None
  assert lineage["library_dir"] == LIBRARY
  assert yaml.safe_load(yaml.safe_dump(lineage)) == lineage  # Plain types only.


def test_resolve_stage1_checkpoint_is_numeric(tmp_path: Path) -> None:
  # Alphabetically model_9 sorts last; numerically model_610 is the newest.
  for name in ("model_9.pt", "model_70.pt", "model_610.pt", "model_best.pt"):
    (tmp_path / name).touch()
  (tmp_path / "model_7000.pt.bak").touch()
  (tmp_path / "warm_start").mkdir()
  (tmp_path / "warm_start" / "model_9999.pt").touch()
  assert resolve_stage1_checkpoint(tmp_path) == tmp_path / "model_610.pt"
  assert resolve_stage1_checkpoint(tmp_path, "model_70.pt") == tmp_path / "model_70.pt"
  assert resolve_stage1_checkpoint(tmp_path / "model_9.pt") == tmp_path / "model_9.pt"
  with pytest.raises(FileNotFoundError):
    resolve_stage1_checkpoint(tmp_path, "model_1.pt")
  with pytest.raises(FileNotFoundError):
    resolve_stage1_checkpoint(tmp_path / "warm_start" / "missing")
  with pytest.raises(ValueError):
    resolve_stage1_checkpoint(tmp_path / "model_9.pt", "model_70.pt")
  (tmp_path / "empty").mkdir()
  with pytest.raises(FileNotFoundError):
    resolve_stage1_checkpoint(tmp_path / "empty")
  out = default_output_path(tmp_path / "model_610.pt")
  assert out == tmp_path / "warm_start" / "model_610.pt"


def test_check_compatible(tmp_path: Path) -> None:
  ws = build_warm_start(_stage1_run(tmp_path, "msk"), verify=False)
  assert check_compatible(ws, model="msk", socket="compliant", reward="clf") is False
  # The reward switch is the one change allowed.
  assert check_compatible(ws, model="msk", socket="compliant", reward="mimic") is True
  with pytest.raises(ValueError, match="model cannot switch"):
    check_compatible(ws, model="robot", socket="compliant", reward="clf")
  with pytest.raises(ValueError, match="socket cannot switch"):
    check_compatible(ws, model="msk", socket="rigid", reward="clf")
  with pytest.raises(ValueError, match="Unknown reward"):
    check_compatible(ws, model="msk", socket="compliant", reward="ppo")


def test_architecture_mismatches(tmp_path: Path) -> None:
  # The stage-2 defaults rebuild the stage-1 ones: as dataclasses, at full size, and
  # against a lineage (lists for tuples; the runner drops a None distribution).
  stage1 = osl_gait_library_runner_cfg(
    "msk", "compliant", "clf", "test", variant="unicycle"
  )
  cfg = osl_split_runner_cfg("msk", "compliant", "clf", "test")
  for stage2 in (cfg.critic, cfg.osl_critic):
    assert architecture_mismatches(asdict(stage1.critic), asdict(stage2)) == []
  assert architecture_mismatches(asdict(stage1.actor), asdict(cfg.actor)) == []
  run_dir = _stage1_run(tmp_path, "msk", iterations=(12,))
  lineage = build_warm_start(run_dir, verify=False).lineage
  stage1_actor, stage1_critic = lineage["actor_cfg"], lineage["critic_cfg"]
  actor: dict[str, Any] = {**asdict(cfg.actor), "hidden_dims": HIDDEN}
  critic: dict[str, Any] = {**asdict(cfg.critic), "hidden_dims": HIDDEN}
  assert architecture_mismatches(stage1_actor, actor) == []
  del critic["distribution_cfg"]
  assert architecture_mismatches(stage1_critic, critic) == []

  # Each compared key, changed or unset on the stage-2 side.
  changes = {
    "hidden_dims": (8, 7),
    "activation": "elu",
    "layer_norm": False,
    "layer_norm_eps": 1e-6,
    "obs_normalization": False,
  }
  for key, value in changes.items():
    lines = architecture_mismatches(stage1_actor, {**actor, key: value})
    assert len(lines) == 1 and lines[0].startswith(f"{key}: stage 1 "), lines
  assert architecture_mismatches(stage1_actor, {**actor, "activation": "elu"}) == [
    "activation: stage 1 'swish', stage 2 'elu'"
  ]
  unset = {key: value for key, value in actor.items() if key != "layer_norm_eps"}
  assert architecture_mismatches(stage1_actor, unset) == [
    "layer_norm_eps: stage 1 1e-05, stage 2 not set"
  ]

  # The distribution's std_type, and its std_range where both set one.
  dist = actor["distribution_cfg"]
  scalar = {**actor, "distribution_cfg": {**dist, "std_type": "scalar"}}
  assert architecture_mismatches(stage1_actor, scalar) == [
    "distribution_cfg.std_type: stage 1 'log', stage 2 'scalar'"
  ]
  floored = {**actor, "distribution_cfg": {**dist, "std_range": (0.1, 1e6)}}
  assert architecture_mismatches(stage1_actor, floored) == []  # Stage 1 sets none.
  stage1_dist = {**stage1_actor["distribution_cfg"], "std_range": [0.1, 1e6]}
  stage1_floored = {**stage1_actor, "distribution_cfg": stage1_dist}
  assert architecture_mismatches(stage1_floored, floored) == []
  stage1_dist["std_range"] = [0.2, 1e6]
  assert architecture_mismatches(stage1_floored, floored) == [
    "distribution_cfg.std_range: stage 1 [0.2, 1000000.0], stage 2 [0.1, 1000000.0]"
  ]
  # A distribution on one side only.
  one_sided = {**critic, "distribution_cfg": dist}
  assert architecture_mismatches(stage1_critic, one_sided) == [
    f"distribution_cfg: stage 1 None, stage 2 {dist!r}"
  ]


def test_save_load_round_trip(tmp_path: Path) -> None:
  run_dir = _stage1_run(tmp_path, "robot")
  ws = build_warm_start(run_dir)
  path = tmp_path / "out" / "ws.pt"
  ws.save(path)
  assert lineage_yaml_path(path) == tmp_path / "out" / "ws.pt.yaml"
  assert yaml.safe_load(lineage_yaml_path(path).read_text()) == ws.lineage
  torch.load(path, weights_only=True)  # Plain types and tensors only.

  loaded = WarmStart.load(path)
  for name in ("human_actor", "human_critic", "teacher_head", "return_stats"):
    before, after = getattr(ws, name), getattr(loaded, name)
    assert list(before) == list(after)
    assert all(torch.equal(before[k], after[k]) for k in before), name
  assert loaded.lineage == ws.lineage and loaded.layout == ws.layout

  # load_warm_start reads a converted file as is and converts a run in memory.
  assert load_warm_start(path).lineage == ws.lineage
  assert load_warm_start(run_dir, "model_9.pt").lineage["iter"] == 9
  with pytest.raises(ValueError, match="is a warm-start file"):
    load_warm_start(path, "model_9.pt")
  with pytest.raises(ValueError, match="already a warm-start file"):
    build_warm_start(path)
  with pytest.raises(ValueError, match="not a warm-start file"):
    WarmStart.load(run_dir / "model_9.pt")


def test_verification_catches_a_wrong_split(tmp_path: Path) -> None:
  run_dir = _stage1_run(tmp_path, "msk")
  ws = build_warm_start(run_dir)
  actor = torch.load(run_dir / "model_12.pt")["actor_state_dict"]
  assert verify_warm_start(ws, actor) <= 1e-5
  # musclemimic's convention (the OSL rows last) is wrong for this layout.
  for key in (f"{LAST}.weight", f"{LAST}.bias", LOG_STD):
    ws.human_actor[key] = actor[key][:-2].clone()
  with pytest.raises(RuntimeError, match="verification failed"):
    verify_warm_start(ws, actor)


def test_refuses_other_inputs(tmp_path: Path) -> None:
  run_dir = _stage1_run(tmp_path, "msk")
  torch.save({"human_actor_state_dict": {}, "iter": 3}, run_dir / "model_99.pt")
  with pytest.raises(ValueError, match="stage-2 checkpoint"):
    build_warm_start(run_dir)
  # A checkpoint away from its run's params/agent.yaml.
  (tmp_path / "loose").mkdir()
  (run_dir / "model_12.pt").rename(tmp_path / "loose" / "model_12.pt")
  with pytest.raises(FileNotFoundError, match="agent.yaml"):
    build_warm_start(tmp_path / "loose")
  # A musculoskeletal checkpoint whose run claims the robot model.
  _write_agent_yaml(run_dir, "robot", "compliant", "clf", "mislabelled")
  with pytest.raises(ValueError, match="'servos' term drives 27 actuators"):
    build_warm_start(run_dir, "model_9.pt")


def test_find_wandb_run_id(tmp_path: Path) -> None:
  repo = tmp_path.resolve() / "repo"
  run_dir = repo / "logs" / "rsl_rl" / "exp" / "2026-11-05_02-28-46_v1"
  run_dir.mkdir(parents=True)
  wandb = repo / "wandb"
  # A newer run whose directory merely starts with this one's name.
  restart = Path(f"{run_dir}_restart")
  _wandb_run(
    wandb, "run-20261105_031748-restart1", restart, files=(DEBUG_LOG, CONFIG_YAML)
  )
  # This run, finished, found from its config.yaml alone.
  _wandb_run(wandb, "run-20261105_022859-r7r2vnis", run_dir, files=(CONFIG_YAML,))
  _wandb_file(wandb, "run-20261104_101010-other123", DEBUG_LOG, "config: {}\n")
  assert find_wandb_run_id(run_dir) == "r7r2vnis"  # Found from <repo>/wandb.
  assert find_wandb_run_id(str(run_dir), wandb_root=wandb) == "r7r2vnis"
  assert find_wandb_run_id(restart) == "restart1"
  assert find_wandb_run_id(run_dir, wandb_root=tmp_path / "nowhere") is None
  assert find_wandb_run_id(run_dir.parent / "other", wandb_root=wandb) is None


@pytest.mark.parametrize("name", [DEBUG_LOG, CONFIG_YAML])
@pytest.mark.parametrize("converted", [False, True], ids=["run_dir", "converted"])
def test_find_wandb_run_id_ignores_runs_started_from_it(
  tmp_path: Path, converted: bool, name: str
) -> None:
  """A later stage-2 run started from the stage-1 run, given its directory or the
  converter's default output inside it, logs that path in its own train config and
  command line. Only the ``log_dir`` that ``wandb.init`` recorded names a run."""
  root = tmp_path.resolve()
  run_dir = root / "logs" / "rsl_rl" / "exp" / "2026-11-06_18-40-18"
  run_dir.mkdir(parents=True)
  wandb = root / "wandb"
  _wandb_run(wandb, f"run-20261106_184026-{WANDB_ID}", run_dir, files=(name,))
  source = default_output_path(run_dir / "model_610.pt") if converted else run_dir
  stage2 = root / "logs" / "rsl_rl" / "split" / "2026-11-08_10-00-00"
  _wandb_run(wandb, "run-20261108_100005-stage2id", stage2, source, files=(name,))
  assert find_wandb_run_id(run_dir) == WANDB_ID
  assert find_wandb_run_id(stage2) == "stage2id"


##
# Command line.
##


def test_cli_writes_file_and_yaml(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  run_dir = _stage1_run(tmp_path, "msk")
  out = main([str(run_dir), "--out", str(tmp_path / "ws.pt")])
  assert out == tmp_path / "ws.pt" and lineage_yaml_path(out).is_file()
  ws = WarmStart.load(out)
  assert ws.lineage["iter"] == 12
  assert ws.lineage["verified_max_abs_diff"] is not None
  printed = capsys.readouterr().out
  assert "stage 2 starts at iteration 13" in printed and str(out) in printed

  # Defaults: beside the run; --checkpoint and --no-verify.
  default = main([str(run_dir), "--checkpoint", "model_9.pt", "--no-verify"])
  assert default == run_dir / "warm_start" / "model_9.pt"
  assert WarmStart.load(default).lineage["verified_max_abs_diff"] is None
  assert resolve_stage1_checkpoint(run_dir) == run_dir / "model_12.pt"
  ckpt = str(run_dir / "model_9.pt")
  with pytest.raises(SystemExit):
    main([ckpt, "--out", ckpt])


##
# A real stage-1 run (opt-in).
##


def _model_kwargs(cfg: Any) -> dict[str, Any]:
  """A runner config's model block as the model's keyword arguments."""
  skip = {"class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"}
  kwargs = {k: v for k, v in asdict(cfg).items() if k not in skip}
  return {k: v for k, v in kwargs.items() if v is not None}


@pytest.mark.skipif(
  not STAGE1_RUN, reason="Set MJLAB_OSL_SPLIT_STAGE1_RUN to a stage-1 run directory."
)
def test_real_stage1_run() -> None:
  """Converts the run in memory and loads the result into the networks stage 2
  builds from its own runner config. Reads the run; writes nothing."""
  ws = build_warm_start(STAGE1_RUN)
  lineage = ws.lineage
  model = lineage["model"]
  assert ws.layout.rows("osl") == OSL_ROWS[model]
  assert lineage["verified_max_abs_diff"] <= 1e-5
  assert Path(lineage["stage1_checkpoint"]).name == f"model_{lineage['iter']}.pt"

  cfg = osl_split_runner_cfg(model, lineage["socket"], lineage["reward"], "test")
  obs = TensorDict({"actor": torch.zeros(1, lineage["obs_dim"])}, batch_size=[1])
  groups = {"actor": ["actor"]}
  human = MuscleMimicMLPModel(
    obs, groups, "actor", ws.layout.dim("human"), **_model_kwargs(cfg.actor)
  )
  human.load_state_dict(ws.human_actor, strict=True)
  critic = MuscleMimicMLPModel(obs, groups, "actor", 1, **_model_kwargs(cfg.critic))
  critic.load_state_dict(ws.human_critic, strict=True)
