"""Tests for the two-stage pipeline driver (``osl_split/scripts/two_stage.py``).

Nothing heavy is launched: ``subprocess.Popen`` and ``subprocess.run`` are replaced
by fakes that play the steps (a training stage announces its run directory and
writes an event file and checkpoints, the split writes its output, ``nvidia-smi``
prints canned CSV), so everything runs on the CPU in seconds. Only the signal tests
and the launch test start real processes: the driver itself, run as its documented
file form with no terminal, over a dummy stage 1 that only waits for signals. The
driver's commands are parsed by ``train``'s real tyro CLI, the W&B pins are replayed
through the installed wandb's settings merge, and a combined TensorBoard dir is read
back by TensorBoard itself.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from itertools import count
from pathlib import Path
from typing import Any, get_args

import pytest
import tyro
from rsl_rl.utils.wandb_log_writer import WandbLogWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from wandb.sdk.wandb_settings import Settings

import mjlab
import mjlab.scripts.train
from mjlab.asset_zoo.robots.osl_fullbody import osl_constants
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg
from mjlab.tasks.osl_gait_library.config.rl_cfg import GaitLibraryRunnerCfg
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic import touch_sites as mimic_touch_sites
from mjlab.tasks.osl_mimic.config import env_cfgs as mimic_env_cfgs
from mjlab.tasks.osl_mimic.config.rl_cfg import MuscleMimicPpoCfg
from mjlab.tasks.osl_split import config as split_config
from mjlab.tasks.osl_split import scripts as split_scripts
from mjlab.tasks.osl_split.config.rl_cfg import SplitRunnerCfg, osl_split_runner_cfg
from mjlab.tasks.osl_split.rl.warmstart import (
  LINEAGE_KEYS,
  WARM_START_FORMAT,
  WarmStart,
  lineage_touch_sites,
  lineage_yaml_path,
)
from mjlab.tasks.osl_split.scripts import two_stage
from mjlab.tasks.osl_split.scripts.two_stage import (
  GpuProcess,
  PipelinePaths,
  Stage1Cfg,
  Stage2Cfg,
  TwoStageCfg,
  sync_combined_tb,
)
from mjlab.tasks.registry import list_tasks, load_rl_cfg

COMBOS: list[tuple[two_stage.Model, two_stage.Socket, two_stage.Reward]] = [
  (model, socket, reward)
  for model in ("robot", "msk")
  for socket in ("rigid", "compliant")
  for reward in ("mimic", "clf")
]
SMI_BUSY = (
  "498800, /home/u/mjlab/.venv/bin/python3, 17952 MiB\n"
  "4242, /opt/odd, name/python, [N/A]\n"
)
AGENT_YAML = """\
seed: 42
num_steps_per_env: 80
max_iterations: {max_iterations}
obs_groups:
  actor: !!python/tuple
  - actor
save_interval: {save_interval}
resume: {resume}
load_run: .*
load_checkpoint: {load_checkpoint}
validation:
  interval: 30
  num_envs: 0
  num_steps: 1000
  model: {model}
  socket: {socket}
  reward: {reward}
  variant: unicycle
export_osl_onnx: true
"""


@pytest.fixture
def library(tmp_path: Path) -> Path:
  path = tmp_path / "library"
  path.mkdir()
  return path


def make_cfg(tmp_path: Path, **changes: Any) -> TwoStageCfg:
  cfg = TwoStageCfg(
    name="p", library_dir=str(tmp_path / "library"), log_root=str(tmp_path / "logs")
  )
  return replace(cfg, **changes)


def make_stage1_run(
  run: Path,
  checkpoints: tuple[int, ...] = (0, 5, 11),
  model: str = "msk",
  socket: str = "compliant",
  reward: str = "clf",
  *,
  max_iterations: int | None = None,
  save_interval: int = 610,
  resume: str = "false",
  load_checkpoint: str = "model_.*.pt",
) -> Path:
  """A stage-1 run directory; by default a finished one, whose last checkpoint is
  its last iteration's."""
  (run / "params").mkdir(parents=True)
  (run / "params" / "agent.yaml").write_text(
    AGENT_YAML.format(
      model=model,
      socket=socket,
      reward=reward,
      max_iterations=max(checkpoints) + 1 if max_iterations is None else max_iterations,
      save_interval=save_interval,
      resume=resume,
      load_checkpoint=load_checkpoint,
    )
  )
  for iteration in checkpoints:
    (run / f"model_{iteration}.pt").write_bytes(b"")
  (run / "events.out.tfevents.1000.host.1.0").write_bytes(b"")
  return run


def split_lineage(touch_sites: str | None) -> dict[str, Any]:
  """A lineage as the split builds it, abridged (every required key, in order),
  recording stage 1's ``touch_sites`` after the socket (None: no key, as in a file
  converted before the setting)."""
  values: dict[str, Any] = {
    "format": WARM_START_FORMAT,
    "stage1_run_dir": "/logs/rsl_rl/osl_fullbody_unicycle_msk_clf_compliant/run",
    "iter": 11,
    "model": "msk",
    "socket": "compliant",
    "reward": "clf",
    "action_terms": [["osl_motors", 2], ["muscles", 339]],
    "actor_cfg": {"hidden_dims": [1024, 512, 256], "activation": "swish"},
    "human_std": {"min": 0.05, "median": 0.07, "max": 0.1},
    "osl_std": [0.12, 0.08],
  }
  lineage: dict[str, Any] = {}
  for key in LINEAGE_KEYS:
    lineage[key] = values.get(key)
    if key == "socket" and touch_sites is not None:
      lineage[mimic_touch_sites.ENV_FIELD] = touch_sites
  return lineage


def save_warm_start(path: Path, lineage: dict[str, Any]) -> None:
  """``path`` and its lineage YAML, written by the split's own ``WarmStart.save``."""
  empty: dict[str, Any] = {}
  WarmStart(empty, empty, empty, empty, lineage).save(path)


##
# Fakes for the launched steps and for nvidia-smi.
##


Output = list[str | Callable[[], None]]


class FakeStdout:
  """A step's output: lines, and actions taken while it is being read (the driver
  receiving a signal, say)."""

  def __init__(self, items: Output) -> None:
    self._items = list(items)

  def __iter__(self) -> Iterator[str]:
    return self

  def __next__(self) -> str:
    while self._items:
      item = self._items.pop(0)
      if isinstance(item, str):
        return item
      item()
    raise StopIteration

  def close(self) -> None:
    pass


_PIDS = count(2**30)
"""Fake process ids, above any ``pid_max`` (at most 2**22), so that a signal to a
fake step's process group can never reach a real one."""


class FakeStep:
  """A launched step (``subprocess.Popen``) that has already run to ``code``."""

  def __init__(self, items: Output, code: int = 0) -> None:
    self.pid = next(_PIDS)
    self.stdout = FakeStdout(items)
    self.code = code
    self.returncode: int | None = None

  def wait(self) -> int:
    self.returncode = self.code
    return self.code


@dataclass
class Launch:
  args: list[str]
  env: dict[str, str]


def step_name(args: list[str]) -> str:
  if args[2] == two_stage.SPLIT_MODULE:
    return "split"
  return "stage1" if args[3].startswith(two_stage.STAGE1_TASK) else "stage2"


def fake_steps(
  monkeypatch: pytest.MonkeyPatch,
  root: Path,
  codes: dict[str, int] | None = None,
  events: list[str] | None = None,
  lineage: dict[str, Any] | None = None,
) -> list[Launch]:
  """Replace ``subprocess.Popen`` with the three steps' parts; return the launches.

  A training stage creates its run directory (stage 1 also its checkpoints and
  ``params/agent.yaml``) with an event file, and prints ``train``'s run-directory
  line; the split writes its ``--out`` file: with a ``lineage``, as the real split
  does (``WarmStart.save``, the lineage YAML beside it), else bytes alone.
  """
  launches: list[Launch] = []

  def popen(args: list[str], *, env: dict[str, str], **kwargs: Any) -> FakeStep:
    del kwargs
    launch = Launch(list(args), dict(env))
    launches.append(launch)
    step = step_name(launch.args)
    if events is not None:
      events.append(step)
    if step == "split":
      out = Path(launch.args[launch.args.index("--out") + 1])
      if lineage is None:
        out.write_bytes(b"warm start")
      else:
        save_warm_start(out, lineage)
      return FakeStep(["split: ok\n"], (codes or {}).get(step, 0))
    run = root / "runs" / step
    if step == "stage1":
      make_stage1_run(run)
    else:
      run.mkdir(parents=True)
      (run / "events.out.tfevents.2000.host.2.0").write_bytes(b"")
    lines: Output = [
      "[INFO] Training with: device=cuda:0, seed=42, rank=0\n",
      f"[INFO] Logging experiment in directory: {run}\n",
      f"{step} iteration\n",
    ]
    return FakeStep(lines, (codes or {}).get(step, 0))

  monkeypatch.setattr(subprocess, "Popen", popen)
  return launches


def fake_smi(
  monkeypatch: pytest.MonkeyPatch,
  outputs: list[str | BaseException | subprocess.CompletedProcess[str]],
  events: list[str] | None = None,
) -> list[list[str]]:
  """Replace ``subprocess.run`` with an ``nvidia-smi`` that answers ``outputs`` in
  turn (repeating the last); return its calls."""
  calls: list[list[str]] = []

  def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    del kwargs
    calls.append(list(args))
    if events is not None:
      events.append("smi")
    output = outputs.pop(0) if len(outputs) > 1 else outputs[0]
    if isinstance(output, BaseException):
      raise output
    if isinstance(output, str):
      return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")
    return output

  monkeypatch.setattr(subprocess, "run", run)
  return calls


def refuse_to_launch(monkeypatch: pytest.MonkeyPatch) -> None:
  def refuse(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(f"launched {args} {kwargs}")

  monkeypatch.setattr(subprocess, "Popen", refuse)
  monkeypatch.setattr(subprocess, "run", refuse)


##
# The real driver, on a dummy stage 1.
##

DRIVER_SCRIPT = """\
import sys

scripts, root, step, repeat_s = sys.argv[1:]
sys.path.insert(0, scripts)  # two_stage as a plain module, as its file form runs.
import two_stage

two_stage.SIGNAL_REPEAT_S = float(repeat_s)
two_stage.query_gpu_processes = lambda: []  # Never ask about the real GPU.
cfg = two_stage.TwoStageCfg(name="p", library_dir=root, log_root=f"{root}/logs")
sys.exit(two_stage.run_pipeline(cfg, python=step))
"""
STEP_SCRIPT = """\
import os
import signal
import sys
import time


def report(signum, frame):
  print("step: got", signal.Signals(signum).name, flush=True)


if os.environ.get("STEP_IGNORES_SIGNALS"):
  signal.signal(signal.SIGINT, report)
  signal.signal(signal.SIGTERM, report)
print("step: ready", os.getpid(), flush=True)
try:
  while True:
    time.sleep(0.05)
except KeyboardInterrupt:
  print("step: got SIGINT, cleaning up", flush=True)
  time.sleep(0.5)
  print("step: cleaned up", flush=True)
  sys.exit(130)
"""


class DriverRun:
  """``run_pipeline`` in a process of its own, with no terminal (as under a service
  manager or ``nohup``), whose stage 1 is ``STEP_SCRIPT``: the driver takes it for
  the Python interpreter and launches it in place of ``train``."""

  def __init__(
    self, root: Path, repeat_s: float = 1.0, ignores_signals: bool = False
  ) -> None:
    step = root / "python"
    step.write_text(f"#!{sys.executable}\n{STEP_SCRIPT}")
    step.chmod(0o755)
    driver = root / "driver.py"
    driver.write_text(DRIVER_SCRIPT)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    if ignores_signals:
      env["STEP_IGNORES_SIGNALS"] = "1"
    scripts = Path(two_stage.__file__).parent
    self.log = root / "logs" / "pipelines" / "p.log"
    self.step_pid: int | None = None
    self.lines: list[str] = []
    self._lines: queue.Queue[str | None] = queue.Queue()
    self.process = subprocess.Popen(
      [sys.executable, str(driver), str(scripts), str(root), str(step), str(repeat_s)],
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      env=env,
      start_new_session=True,
    )
    threading.Thread(target=self._read, daemon=True).start()

  def _read(self) -> None:
    assert self.process.stdout is not None
    for line in self.process.stdout:
      self._lines.put(line)
    self._lines.put(None)

  def expect(self, text: str, timeout: float = 15.0) -> str:
    """The next output line that contains ``text``."""
    deadline = time.monotonic() + timeout
    while True:
      try:
        line = self._lines.get(timeout=max(deadline - time.monotonic(), 0.0))
      except queue.Empty:
        raise AssertionError(f"no {text!r} in {timeout} s: {self.lines}") from None
      if line is None:
        raise AssertionError(f"the output ended without {text!r}: {self.lines}")
      self.lines.append(line)
      if text in line:
        return line

  def ready(self) -> None:
    """Wait until stage 1 runs, the driver's signal handlers with it."""
    self.step_pid = int(self.expect("step: ready").split()[-1])

  def signal(self, signum: int, group: bool = False) -> None:
    """Signal the driver alone, as ``kill`` or a service manager would, or its
    whole process group, as a terminal's Ctrl-C would."""
    if group:
      os.killpg(self.process.pid, signum)
    else:
      os.kill(self.process.pid, signum)

  def finish(self, timeout: float = 15.0) -> int:
    """The driver's exit status, once it has exited and all its output is read."""
    code = self.process.wait(timeout=timeout)
    while (line := self._lines.get(timeout=timeout)) is not None:
      self.lines.append(line)
    return code

  def kill(self) -> None:
    """Leave nothing running, whatever the test did: the step's process group is
    its own, unless the driver under test failed to give it one."""
    for group in (self.step_pid, self.process.pid):
      if group is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
          os.killpg(group, signal.SIGKILL)
    self.process.wait(timeout=15.0)


@pytest.fixture
def start_driver(tmp_path: Path) -> Iterator[Callable[..., DriverRun]]:
  runs: list[DriverRun] = []

  def start(**kwargs: Any) -> DriverRun:
    runs.append(DriverRun(tmp_path, **kwargs))
    return runs[-1]

  yield start
  for run in runs:
    run.kill()


##
# The driver itself.
##


def test_driver_imports_only_the_standard_library_and_tyro() -> None:
  tree = ast.parse(Path(two_stage.__file__).read_text())
  roots: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      roots.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      assert node.level == 0, "a relative import would load the mjlab package"
      roots.add((node.module or "").split(".")[0])
  assert roots - set(sys.stdlib_module_names) - {"__future__"} == {"tyro"}


FILE_FORM_PROBE = """\
import os
import runpy
import sys

path = sys.argv[1]
sys.argv = [path, "--help"]
sys.path[0] = os.path.dirname(path)  # What ``python <path>`` starts with.
try:
  runpy.run_path(path, run_name="__main__")
except SystemExit as e:
  print("exit status:", e.code)
loaded = {name.partition(".")[0] for name in sys.modules}
print("heavy imports:", sorted(loaded & {"mjlab", "torch", "warp"}))
"""


def test_the_documented_launch_runs_the_file_without_mjlab_torch_or_warp() -> None:
  """The driver is launched as a file: with ``-m`` Python would import the mjlab
  package first, and with it torch, warp and every task, about 1 GB that a process
  waiting on its steps for days would hold."""
  path = Path(two_stage.__file__)
  launch = (
    f"uv run --no-sync python {path.relative_to(Path(mjlab.__file__).parents[2])}"
  )
  for module in (two_stage, split_scripts, split_config):
    doc = " ".join((module.__doc__ or "").split())
    assert "-m mjlab.tasks.osl_split.scripts.two_stage" not in doc
    assert launch in doc, module.__name__
  probe = subprocess.run(
    [sys.executable, "-c", FILE_FORM_PROBE, str(path)],
    capture_output=True,
    text=True,
    timeout=120,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    check=False,
  )
  assert "--stage1.run-dir" in probe.stdout, probe.stderr
  assert "exit status: 0" in probe.stdout
  assert "heavy imports: []" in probe.stdout


def test_literals_mirror_mjlab() -> None:
  assert set(get_args(two_stage.Model)) == set(get_args(osl_constants.Model))
  assert set(get_args(two_stage.Socket)) == set(get_args(osl_constants.Socket))
  assert set(get_args(two_stage.Reward)) == set(get_args(mimic_env_cfgs.Reward))
  assert two_stage.MODEL_TAGS == osl_constants.MODEL_TAGS
  assert set(get_args(two_stage.TouchSites)) == set(get_args(osl_constants.TouchSites))
  assert two_stage.LEGACY_TOUCH_SITES == osl_constants.LEGACY_TOUCH_SITES
  assert two_stage.TOUCH_SITES_KEY == mimic_touch_sites.ENV_FIELD
  assert two_stage.TOUCH_SITES_FLAG == mimic_touch_sites.CLI_FLAG


@pytest.mark.parametrize(("model", "socket", "reward"), COMBOS)
def test_stage1_task_ids_are_the_registered_unicycle_tasks(
  model: two_stage.Model, socket: two_stage.Socket, reward: two_stage.Reward
) -> None:
  rl_cfg = load_rl_cfg(two_stage.stage1_task_id(model, socket, reward))
  assert isinstance(rl_cfg, GaitLibraryRunnerCfg)
  validation = rl_cfg.validation
  assert (validation.model, validation.socket, validation.reward) == (
    model,
    socket,
    reward,
  )
  assert validation.variant == "unicycle"


def test_stage2_task_ids() -> None:
  assert two_stage.stage1_task_id("msk", "compliant", "clf") == (
    "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
  )
  assert two_stage.stage2_task_id("msk", "compliant", "clf") == (
    "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
  )
  assert two_stage.stage2_task_id("robot", "rigid", "mimic") == (
    "Mjlab-Split-Unicycle-OSL-Fullbody-ROBOT"
  )
  for model, socket, reward in COMBOS:
    stage1 = two_stage.stage1_task_id(model, socket, reward)
    stage2 = two_stage.stage2_task_id(model, socket, reward)
    assert stage2.removeprefix(two_stage.STAGE2_TASK) == stage1.removeprefix(
      two_stage.STAGE1_TASK
    )


@pytest.mark.skipif(
  not any(task.startswith(two_stage.STAGE2_TASK) for task in list_tasks()),
  reason="osl_split.config registers no stage-2 task yet",
)
@pytest.mark.parametrize(("model", "socket", "reward"), COMBOS)
def test_stage2_task_ids_are_registered(
  model: two_stage.Model, socket: two_stage.Socket, reward: two_stage.Reward
) -> None:
  rl_cfg = load_rl_cfg(two_stage.stage2_task_id(model, socket, reward))
  assert isinstance(rl_cfg, SplitRunnerCfg)
  validation = rl_cfg.validation
  assert (validation.model, validation.socket, validation.reward) == (
    model,
    socket,
    reward,
  )


def test_commands(tmp_path: Path, library: Path) -> None:
  cfg = make_cfg(
    tmp_path,
    stage1=Stage1Cfg(iterations=12, extra_args="--agent.seed 3 --video True"),
    stage2=Stage2Cfg(
      iterations=7, human_freeze_iterations=2, extra_args="--agent.logger tensorboard"
    ),
  )
  logs = str(tmp_path / "logs")
  assert two_stage.stage1_command(cfg, "py") == [
    "py",
    "-m",
    "mjlab.scripts.train",
    "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-MSK-CLF-Compliant",
    "--env.commands.motion.library-dir",
    str(library),
    "--agent.run-name",
    "p_stage1",
    "--agent.max-iterations",
    "12",
    "--agent.algorithm.lr-schedule-updates",
    "12",
    "--log-root",
    logs,
    "--agent.seed",
    "3",
    "--video",
    "True",
  ]
  run, out = tmp_path / "run", tmp_path / "ws.pt"
  split = ["py", "-m", "mjlab.tasks.osl_split.scripts.split", str(run)]
  assert two_stage.split_command(run, out, "", "py") == split + ["--out", str(out)]
  assert two_stage.split_command(run, out, "model_610.pt", "py") == split + [
    "--checkpoint",
    "model_610.pt",
    "--out",
    str(out),
  ]
  stage2 = [
    "py",
    "-m",
    "mjlab.scripts.train",
    "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-Compliant",
    "--env.commands.motion.library-dir",
    str(library),
    "--agent.run-name",
    "p_stage2",
    "--agent.warm-start.path",
    str(out),
    "--agent.max-iterations",
    "7",
    "--agent.freeze.human-actor-iterations",
    "2",
    "--log-root",
    logs,
    "--agent.logger",
    "tensorboard",
  ]
  assert two_stage.stage2_command(cfg, out, "mimic", "py") == stage2
  # Stage 1's touch sites, when known, go before the extra arguments, which come
  # last (and so win).
  for touch in ("original", "heel_extended"):
    assert two_stage.stage2_command(cfg, out, "mimic", "py", touch_sites=touch) == [
      *stage2[:-2],
      "--env.osl-touch-sites",
      touch,
      "--agent.logger",
      "tensorboard",
    ]
  assert two_stage.stage2_reward(cfg, "mimic") == "mimic"
  switched = replace(cfg, stage2=Stage2Cfg(reward="clf"))
  assert two_stage.stage2_reward(switched, "mimic") == "clf"


def test_train_cli_accepts_the_commands(tmp_path: Path, library: Path) -> None:
  """The flags the driver writes are the ones ``train``'s tyro CLI parses."""
  cfg = make_cfg(
    tmp_path,
    stage1=Stage1Cfg(
      iterations=12, extra_args="--agent.seed 3 --agent.max-iterations 20"
    ),
  )

  def parse(command: list[str], default: TrainConfig) -> TrainConfig:
    assert command[1:3] == ["-m", two_stage.TRAIN_MODULE]
    # train.py parses the task id itself, then the rest against its TrainConfig.
    return tyro.cli(
      TrainConfig, args=command[4:], default=default, config=mjlab.TYRO_FLAGS
    )

  task1 = two_stage.stage1_task_id("msk", "compliant", "clf")
  stage1 = parse(two_stage.stage1_command(cfg), TrainConfig.from_task(task1))
  motion = stage1.env.commands["motion"]
  assert isinstance(motion, GaitLibraryCommandCfg)
  assert motion.library_dir == str(library)
  agent1 = stage1.agent
  assert isinstance(agent1, GaitLibraryRunnerCfg)
  assert agent1.run_name == "p_stage1"
  assert agent1.seed == 3
  assert agent1.max_iterations == 20  # Extra arguments come last and win.
  assert isinstance(agent1.algorithm, MuscleMimicPpoCfg)
  assert agent1.algorithm.lr_schedule_updates == 12
  assert stage1.log_root == str(tmp_path / "logs")

  # The stage-2 env is the unicycle env's, so its motion flag is the same; the
  # runner settings are the split's.
  default2 = TrainConfig(
    env=osl_gait_library_env_cfg("msk", "compliant", "clf", "train", "unicycle"),
    agent=osl_split_runner_cfg("msk", "compliant", "clf", "split"),
  )
  out = tmp_path / "ws.pt"
  stage2 = parse(two_stage.stage2_command(cfg, out, "clf"), default2)
  agent = stage2.agent
  assert isinstance(agent, SplitRunnerCfg)
  assert agent.warm_start.path == str(out)
  assert agent.max_iterations == 1000
  assert agent.freeze.human_actor_iterations == 200
  assert agent.run_name == "p_stage2"
  assert stage2.log_root == str(tmp_path / "logs")
  assert isinstance(stage2.env, mimic_touch_sites.OslEnvCfg)
  assert stage2.env.osl_touch_sites == "heel_extended"  # Not forwarded: the default.

  # The touch-site flag the driver forwards reaches the env and its robot entity.
  forwarded = two_stage.stage2_command(cfg, out, "clf", touch_sites="original")
  env = parse(forwarded, default2).env
  assert isinstance(env, mimic_touch_sites.OslEnvCfg)
  assert env.osl_touch_sites == "original"
  robot = env.scene.entities["robot"]
  assert isinstance(robot, osl_constants.OslEntityCfg)
  assert robot.touch_sites == "original"


def test_cli_flags() -> None:
  cfg = tyro.cli(
    TwoStageCfg,
    args=[
      "--name",
      "p",
      "--model",
      "robot",
      "--socket",
      "rigid",
      "--stage1.run-dir",
      "/r",
      "--stage1.checkpoint",
      "model_5.pt",
      "--stage1.extra-args",
      "--video",
      "--stage2.reward",
      "mimic",
      "--stage2.human-freeze-iterations",
      "50",
      "--stage2.skip",
      "--curves",
      "none",
      "--wait-for-gpu",
      "--gpu-poll-s",
      "5",
    ],
  )
  assert cfg == TwoStageCfg(
    name="p",
    model="robot",
    socket="rigid",
    stage1=Stage1Cfg(run_dir="/r", checkpoint="model_5.pt", extra_args="--video"),
    stage2=Stage2Cfg(reward="mimic", human_freeze_iterations=50, skip=True),
    curves="none",
    wait_for_gpu=True,
    gpu_poll_s=5.0,
  )


##
# Run directories, the GPU guard, W&B and TensorBoard.
##


def test_parse_run_dir_reads_trains_announcement() -> None:
  source = Path(mjlab.scripts.train.__file__).read_text()
  assert 'print(f"[INFO] Logging experiment in directory: {log_dir}")' in source
  parse = two_stage.parse_run_dir
  assert parse("[INFO] Logging experiment in directory: /a/b c/run\n") == Path(
    "/a/b c/run"
  )
  assert parse("[INFO] Logging experiment in directory: /a/run \r\n") == Path("/a/run")
  assert parse("[INFO] Training with: device=cuda:0, seed=42, rank=0\n") is None
  assert parse("Logging experiment in directory: /a\n") is None


def test_warm_start_touch_sites_reads_the_splits_lineage(tmp_path: Path) -> None:
  """Stage 1's OSL touch sites, from the lineage YAML ``WarmStart.save`` writes
  beside the warm start, read without a YAML parser: what mjlab reads too."""
  for value in ("original", "heel_extended", None):
    path = tmp_path / str(value) / "ws.pt"
    save_warm_start(path, split_lineage(value))
    read = two_stage.warm_start_touch_sites(path)
    # None: a file converted before the setting, which trained with the original.
    assert read == (value or "original")
    assert read == lineage_touch_sites(WarmStart.load(path).lineage)
  # No lineage beside the file: nothing to forward.
  bare = tmp_path / "bare.pt"
  bare.write_bytes(b"warm start")
  assert two_stage.warm_start_touch_sites(bare) is None
  assert two_stage.warm_start_touch_sites(tmp_path / "missing.pt") is None
  # An unknown version is an error, not a guess.
  unknown = tmp_path / "unknown" / "ws.pt"
  save_warm_start(unknown, split_lineage("toes"))
  with pytest.raises(two_stage.PipelineError, match="unknown osl_touch_sites: 'toes'"):
    two_stage.warm_start_touch_sites(unknown)
  # Only the top-level key counts.
  lineage_yaml_path(bare).write_text("stage1_env:\n  osl_touch_sites: heel_extended\n")
  assert two_stage.warm_start_touch_sites(bare) == "original"


def test_parse_gpu_processes() -> None:
  assert two_stage.parse_gpu_processes(SMI_BUSY) == [
    GpuProcess(498800, "/home/u/mjlab/.venv/bin/python3", "17952 MiB"),
    GpuProcess(4242, "/opt/odd, name/python", "[N/A]"),
  ]
  assert two_stage.parse_gpu_processes("") == []
  assert two_stage.parse_gpu_processes("No running processes found\n") == []
  # What cannot be parsed still counts as a GPU user.
  assert two_stage.parse_gpu_processes("garbage\n") == [GpuProcess(None, "garbage", "")]


def test_gpu_guard(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  log = two_stage.PipelineLog(None)
  calls = fake_smi(monkeypatch, [SMI_BUSY])
  with pytest.raises(two_stage.GpuBusyError, match="pid 498800"):
    two_stage.ensure_gpu_free(log, wait=False, poll_s=1.0)
  assert calls == [list(two_stage.NVIDIA_SMI_QUERY)]
  # The driver and its own steps never count.
  two_stage.ensure_gpu_free(log, wait=False, poll_s=1.0, ignore_pids={498800, 4242})

  sleeps: list[float] = []
  monkeypatch.setattr(two_stage.time, "sleep", sleeps.append)
  fake_smi(monkeypatch, [SMI_BUSY, SMI_BUSY, ""])
  two_stage.ensure_gpu_free(log, wait=True, poll_s=5.0)
  assert sleeps == [5.0, 5.0]
  out = capsys.readouterr().out
  assert out.count("Waiting for the GPU") == 1
  assert "The GPU is free." in out

  # No nvidia-smi: warn and go on. A failing one is an error.
  fake_smi(monkeypatch, [FileNotFoundError("nvidia-smi")])
  two_stage.ensure_gpu_free(log, wait=False, poll_s=1.0)
  assert "nvidia-smi not found" in capsys.readouterr().out
  failed = subprocess.CompletedProcess([], 9, stdout="", stderr="no driver")
  fake_smi(monkeypatch, [failed])
  with pytest.raises(two_stage.PipelineError, match="no driver"):
    two_stage.ensure_gpu_free(log, wait=True, poll_s=1.0)


def test_wandb_takes_the_pinned_ids_from_the_environment() -> None:
  """The installed wandb and rsl_rl honour ``WANDB_RUN_ID``/``WANDB_RUN_GROUP``."""
  # rsl_rl's writer passes neither ``id`` nor ``group`` to wandb.init.
  tree = ast.parse(textwrap.dedent(inspect.getsource(WandbLogWriter.__init__)))
  inits = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "wandb.init"
  ]
  assert len(inits) == 1
  assert {keyword.arg for keyword in inits[0].keywords} == {
    "project",
    "entity",
    "name",
    "config",
  }
  # wandb.setup() reads the environment into the global settings ...
  global_settings = Settings()
  global_settings.update_from_env_vars(two_stage.wandb_env("p", "abc12345"))
  # ... and wandb.init(project=, name=) starts from a copy, overriding only those.
  init_settings = Settings()
  init_settings.project = "mjlab"
  init_settings.run_name = "2026-11-07_00-00-00_p_stage2"
  run_settings = global_settings.model_copy()
  run_settings.update_from_settings(init_settings)
  assert run_settings.run_id == "abc12345"
  assert run_settings.run_group == "p"
  assert run_settings.run_name == "2026-11-07_00-00-00_p_stage2"
  assert re.fullmatch(r"[a-z0-9]{8}", two_stage.new_run_id())


def test_combined_tensorboard_dir_links_exactly_the_runs(tmp_path: Path) -> None:
  run1, run2, old = (tmp_path / name for name in ("run1", "run2", "old"))
  for run in (run1, run2, old):
    run.mkdir()
  event1 = run1 / "events.out.tfevents.1000.host.1.0"
  event1.write_bytes(b"")
  (run1 / "model_0.pt").write_bytes(b"")
  stale = old / "events.out.tfevents.1500.host.3.0"
  stale.write_bytes(b"")
  tb = tmp_path / "p_tb"
  tb.mkdir()
  (tb / stale.name).symlink_to(stale)  # An earlier attempt's stage 2.
  (tb / "notes.txt").write_text("not a link: kept")

  # Stage 2's event file appears only once its env is built.
  assert sync_combined_tb(tb, [run1, run2]) == [event1.resolve()]
  event2 = run2 / "events.out.tfevents.2000.host.2.0"
  event2.write_bytes(b"")
  for _ in range(2):  # Idempotent; a run dir that does not exist yet is empty.
    linked = sync_combined_tb(tb, [run1, run2, tmp_path / "missing"])
    assert linked == [event1.resolve(), event2.resolve()]
  assert sorted(path.name for path in tb.iterdir()) == sorted(
    [event1.name, event2.name, "notes.txt"]
  )
  assert (tb / event2.name).resolve() == event2.resolve()


def test_tensorboard_reads_the_combined_dir_as_one_curve(tmp_path: Path) -> None:
  for run, steps, stamp in (("run1", range(0, 3), 1), ("run2", range(3, 6), 2)):
    writer = SummaryWriter(log_dir=str(tmp_path / run))
    for step in steps:
      writer.add_scalar("Train/mean_reward", float(step), step)
    writer.close()
    # Fixed creation stamps: the writers' own may tie within one second.
    (event,) = (tmp_path / run).glob(two_stage.EVENT_FILE_GLOB)
    event.rename(event.with_name(f"events.out.tfevents.{stamp:010d}.host.1.0"))
  tb = tmp_path / "p_tb"
  sync_combined_tb(tb, [tmp_path / "run1", tmp_path / "run2"])
  accumulator = EventAccumulator(str(tb), size_guidance={"scalars": 0})
  accumulator.Reload()
  scalars = accumulator.Scalars("Train/mean_reward")
  assert [(event.step, event.value) for event in scalars] == [
    (step, float(step)) for step in range(6)
  ]


##
# Whole pipelines, with fake steps.
##


def test_dry_run_prints_every_command_and_runs_nothing(
  tmp_path: Path,
  library: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  refuse_to_launch(monkeypatch)
  logs = tmp_path / "logs"
  argv = ["--name", "p", "--library-dir", str(library), "--log-root", str(logs)]
  assert two_stage.main([*argv, "--dry-run", "--stage2.reward", "mimic"]) == 0
  out = capsys.readouterr().out
  cfg = make_cfg(tmp_path)
  paths = PipelinePaths.of(cfg)
  assert shlex.join(two_stage.stage1_command(cfg)) in out
  split = two_stage.split_command("<stage-1 run dir>", paths.warm_start)
  assert f"CUDA_VISIBLE_DEVICES='' {shlex.join(split)}" in out
  assert shlex.join(two_stage.stage2_command(cfg, paths.warm_start, "mimic")) in out
  ids = re.findall(r"WANDB_RUN_GROUP=p WANDB_RUN_ID=([a-z0-9]{8}) ", out)
  assert len(ids) == 2 and ids[0] != ids[1]
  assert "Reward switch: stage 1 clf, stage 2 mimic" in out
  assert str(paths.tensorboard) in out
  assert not logs.exists()


def test_pipeline_runs_the_steps_in_order(
  tmp_path: Path, library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  events: list[str] = []
  fake_smi(monkeypatch, [""], events)
  launches = fake_steps(monkeypatch, tmp_path, events=events)
  cfg = make_cfg(tmp_path)
  assert two_stage.run_pipeline(cfg, python="py") == 0

  # The GPU is checked before each GPU stage and before the split that feeds one.
  assert events == ["smi", "stage1", "smi", "split", "smi", "stage2"]
  stage1, split, stage2 = launches
  run1, run2 = tmp_path / "runs" / "stage1", tmp_path / "runs" / "stage2"
  paths = PipelinePaths.of(cfg)
  assert stage1.args == two_stage.stage1_command(cfg, "py")
  assert split.args == two_stage.split_command(run1, paths.warm_start, "", "py")
  assert stage2.args == two_stage.stage2_command(cfg, paths.warm_start, "clf", "py")

  # One W&B group, two fresh pinned ids; the split never sees the GPU.
  assert stage1.env["WANDB_RUN_GROUP"] == stage2.env["WANDB_RUN_GROUP"] == "p"
  assert re.fullmatch(r"[a-z0-9]{8}", stage1.env["WANDB_RUN_ID"])
  assert stage1.env["WANDB_RUN_ID"] != stage2.env["WANDB_RUN_ID"]
  assert split.env["CUDA_VISIBLE_DEVICES"] == ""
  assert all(launch.env["PYTHONUNBUFFERED"] == "1" for launch in launches)

  log = paths.log.read_text()
  for line in ("stage1 iteration", "split: ok", "stage2 iteration"):
    assert line in log
  assert "stage 2 starts at iteration 12" in log
  assert stage1.env["WANDB_RUN_ID"] in log and stage2.env["WANDB_RUN_ID"] in log
  linked = sorted(path.resolve() for path in paths.tensorboard.iterdir())
  assert linked == sorted(
    (run / name).resolve()
    for run, name in (
      (run1, "events.out.tfevents.1000.host.1.0"),
      (run2, "events.out.tfevents.2000.host.2.0"),
    )
  )


@pytest.mark.parametrize("touch_sites", ["original", "heel_extended"])
def test_stage2_builds_the_touch_sites_stage1_trained_with(
  tmp_path: Path, library: Path, monkeypatch: pytest.MonkeyPatch, touch_sites: str
) -> None:
  """The driver reads stage 1's OSL touch sites from the lineage the split wrote
  and passes them to stage 2, whose runner refuses a switch, before the extra
  arguments."""
  fake_smi(monkeypatch, [""])
  launches = fake_steps(monkeypatch, tmp_path, lineage=split_lineage(touch_sites))
  cfg = make_cfg(tmp_path, stage2=Stage2Cfg(extra_args="--agent.logger tensorboard"))
  assert two_stage.run_pipeline(cfg, python="py") == 0
  assert [step_name(launch.args) for launch in launches] == [
    "stage1",
    "split",
    "stage2",
  ]
  paths = PipelinePaths.of(cfg)
  assert launches[-1].args == two_stage.stage2_command(
    cfg, paths.warm_start, "clf", "py", touch_sites=touch_sites
  )
  assert launches[-1].args[-4:] == [
    two_stage.TOUCH_SITES_FLAG,
    touch_sites,
    "--agent.logger",
    "tensorboard",
  ]


def test_touch_sites_in_printed_stage2_commands(
  tmp_path: Path,
  library: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  """A skipped stage 2's command for later carries stage 1's touch sites. A dry
  run splits nothing, so it never reads a lineage left at the warm-start path (an
  earlier run's), and its stage-2 command carries none."""
  run1 = make_stage1_run(tmp_path / "given")
  fake_smi(monkeypatch, [""])
  launches = fake_steps(monkeypatch, tmp_path, lineage=split_lineage("original"))
  cfg = make_cfg(
    tmp_path, stage1=Stage1Cfg(run_dir=str(run1)), stage2=Stage2Cfg(skip=True)
  )
  assert two_stage.run_pipeline(cfg, python="py") == 0
  assert [step_name(launch.args) for launch in launches] == ["split"]
  paths = PipelinePaths.of(cfg)
  later = two_stage.stage2_command(
    cfg, paths.warm_start, "clf", "py", touch_sites="original"
  )
  assert shlex.join(later) in capsys.readouterr().out

  assert two_stage.warm_start_touch_sites(paths.warm_start) == "original"
  dry = replace(cfg, stage2=Stage2Cfg(), dry_run=True)
  assert two_stage.run_pipeline(dry, python="py") == 0
  assert [step_name(launch.args) for launch in launches] == ["split"]
  out = capsys.readouterr().out
  assert shlex.join(two_stage.stage2_command(dry, paths.warm_start, "clf", "py")) in out
  assert two_stage.TOUCH_SITES_FLAG not in out


def test_an_unknown_touch_site_version_stops_the_pipeline(
  tmp_path: Path,
  library: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  fake_smi(monkeypatch, [""])
  launches = fake_steps(monkeypatch, tmp_path, lineage=split_lineage("toes"))
  assert two_stage.run_pipeline(make_cfg(tmp_path), python="py") == 1
  assert [step_name(launch.args) for launch in launches] == ["stage1", "split"]
  assert "unknown osl_touch_sites: 'toes'" in capsys.readouterr().err


def test_an_existing_stage1_run_keeps_its_reward_and_socket(
  tmp_path: Path,
  library: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  # Unfinished (3051 iterations planned), but a named checkpoint is split anyway.
  run1 = make_stage1_run(
    tmp_path / "given", (0, 610, 1220), reward="mimic", max_iterations=3051
  )
  fake_smi(monkeypatch, [""])
  launches = fake_steps(monkeypatch, tmp_path)
  cfg = make_cfg(
    tmp_path, stage1=Stage1Cfg(run_dir=str(run1), checkpoint="model_610.pt")
  )
  assert two_stage.run_pipeline(cfg, python="py") == 0
  assert [step_name(launch.args) for launch in launches] == ["split", "stage2"]
  paths = PipelinePaths.of(cfg)
  assert launches[0].args == two_stage.split_command(
    run1, paths.warm_start, "model_610.pt", "py"
  )
  # The run's reward (mimic), not --stage1.reward's default (clf).
  assert launches[1].args[3] == "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-Compliant"
  out = capsys.readouterr().out
  assert "stage 2 starts at iteration 611" in out
  assert "interleave" in out  # Stage 1 ran on to iteration 1220.

  # A model or socket the run was not trained with is refused before any step.
  launches.clear()
  assert two_stage.run_pipeline(replace(cfg, socket="rigid")) == 1
  assert launches == []
  assert "trained with socket 'compliant'" in capsys.readouterr().err


@pytest.mark.parametrize(
  ("checkpoints", "settings", "last"),
  [
    # A fresh run plans iterations 0 to max_iterations - 1.
    ((0, 610, 1220), {"max_iterations": 3051}, 3050),
    # A resumed run counts max_iterations from the checkpoint it loaded: the named
    # one, ...
    (
      (1220, 1830),
      {"max_iterations": 1831, "resume": "true", "load_checkpoint": "model_1220.pt"},
      3050,
    ),
    # ... else its first, which it saved again on resuming, ...
    ((1220, 1830, 2440), {"max_iterations": 1831, "resume": "true"}, 3050),
    # ... unless it resumed off the save interval (from a finished run's last
    # checkpoint, 3050 here): a last checkpoint off the interval is the end.
    (
      (3500, 4000, 4049),
      {"max_iterations": 1000, "resume": "true", "save_interval": 500},
      4049,
    ),
  ],
)
def test_planned_last_iteration(
  tmp_path: Path, checkpoints: tuple[int, ...], settings: dict[str, Any], last: int
) -> None:
  run = make_stage1_run(tmp_path / "run", checkpoints, **settings)
  assert two_stage.planned_last_iteration(run) == last


def test_an_unfinished_stage1_run_is_not_split(
  tmp_path: Path,
  library: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  """A stage-1 run short of its last iteration crashed, was stopped or is still
  training: it is refused before the split, unless ``--stage1.checkpoint`` names
  the checkpoint, and stage 2 is never trained from it unasked. With
  ``--stage2.skip``, and in a dry run, it is a loud warning instead."""
  run1 = make_stage1_run(tmp_path / "given", (0, 610, 1220), max_iterations=3051)
  fake_smi(monkeypatch, [""])
  launches = fake_steps(monkeypatch, tmp_path)
  cfg = make_cfg(tmp_path, stage1=Stage1Cfg(run_dir=str(run1)))
  stopped = f"{run1} stopped at iteration 1220 of 3050"
  assert two_stage.run_pipeline(cfg) == 1
  assert launches == []
  err = capsys.readouterr().err
  assert stopped in err and "--stage1.checkpoint model_1220.pt" in err

  assert two_stage.run_pipeline(replace(cfg, dry_run=True)) == 0
  assert launches == []
  assert f"WARNING: {stopped}" in capsys.readouterr().err

  assert two_stage.run_pipeline(replace(cfg, stage2=Stage2Cfg(skip=True))) == 0
  assert [step_name(launch.args) for launch in launches] == ["split"]
  assert f"WARNING: {stopped}" in capsys.readouterr().err

  # Without max_iterations the driver cannot tell, and says so.
  agent_yaml = run1 / "params" / "agent.yaml"
  agent_yaml.write_text(agent_yaml.read_text().replace("max_iterations: 3051\n", ""))
  assert two_stage.run_pipeline(replace(cfg, dry_run=True)) == 0
  assert "cannot tell whether" in capsys.readouterr().out


def test_a_failed_step_or_a_busy_gpu_stops_the_pipeline(
  tmp_path: Path,
  library: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  fake_smi(monkeypatch, [""])
  launches = fake_steps(monkeypatch, tmp_path / "a", codes={"stage1": 3})
  assert two_stage.run_pipeline(make_cfg(tmp_path)) == 3
  assert [step_name(launch.args) for launch in launches] == ["stage1"]
  assert "Stage 1 failed (exit code 3)" in capsys.readouterr().err

  # The GPU is taken while stage 1 ends: stop before the split.
  fake_smi(monkeypatch, ["", SMI_BUSY])
  launches = fake_steps(monkeypatch, tmp_path / "b")
  assert two_stage.run_pipeline(make_cfg(tmp_path)) == 1
  assert [step_name(launch.args) for launch in launches] == ["stage1"]
  assert "--wait-for-gpu" in capsys.readouterr().err


def deliver(signum: int, clock: list[float], after_s: float) -> Callable[[], None]:
  """An action for a fake step's output: the driver receives ``signum``,
  ``after_s`` seconds (on ``clock``) after the previous one."""

  def action() -> None:
    clock[0] += after_s
    handler = signal.getsignal(signum)
    assert callable(handler) and handler is not signal.default_int_handler, (
      f"the driver leaves {signal.Signals(signum).name} to {handler}"
    )
    handler(signum, None)

  return action


def test_signals_to_the_driver_go_to_the_steps_process_group(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """While a step runs, the driver turns SIGINT, SIGTERM and SIGHUP alike into
  SIGINT, then SIGTERM, then SIGKILL to the step's own process group, takes a repeat
  within a second for the same request, drains the step to its end, puts its
  handlers back and stops the pipeline with 130."""
  clock = [1000.0]
  monkeypatch.setattr(two_stage.time, "monotonic", lambda: clock[0])
  sent: list[tuple[int, int]] = []
  monkeypatch.setattr(os, "killpg", lambda group, signum: sent.append((group, signum)))
  outputs: list[Output] = [
    [
      "one\n",
      deliver(signal.SIGHUP, clock, 0.0),
      deliver(signal.SIGINT, clock, 0.5),  # The same request (uv's duplicate).
      "two\n",
      deliver(signal.SIGTERM, clock, 1.0),
      deliver(signal.SIGINT, clock, 1.5),
      deliver(signal.SIGINT, clock, 1.5),
      "three\n",
    ],
    [deliver(signal.SIGINT, clock, 10.0)],
  ]
  steps: list[FakeStep] = []

  def popen(args: list[str], **kwargs: Any) -> FakeStep:
    del args
    assert kwargs.get("start_new_session"), "the step shares the driver's group"
    steps.append(FakeStep(outputs.pop(0), code=-9))
    return steps[-1]

  monkeypatch.setattr(subprocess, "Popen", popen)
  forwarded = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
  handlers = {signum: signal.getsignal(signum) for signum in forwarded}
  with two_stage.PipelineLog(tmp_path / "p.log") as log:
    with pytest.raises(KeyboardInterrupt):
      two_stage.run_child(["train"], {}, log)
  step = steps[0].pid
  assert sent == [
    (step, signal.SIGINT),
    (step, signal.SIGTERM),
    (step, signal.SIGKILL),
    (step, signal.SIGKILL),
  ]
  assert {signum: signal.getsignal(signum) for signum in forwarded} == handlers
  text = (tmp_path / "p.log").read_text()
  assert "\nthree\n" in text  # Drained to its end, after every signal.
  assert "SIGHUP: sent SIGINT to the step" in text
  assert "SIGTERM again: sent SIGTERM to the step." in text
  assert "The step has stopped (killed by SIGKILL)." in text

  # The pipeline then stops, with the shell's status for an interrupt.
  sent.clear()
  fake_smi(monkeypatch, [""])
  assert two_stage.run_pipeline(make_cfg(tmp_path, library_dir=str(tmp_path))) == 130
  assert sent == [(steps[1].pid, signal.SIGINT)]  # Stage 1 only.


@pytest.mark.parametrize(
  ("signum", "group"),
  [
    (signal.SIGINT, False),
    (signal.SIGTERM, False),
    (signal.SIGHUP, False),
    (signal.SIGINT, True),
  ],
)
def test_a_signal_stops_the_step_gently(
  start_driver: Callable[..., DriverRun], signum: int, group: bool
) -> None:
  """With no terminal, a signal to the driver alone (``kill``, a service manager, a
  closed terminal) or to its whole process group reaches the step as one SIGINT, so
  the step cleans up in peace and the driver exits with 130. The same signal again
  at once, as ``uv run`` forwards it to the driver, is the same request: it must
  not SIGTERM the step's cleanup."""
  run = start_driver()
  run.ready()
  run.signal(signum, group)
  run.expect("step: got SIGINT, cleaning up")
  run.signal(signum)
  assert run.finish() == 130
  assert "step: cleaned up\n" in run.lines
  log = run.log.read_text()
  assert f"{signal.Signals(signum).name}: sent SIGINT to the step" in log
  assert "sent SIGTERM" not in log
  assert "The step has stopped (exit code 130)." in log


def test_a_step_deaf_to_signals_is_killed_by_the_third(
  start_driver: Callable[..., DriverRun],
) -> None:
  """Signals that are not repeats escalate: SIGINT, SIGTERM, then SIGKILL."""
  run = start_driver(repeat_s=0.2, ignores_signals=True)
  run.ready()
  for expected in ("step: got SIGINT", "step: got SIGTERM"):
    run.signal(signal.SIGINT)
    run.expect(expected)
    time.sleep(0.4)  # Past the repeat window: a new request.
  run.signal(signal.SIGINT)
  assert run.finish() == 130
  assert "The step has stopped (killed by SIGKILL)." in run.log.read_text()


@pytest.mark.parametrize(
  ("changes", "message"),
  [
    ({"name": "a/b"}, "--name"),
    ({"library_dir": ""}, "--library-dir"),
    ({"stage2": Stage2Cfg(iterations=0)}, "--stage2.iterations"),
    ({"stage1": Stage1Cfg(extra_args="'unbalanced")}, "--stage1.extra-args"),
    ({"stage1": Stage1Cfg(run_dir="/nonexistent/run")}, "--stage1.run-dir"),
  ],
)
def test_bad_settings_are_refused_before_anything_runs(
  tmp_path: Path,
  library: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  changes: dict[str, Any],
  message: str,
) -> None:
  refuse_to_launch(monkeypatch)
  assert two_stage.run_pipeline(make_cfg(tmp_path, **changes)) == 1
  assert message in capsys.readouterr().err
  assert not (tmp_path / "logs").exists()
