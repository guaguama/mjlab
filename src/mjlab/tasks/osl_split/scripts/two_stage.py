"""Run stage 1, the split and stage 2 of the OSL plan, start to finish.

A thin driver: every step is its own process, launched as it would be by hand, so
the pipeline's runs are ordinary runs (the same ``train`` CLI, ``params/`` dumps,
W&B runs and ``play``) and any step can be redone on its own:

1. **Stage 1**: ``python -m mjlab.scripts.train
   Mjlab-Gait-Library-Unicycle-OSL-Fullbody<tags>``, unless ``--stage1.run-dir``
   names an existing run. ``--stage1.iterations`` sets both ``max_iterations`` and
   ``lr_schedule_updates``, so the rate's linear decay ends at the last iteration,
   at the task's ``lr_final_ratio``. The default (musclemimic's budget) is shorter
   than the MSK unicycle recipe's 4500 iterations, whose std cap and env stages
   count updates up to 4000.
2. **The split**: ``python -m mjlab.tasks.osl_split.scripts.split`` converts the
   run's highest-numbered checkpoint (or ``--stage1.checkpoint``) into
   ``<log-root>/pipelines/<name>_warm_start.pt``, on the CPU
   (``CUDA_VISIBLE_DEVICES=""``). A run that ended short of its last iteration is
   refused, unless ``--stage1.checkpoint`` is given (see **Resuming**).
3. **Stage 2**: ``python -m mjlab.scripts.train
   Mjlab-Split-Unicycle-OSL-Fullbody<tags> --agent.warm-start.path <that file>``.
   Its iteration numbers continue stage 1's.
4. **One TensorBoard view** (``--curves offset``): ``<log-root>/pipelines/<name>_tb/``
   holds symlinks to both runs' event files. TensorBoard reads the event files of
   one directory as one run, and stage 2's steps start after stage 1's, so every
   curve continues across the handover. The links follow stage 2 while it runs.

The tags are ``-ROBOT`` or ``-MSK``, then ``-CLF`` and ``-Compliant``. Model and socket are given once,
for both stages, because a warm-started human actor can change neither (its inputs
would change; the stage-2 runner refuses it too). Only the reward may switch
(``--stage2.reward``; empty keeps stage 1's), and the stage-2 critics then start
fresh. With ``--stage1.run-dir``, the run's model, socket and reward come from its
``params/agent.yaml``: a model or socket other than ``--model``/``--socket`` is
refused, and stage 2's default reward is the run's own, never ``--stage1.reward``.

Why processes rather than one Python process calling ``launch_training`` twice:
each stage then starts on a clean GPU (``ManagerBasedRlEnv.close`` releases only the
renderer and recorder, not the simulation), and wandb reads its environment once
per process, so stage 2's run id could not be pinned. Each child runs on
``sys.executable``, never ``uv run``, which can re-sync the virtual environment
under a running stage. This module imports only the standard library and tyro.

**GPU guard.** Before stage 1; before the split when stage 2 follows, so that a
stage 1 still running elsewhere is split only once it has finished; and again just
before stage 2, the driver asks ``nvidia-smi --query-compute-apps=pid,process_name,
used_memory --format=csv,noheader`` whether another process holds a GPU. If one
does, it exits with a message, or, with ``--wait-for-gpu``, checks again every
``--gpu-poll-s`` seconds. Any GPU counts, so a multi-GPU machine is guarded
conservatively. Without ``nvidia-smi`` it warns and goes on.

**W&B.** Both training runs join the W&B group ``<name>`` under fresh run ids,
pinned through ``WANDB_RUN_GROUP`` and ``WANDB_RUN_ID`` in each child's environment.
Verified on the installed wandb 0.30.0 and rsl_rl 5.5.1: rsl_rl's ``WandbLogWriter``
calls ``wandb.init(project=, entity=, name=, config=)``, with no ``id`` or
``group``; ``wandb.setup()`` loads each ``WANDB_<KEY>`` variable into its global
settings as ``<key>`` (``Settings.update_from_env_vars``, so ``run_id`` and
``run_group``); and ``wandb.init`` starts from a copy of those settings and
overrides only the arguments it was given (``_WandbInit.make_run_settings``). So the
pins reach the runs with no change to rsl_rl or mjlab. They must be in the
environment when the process starts: wandb ignores changes made after
``wandb.setup()``, which mjlab's ``add_wandb_tags`` calls early. The ids are
printed and logged; ``tests/test_osl_split_pipeline.py`` replays the settings merge
on the installed wandb.

**Output.** Everything goes to the console and to
``<log-root>/pipelines/<name>.log``. Each run directory is read from ``train``'s
``[INFO] Logging experiment in directory: <path>`` line. ``--dry-run`` prints every
command and runs nothing. Extra ``train`` arguments (``--stage1.extra-args``,
``--stage2.extra-args``) go last, so they win over the driver's own flags.

**Stopping.** Each step runs in a process group (and session) of its own, so a
signal reaches it only through the driver, whatever sent it: Ctrl-C, ``kill``, a
service manager, a closed terminal or ``uv``. While a step runs, the driver's first
SIGINT, SIGTERM or SIGHUP sends SIGINT to the step's group, which stops it as Ctrl-C
would, and the driver drains the step's output until it has exited; a second sends
SIGTERM, a third SIGKILL. A signal within a second of the last one counts once:
``uv run`` without a terminal forwards to the driver the SIGINT that its whole
process group has just received. The driver then exits with status 130. A step's
failure stops the pipeline with the step's exit status. If the driver itself is
killed (SIGKILL), the running step fails at its next output line (its pipe is gone).

**Resuming.** A stage that crashed is resumed with ``train`` itself, into a new run
directory (``--agent.resume True --agent.load-run <run> --agent.load-checkpoint
model_<N>.pt``). A resumed run counts ``--agent.max-iterations`` from ``N``, so pass
the iterations still to go. A resumed stage 2 keeps the curriculum its checkpoint
was trained with (freeze, warm-up, learning rates, blend and reward channels), and
warns about any flag that sets it otherwise, so its stage-2 flags need not be
repeated (``--agent.resume-curriculum config`` takes the command line's curriculum
instead, e.g. to extend the schedule with ``--agent.algorithm.schedule-iterations``);
its network flags (an OSL critic's ``--agent.osl-critic.hidden-dims``, say) must be
repeated. After a stage-1 resume, continue with ``--stage1.run-dir`` on the resumed
run. Before the split, the driver compares the run's highest-numbered checkpoint
with the last iteration its ``params/agent.yaml`` plans (``max_iterations``,
counted from ``N`` in a resumed run; see :func:`planned_last_iteration`). A
checkpoint short of it means stage 1 crashed, was stopped or is still training: the
driver refuses to split it unless ``--stage1.checkpoint`` names the checkpoint to
use, and with ``--stage2.skip`` it only warns.

Run it from the repository root, as a file, e.g.::

  uv run --no-sync python src/mjlab/tasks/osl_split/scripts/two_stage.py \\
    --name unicycle_split_v1 --library-dir <clipper unicycle library folder>

or start from an existing stage-1 run, waiting for the GPU if it is busy::

  uv run --no-sync python src/mjlab/tasks/osl_split/scripts/two_stage.py \\
    --name split_v1 --library-dir <folder> --wait-for-gpu \\
    --stage1.run-dir logs/rsl_rl/osl_fullbody_unicycle_msk_clf_compliant/<run>

As a file it imports only the standard library and tyro. Not as a module (``-m``):
Python would first import its parent packages, ``mjlab`` (warp) and ``mjlab.tasks``
(every task, and torch), about 1 GB of host memory that the driver would then hold
for the days it waits on its steps. (The steps themselves do run as modules.)
"""

from __future__ import annotations

import math
import os
import re
import secrets
import shlex
import signal
import string
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, TracebackType
from typing import IO, Any, Literal

import tyro

# Mirrors of ``osl_constants.Model``/``Socket`` and ``osl_mimic``'s ``Reward``
# (the tests check they match): importing those would load the mjlab package, and
# with it torch and warp, into a process that only waits on its children.
Model = Literal["robot", "msk"]
Socket = Literal["rigid", "compliant"]
Reward = Literal["mimic", "clf"]
Curves = Literal["offset", "none"]
MODEL_TAGS: dict[Model, str] = {"robot": "-ROBOT", "msk": "-MSK"}
"""Mirror of ``osl_constants.MODEL_TAGS``."""

STAGE1_TASK = "Mjlab-Gait-Library-Unicycle-OSL-Fullbody"
STAGE2_TASK = "Mjlab-Split-Unicycle-OSL-Fullbody"
TRAIN_MODULE = "mjlab.scripts.train"
SPLIT_MODULE = "mjlab.tasks.osl_split.scripts.split"
STAGE1_ITERATIONS = 3051
"""musclemimic's budget: 1e9 steps at 4096 envs x 80 steps (the MSK unicycle recipe
trains 4500 iterations)."""
NVIDIA_SMI_QUERY = (
  "nvidia-smi",
  "--query-compute-apps=pid,process_name,used_memory",
  "--format=csv,noheader",
)
RUN_DIR_LINE = re.compile(r"\[INFO\] Logging experiment in directory: (?P<path>.*\S)")
"""``train``'s announcement of its run directory (printed by rank 0 only)."""
EVENT_FILE_GLOB = "events.out.tfevents.*"
TB_SYNC_INTERVAL_S = 30.0
"""How often the combined TensorBoard dir is refreshed while stage 2 runs (its
event file appears only once the env is built)."""
RUN_ID_LENGTH = 8
SIGNAL_REPEAT_S = 1.0
"""A signal this soon after the last one the driver acted on is the same request:
``uv run`` without a terminal forwards to the driver the SIGINT that its whole
process group has just received, and a held-down Ctrl-C repeats."""

_FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_ESCALATION = (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)
"""What the step's group gets for the driver's first, second and later signals."""
_SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None
_CHECKPOINT = re.compile(r"model_(\d+)\.pt")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RUN_ID_CHARS = string.ascii_lowercase + string.digits
_MODELS: tuple[Model, ...] = ("robot", "msk")
_SOCKETS: tuple[Socket, ...] = ("rigid", "compliant")
_REWARDS: tuple[Reward, ...] = ("mimic", "clf")


@dataclass
class Stage1Cfg:
  reward: Reward = "clf"
  """Stage 1's reward. Unused with ``run-dir``: that run's own reward counts."""
  iterations: int = STAGE1_ITERATIONS
  """Iterations to train, and the length of the linear learning-rate decay."""
  run_dir: str = ""
  """An existing stage-1 run to split instead of training one. Its model and
  socket must be ``--model`` and ``--socket``."""
  checkpoint: str = ""
  """The checkpoint to split, inside the run directory; empty takes the
  highest-numbered ``model_N.pt``."""
  extra_args: str = ""
  """More ``train`` arguments as one shell-quoted string, passed last."""


@dataclass
class Stage2Cfg:
  reward: Literal["", "mimic", "clf"] = ""
  """Stage 2's reward; empty keeps stage 1's. A different reward is the one switch
  allowed between the stages (the critics then start fresh)."""
  iterations: int = 1000
  human_freeze_iterations: int = 200
  """Stage-2 iterations with the human actor frozen."""
  extra_args: str = ""
  """More ``train`` arguments as one shell-quoted string, passed last."""
  skip: bool = False
  """Stop after the split, leaving the warm-start file for a later stage 2."""


@dataclass
class TwoStageCfg:
  """Run OSL stage 1, the split and stage 2 as subprocesses."""

  name: str
  """Names the runs (``<name>_stage1``/``_stage2``), their W&B group and the
  pipeline's own files in ``<log-root>/pipelines/``."""
  model: Model = "msk"
  socket: Socket = "compliant"
  library_dir: str = ""
  """The clipper unicycle library folder, for both stages."""
  stage1: Stage1Cfg = field(default_factory=Stage1Cfg)
  stage2: Stage2Cfg = field(default_factory=Stage2Cfg)
  curves: Curves = "offset"
  """``offset``: link both runs into ``<log-root>/pipelines/<name>_tb/``, where
  stage 2's curves continue stage 1's; ``none``: no combined TensorBoard dir."""
  log_root: str = "logs/rsl_rl"
  """Root of the training runs (passed on as ``--log-root``) and of
  ``pipelines/``."""
  dry_run: bool = False
  """Print every command and run nothing."""
  wait_for_gpu: bool = False
  """Wait while another process holds the GPU, instead of exiting."""
  gpu_poll_s: float = 60.0
  """Seconds between GPU checks while waiting."""


class PipelineError(RuntimeError):
  """A step cannot run or has failed; ``exit_code`` is the driver's exit status."""

  def __init__(self, message: str, exit_code: int = 1) -> None:
    super().__init__(message)
    self.exit_code = exit_code


class GpuBusyError(PipelineError):
  """Another process holds the GPU and the driver was not asked to wait."""


##
# Task ids and commands.
##


def task_tags(model: Model, socket: Socket, reward: Reward) -> str:
  """The task-id suffix, in registration order: ``-ROBOT``/``-MSK``, ``-CLF``,
  ``-Compliant``."""
  return (
    MODEL_TAGS[model]
    + ("-CLF" if reward == "clf" else "")
    + ("-Compliant" if socket == "compliant" else "")
  )


def stage1_task_id(model: Model, socket: Socket, reward: Reward) -> str:
  return STAGE1_TASK + task_tags(model, socket, reward)


def stage2_task_id(model: Model, socket: Socket, reward: Reward) -> str:
  return STAGE2_TASK + task_tags(model, socket, reward)


def stage2_reward(cfg: TwoStageCfg, stage1_reward: Reward) -> Reward:
  """``--stage2.reward``, or stage 1's when it is empty."""
  return cfg.stage2.reward or stage1_reward


def _abspath(path: str | Path) -> str:
  """Absolute, so the runs' saved configs and lineage stay usable from anywhere."""
  return os.path.abspath(os.path.expanduser(path))


def stage1_command(cfg: TwoStageCfg, python: str = sys.executable) -> list[str]:
  iterations = str(cfg.stage1.iterations)
  return [
    python,
    "-m",
    TRAIN_MODULE,
    stage1_task_id(cfg.model, cfg.socket, cfg.stage1.reward),
    "--env.commands.motion.library-dir",
    _abspath(cfg.library_dir),
    "--agent.run-name",
    f"{cfg.name}_stage1",
    "--agent.max-iterations",
    iterations,
    "--agent.algorithm.lr-schedule-updates",
    iterations,
    "--log-root",
    _abspath(cfg.log_root),
    *shlex.split(cfg.stage1.extra_args),
  ]


def split_command(
  run_dir: str | Path,
  out: str | Path,
  checkpoint: str = "",
  python: str = sys.executable,
) -> list[str]:
  command = [python, "-m", SPLIT_MODULE, str(run_dir)]
  if checkpoint:
    command += ["--checkpoint", checkpoint]
  return command + ["--out", str(out)]


def stage2_command(
  cfg: TwoStageCfg,
  warm_start: str | Path,
  reward: Reward,
  python: str = sys.executable,
) -> list[str]:
  return [
    python,
    "-m",
    TRAIN_MODULE,
    stage2_task_id(cfg.model, cfg.socket, reward),
    "--env.commands.motion.library-dir",
    _abspath(cfg.library_dir),
    "--agent.run-name",
    f"{cfg.name}_stage2",
    "--agent.warm-start.path",
    str(warm_start),
    "--agent.max-iterations",
    str(cfg.stage2.iterations),
    "--agent.freeze.human-actor-iterations",
    str(cfg.stage2.human_freeze_iterations),
    "--log-root",
    _abspath(cfg.log_root),
    *shlex.split(cfg.stage2.extra_args),
  ]


def format_command(command: Sequence[str], env: dict[str, str] | None = None) -> str:
  """A copy-pastable shell line: the environment overrides, then the command."""
  prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in (env or {}).items())
  return f"{prefix} {shlex.join(command)}" if prefix else shlex.join(command)


def new_run_id() -> str:
  """A fresh W&B run id, in wandb's own format (8 base-36 characters)."""
  return "".join(secrets.choice(_RUN_ID_CHARS) for _ in range(RUN_ID_LENGTH))


def wandb_env(group: str, run_id: str) -> dict[str, str]:
  return {"WANDB_RUN_GROUP": group, "WANDB_RUN_ID": run_id}


##
# Stage-1 runs and their checkpoints.
##


def parse_run_dir(line: str) -> Path | None:
  """The run directory announced by a ``train`` output line, if it is that line."""
  match = RUN_DIR_LINE.search(line)
  return Path(match["path"]) if match else None


def checkpoint_iteration(path: str | Path) -> int | None:
  """``N`` of a ``model_N.pt`` name: the iteration it was saved at."""
  match = _CHECKPOINT.fullmatch(Path(path).name)
  return int(match[1]) if match else None


def latest_checkpoint(run_dir: Path) -> Path | None:
  """The highest-numbered ``model_N.pt`` (by number: ``model_610`` sorts after
  ``model_3050`` by name)."""
  best: tuple[int, Path] | None = None
  for path in run_dir.glob("model_*.pt"):
    iteration = checkpoint_iteration(path)
    if iteration is not None and (best is None or iteration > best[0]):
      best = (iteration, path)
  return None if best is None else best[1]


def _agent_yaml(run_dir: Path) -> str:
  """A run's ``params/agent.yaml``, which ``train`` writes before it trains.

  The driver reads it with line scans instead of a YAML parser, to stay
  dependency-free: ``dump_yaml`` writes the runner settings as top-level
  ``key: value`` lines and the ``validation`` block as a flat, two-space-indented
  mapping of plain strings.
  """
  path = run_dir / "params" / "agent.yaml"
  try:
    return path.read_text(encoding="utf-8")
  except OSError as e:
    raise PipelineError(f"Cannot read the stage-1 run's settings: {e}") from e


def _top_level(text: str, key: str) -> str | None:
  """The plain value of a top-level ``key`` in an ``agent.yaml``."""
  match = re.search(rf"^{re.escape(key)}:[ \t]*(.*)$", text, re.M)
  return match[1].strip().strip("'\"") if match else None


def read_run_settings(run_dir: Path) -> dict[str, str]:
  """A stage-1 run's ``model``, ``socket``, ``reward`` (and ``variant``), from the
  ``validation`` block of its ``params/agent.yaml``."""
  path = run_dir / "params" / "agent.yaml"
  text = _agent_yaml(run_dir)
  block = re.search(r"^validation:[ \t]*\n((?:[ \t]+\S.*(?:\n|$))*)", text, re.M)
  settings = {
    key: value.strip().strip("'\"")
    for key, value in re.findall(
      r"^  (\w+):[ \t]*(.*)$", block[1] if block else "", re.M
    )
  }
  for key, allowed in (("model", _MODELS), ("socket", _SOCKETS), ("reward", _REWARDS)):
    if settings.get(key) not in allowed:
      raise PipelineError(
        f"{path} has no validation.{key} in {allowed}: is {run_dir} a stage-1 "
        "gait-library run?"
      )
  return settings


def planned_last_iteration(run_dir: Path) -> int | None:
  """The iteration a training run ends at unless something stops it, from its
  ``params/agent.yaml``; None when the file does not say.

  ``learn`` runs ``max_iterations`` iterations from the one it starts at, saving
  every ``save_interval``-th and, once done, the last as
  ``model_<start + max_iterations - 1>.pt``. A fresh run starts at 0. A resumed run
  (``resume: true``) starts at the iteration of the checkpoint it loaded, so its
  ``max_iterations`` counts only the iterations it adds. That start is
  ``load_checkpoint`` when it names one ``model_<N>.pt``. Otherwise it is the run's
  lowest-numbered checkpoint, since the resumed run saves its first iteration again
  when that is a multiple of ``save_interval``, as every checkpoint but a run's last
  is; for the same reason, a highest checkpoint off the save interval can only be
  the last one, and so is the end.
  """
  text = _agent_yaml(run_dir)
  iterations = _top_level(text, "max_iterations") or ""
  if not iterations.isdigit():
    return None
  if _top_level(text, "resume") != "true":
    return int(iterations) - 1
  start = checkpoint_iteration(_top_level(text, "load_checkpoint") or "")
  if start is None:
    saved = sorted(
      iteration
      for path in run_dir.glob("model_*.pt")
      if (iteration := checkpoint_iteration(path)) is not None
    )
    if not saved:
      return None
    interval = _top_level(text, "save_interval") or ""
    if interval.isdigit() and int(interval) > 0 and saved[-1] % int(interval):
      return saved[-1]
    start = saved[0]
  return start + int(iterations) - 1


##
# The GPU guard.
##


@dataclass(frozen=True)
class GpuProcess:
  pid: int | None
  """None when the line did not parse; such a process still counts as busy."""
  name: str
  used_memory: str

  def __str__(self) -> str:
    pid = "pid ?" if self.pid is None else f"pid {self.pid}"
    details = ", ".join(part for part in (self.name, self.used_memory) if part)
    return f"{pid} ({details})" if details else pid


def parse_gpu_processes(text: str) -> list[GpuProcess]:
  """``nvidia-smi``'s ``--query-compute-apps=pid,process_name,used_memory
  --format=csv,noheader`` output: one ``pid, name, memory`` line per process. A name
  may contain commas; an unparseable line still counts, so the guard errs towards
  waiting."""
  processes: list[GpuProcess] = []
  for raw in text.splitlines():
    line = raw.strip()
    if not line or line.lower().startswith("no running"):
      continue
    parts = [part.strip() for part in line.split(",")]
    pid = int(parts[0]) if parts[0].isdigit() else None
    if pid is None:
      name, memory = line, ""
    elif len(parts) >= 3:
      name, memory = ", ".join(parts[1:-1]), parts[-1]
    else:
      name, memory = ", ".join(parts[1:]), ""
    processes.append(GpuProcess(pid, name, memory))
  return processes


def query_gpu_processes() -> list[GpuProcess] | None:
  """The processes holding a GPU, or None when ``nvidia-smi`` is not installed."""
  try:
    result = subprocess.run(
      list(NVIDIA_SMI_QUERY),
      capture_output=True,
      text=True,
      timeout=60.0,
      check=False,
    )
  except FileNotFoundError:
    return None
  except subprocess.TimeoutExpired as e:
    raise PipelineError("nvidia-smi did not answer within 60 s.") from e
  if result.returncode != 0:
    detail = (result.stderr or result.stdout or "").strip()
    raise PipelineError(f"nvidia-smi failed (exit code {result.returncode}): {detail}")
  return parse_gpu_processes(result.stdout)


def ensure_gpu_free(
  log: PipelineLog,
  *,
  wait: bool,
  poll_s: float,
  ignore_pids: Collection[int] = (),
) -> None:
  """Return once no other process holds a GPU; unless ``wait``, raise
  :class:`GpuBusyError` instead of waiting. ``ignore_pids``: this driver and its
  own children (a step that just exited must not block the next one)."""
  reported: set[int | None] | None = None
  while True:
    processes = query_gpu_processes()
    if processes is None:
      log.say("nvidia-smi not found: cannot check that the GPU is free; going on.")
      return
    busy = [p for p in processes if p.pid is None or p.pid not in ignore_pids]
    if not busy:
      if reported is not None:
        log.say("The GPU is free.")
      return
    holders = "; ".join(str(p) for p in busy)
    if not wait:
      raise GpuBusyError(
        f"The GPU is in use by {holders}. Let it finish, or pass --wait-for-gpu."
      )
    pids = {p.pid for p in busy}
    if pids != reported:
      log.say(
        f"Waiting for the GPU (in use by {holders}); checking every {poll_s:g} s."
      )
      reported = pids
    time.sleep(poll_s)


##
# The combined TensorBoard dir.
##


def sync_combined_tb(tb_dir: Path, run_dirs: Sequence[Path]) -> list[Path]:
  """Make ``tb_dir`` hold exactly one symlink per event file of ``run_dirs``.

  The links keep the event files' names, which start with their creation time, so
  TensorBoard reads stage 1 before stage 2. Links to anything else (an earlier
  attempt's stage 2, say) are removed, since its steps would interleave with this
  one's; files that are not symlinks are left alone. Returns the linked files.
  """
  targets: dict[str, Path] = {}
  for run_dir in run_dirs:
    for path in sorted(Path(run_dir).glob(EVENT_FILE_GLOB)):
      if path.is_file():
        targets[path.name] = path.resolve()
  tb_dir.mkdir(parents=True, exist_ok=True)
  for link in tb_dir.iterdir():
    if link.is_symlink() and Path(os.readlink(link)) != targets.get(link.name):
      link.unlink()
  for name, target in targets.items():
    link = tb_dir / name
    if not link.is_symlink() and not link.exists():
      link.symlink_to(target)
  return sorted(targets.values())


class _TensorBoardLinks:
  """:func:`sync_combined_tb` while stage 2 runs: throttled, and warning once
  instead of failing, since a convenience must never stop a training stage."""

  def __init__(self, tb_dir: Path, stage1_run: Path | None, log: PipelineLog) -> None:
    self.tb_dir = tb_dir
    self.stage1_run = stage1_run
    self.log = log
    self._last = -float("inf")
    self._warned = False
    self._announced = False

  def sync(self, stage2_run: Path, force: bool = False) -> None:
    now = time.monotonic()
    if not force and now - self._last < TB_SYNC_INTERVAL_S:
      return
    self._last = now
    runs = [stage2_run] if self.stage1_run is None else [self.stage1_run, stage2_run]
    try:
      sync_combined_tb(self.tb_dir, runs)
    except OSError as e:
      if not self._warned:
        self.log.say(f"Warning: cannot update {self.tb_dir}: {e}")
        self._warned = True
      return
    if not self._announced:
      self.log.say(f"Combined TensorBoard dir: {self.tb_dir}")
      self._announced = True


##
# Running the steps.
##


class PipelineLog:
  """The console, plus ``<log-root>/pipelines/<name>.log`` unless dry-running."""

  def __init__(self, path: Path | None) -> None:
    self.path = path
    self._file: IO[str] | None = None
    if path is not None:
      path.parent.mkdir(parents=True, exist_ok=True)
      self._file = path.open("a", encoding="utf-8")

  def write(self, text: str, stream: IO[str] | None = None) -> None:
    out = sys.stdout if stream is None else stream
    try:
      out.write(text)
      out.flush()
    except OSError:
      # The console went with its terminal (the SIGHUP case). The driver must go on
      # draining the step until it stops, and the file still gets everything.
      pass
    if self._file is not None:
      self._file.write(text)
      self._file.flush()

  def say(self, message: str) -> None:
    self.write(f"[pipeline] {message}\n")

  def warn(self, message: str) -> None:
    """A warning that must not scroll by unread: framed, on stderr."""
    rule = f"[pipeline] {'!' * 68}\n"
    self.write(f"{rule}[pipeline] WARNING: {message}\n{rule}", sys.stderr)

  def error(self, message: str) -> None:
    self.write(f"[pipeline] Error: {message}\n", sys.stderr)

  def close(self) -> None:
    if self._file is not None:
      self._file.close()
      self._file = None

  def __enter__(self) -> PipelineLog:
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
  ) -> None:
    self.close()


class _StepSignals:
  """While a step runs, turns the driver's SIGINT, SIGTERM and SIGHUP into signals
  to the step's process group: the first into SIGINT, which stops the step as
  Ctrl-C would, the second into SIGTERM, the third and any later one into SIGKILL.

  The step has a process group of its own, so this is the only way a signal reaches
  it, whether it came from the terminal, ``kill``, a service manager or ``uv``. A
  signal within :data:`SIGNAL_REPEAT_S` of the last one acted on counts as the same
  request. The handlers neither raise nor write. Not raising lets the driver drain
  the step's output until the step has exited, so that the step never blocks on a
  full pipe and its last lines reach the log. Not writing keeps a handler from
  re-entering a write that the signal interrupted; :meth:`report` logs what they
  did instead. A signal ignored when the step starts (``nohup``'s SIGHUP, say)
  stays ignored.
  """

  def __init__(self) -> None:
    self.received = 0
    """The signals acted on."""
    self._process: subprocess.Popen[str] | None = None
    self._last = -math.inf
    self._notes: list[str] = []
    self._previous: dict[int, _SignalHandler] = {}

  def __enter__(self) -> _StepSignals:
    for signum in _FORWARDED_SIGNALS:
      previous = signal.getsignal(signum)
      if previous is not None and previous != signal.SIG_IGN:
        self._previous[signum] = previous
        signal.signal(signum, self._handle)
    return self

  def __exit__(self, *exc_info: object) -> None:
    for signum, previous in self._previous.items():
      signal.signal(signum, previous)
    self._previous.clear()

  def attach(self, process: subprocess.Popen[str]) -> None:
    """Forward to ``process`` from now on, passing on a signal that came while it
    was being started (the handlers are in place first, so that none is missed)."""
    self._process = process
    if self.received:
      self.send(_ESCALATION[min(self.received, len(_ESCALATION)) - 1])

  def send(self, signum: int) -> None:
    """Send ``signum`` to the step's process group, unless the step is reaped
    already (its process id, which names the group, could then be reused)."""
    process = self._process
    if process is None or process.returncode is not None:
      return
    try:
      os.killpg(process.pid, signum)
    except ProcessLookupError:
      pass  # The whole group has exited.

  def report(self, log: PipelineLog) -> None:
    """Log what the handlers have done since the last report."""
    while self._notes:
      log.say(self._notes.pop(0))

  def _handle(self, signum: int, frame: FrameType | None) -> None:
    del frame
    now = time.monotonic()
    if now - self._last < SIGNAL_REPEAT_S:
      return
    self._last = now
    self.received += 1
    forward = _ESCALATION[min(self.received, len(_ESCALATION)) - 1]
    self.send(forward)
    name = signal.Signals(signum).name
    if self.received == 1:
      self._notes.append(
        f"{name}: sent SIGINT to the step; waiting for it to stop (a second signal "
        "sends SIGTERM, a third SIGKILL)."
      )
    else:
      self._notes.append(f"{name} again: sent {forward.name} to the step.")


def run_child(
  command: Sequence[str],
  env: dict[str, str],
  log: PipelineLog,
  *,
  on_line: Callable[[str], None] | None = None,
  pids: set[int] | None = None,
) -> int:
  """Run ``command`` with ``env`` over the driver's environment, streaming its
  merged output to ``log``; return its exit code. ``pids`` collects the child's.

  The child gets a process group and session of its own, so signals reach it only
  through the driver (see :class:`_StepSignals`): a terminal's Ctrl-C and a
  ``kill`` of the driver alone then act alike. Once the child has exited, an
  interrupt is raised as KeyboardInterrupt.
  """
  log.say(f"$ {format_command(command, env)}")
  with _StepSignals() as signals:
    try:
      process = subprocess.Popen(
        list(command),
        env={**os.environ, "PYTHONUNBUFFERED": "1", **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=True,
      )
    except OSError as e:
      raise PipelineError(f"Cannot start {command[0]}: {e}") from e
    if pids is not None:
      pids.add(process.pid)
    signals.attach(process)
    stdout = process.stdout
    assert stdout is not None
    try:
      for line in stdout:
        signals.report(log)
        log.write(line)
        if on_line is not None:
          on_line(line)
    except BaseException:
      # The driver itself failed. Nothing drains the step from here on, and it
      # would block on a full pipe: stop it.
      signals.send(signal.SIGTERM)
      raise
    finally:
      stdout.close()
      code = process.wait()
  signals.report(log)
  if signals.received:
    log.say(f"The step has stopped ({_exit_status(code)}).")
    raise KeyboardInterrupt
  return code


def _exit_status(code: int) -> str:
  """A child's exit code in words: a negative one is the signal that ended it."""
  if code >= 0:
    return f"exit code {code}"
  try:
    return f"killed by {signal.Signals(-code).name}"
  except ValueError:
    return f"killed by signal {-code}"


def _exit_code(code: int) -> int:
  """A failed child's exit code, as the driver's (signals become 1)."""
  return code if 0 < code < 256 else 1


@dataclass(frozen=True)
class PipelinePaths:
  root: Path
  log: Path
  warm_start: Path
  tensorboard: Path

  @classmethod
  def of(cls, cfg: TwoStageCfg) -> PipelinePaths:
    root = Path(_abspath(cfg.log_root)) / "pipelines"
    return cls(
      root=root,
      log=root / f"{cfg.name}.log",
      warm_start=root / f"{cfg.name}_warm_start.pt",
      tensorboard=root / f"{cfg.name}_tb",
    )


def check_cfg(cfg: TwoStageCfg) -> None:
  """Everything that can be checked before anything runs."""
  if not _NAME.fullmatch(cfg.name):
    raise PipelineError(
      f"--name {cfg.name!r}: use letters, digits, '.', '_' and '-', starting with "
      "a letter or digit (it names directories and the W&B group)."
    )
  if not cfg.stage1.run_dir and cfg.stage1.iterations <= 0:
    raise PipelineError("--stage1.iterations must be positive.")
  if cfg.stage2.iterations <= 0 or cfg.stage2.human_freeze_iterations < 0:
    raise PipelineError(
      "--stage2.iterations must be positive and --stage2.human-freeze-iterations "
      "not negative."
    )
  if cfg.gpu_poll_s <= 0:
    raise PipelineError("--gpu-poll-s must be positive.")
  for flag, extra in (
    ("--stage1.extra-args", cfg.stage1.extra_args),
    ("--stage2.extra-args", cfg.stage2.extra_args),
  ):
    try:
      shlex.split(extra)
    except ValueError as e:
      raise PipelineError(f"{flag}: {e}") from e
  trains = not cfg.stage1.run_dir or not cfg.stage2.skip
  if trains and not (cfg.library_dir and Path(cfg.library_dir).expanduser().is_dir()):
    raise PipelineError(
      "--library-dir must name the clipper unicycle library folder (got "
      f"{cfg.library_dir!r})."
    )
  if cfg.stage1.run_dir and not Path(cfg.stage1.run_dir).expanduser().is_dir():
    raise PipelineError(f"--stage1.run-dir {cfg.stage1.run_dir!r} is not a directory.")


class _Pipeline:
  """One pipeline run: the steps in order, and what each hands to the next."""

  def __init__(
    self, cfg: TwoStageCfg, paths: PipelinePaths, log: PipelineLog, python: str
  ) -> None:
    self.cfg = cfg
    self.paths = paths
    self.log = log
    self.python = python
    self.children: set[int] = set()
    """The steps launched so far: never GPU competitors."""
    self.stage1_id = new_run_id()
    self.stage2_id = new_run_id()
    self.stage1_run: Path | None = None
    """None only while dry-running a stage 1 the driver would train."""

  def run(self) -> int:
    cfg, log = self.cfg, self.log
    verb = "Dry run of" if cfg.dry_run else "Starting"
    log.say(
      f"{verb} pipeline {cfg.name!r} at {time.strftime('%Y-%m-%d %H:%M:%S')}: "
      f"model {cfg.model}, socket {cfg.socket}, W&B group {cfg.name}."
    )
    if cfg.dry_run:
      log.say(
        f"Each GPU stage first needs `{shlex.join(NVIDIA_SMI_QUERY)}` to list no "
        "other process."
      )

    # Stage 1: an existing run, or a new one.
    if cfg.stage1.run_dir:
      self.stage1_run = Path(_abspath(cfg.stage1.run_dir))
      reward1 = self._given_stage1(self.stage1_run)
    else:
      reward1 = cfg.stage1.reward
      command1 = stage1_command(cfg, self.python)
      self.stage1_run = self._train("Stage 1", command1, self.stage1_id)
    reward2 = stage2_reward(cfg, reward1)
    if reward2 != reward1:
      log.say(
        f"Reward switch: stage 1 {reward1}, stage 2 {reward2}; the stage-2 critics "
        "start fresh."
      )
    # Check the GPU before the split when stage 2 follows: a stage 1 still running
    # elsewhere must end first, or the split would take one of its early checkpoints.
    if not cfg.stage2.skip:
      self._guard()
    if self.stage1_run is not None:
      self._check_handover(self.stage1_run)
    self._split()

    command2 = stage2_command(cfg, self.paths.warm_start, reward2, self.python)
    if cfg.stage2.skip:
      later = format_command(command2, {"WANDB_RUN_GROUP": cfg.name})
      log.say(f"Stage 2 skipped; to run it later:\n  {later}")
      return 0
    stage2_run = self._train("Stage 2", command2, self.stage2_id, link_tensorboard=True)
    if self.stage1_run is not None and stage2_run is not None:
      self._summary(self.stage1_run, stage2_run)
    return 0

  def _guard(self) -> None:
    if not self.cfg.dry_run:
      ensure_gpu_free(
        self.log,
        wait=self.cfg.wait_for_gpu,
        poll_s=self.cfg.gpu_poll_s,
        ignore_pids={os.getpid(), *self.children},
      )

  def _given_stage1(self, run_dir: Path) -> Reward:
    """Check an existing stage-1 run against ``--model``/``--socket``; return its
    reward."""
    cfg, log = self.cfg, self.log
    settings = read_run_settings(run_dir)
    for key, wanted in (("model", cfg.model), ("socket", cfg.socket)):
      if settings[key] != wanted:
        raise PipelineError(
          f"{run_dir} was trained with {key} {settings[key]!r}, not --{key} "
          f"{wanted!r}: a warm-started human actor cannot change its model or "
          "socket (only the reward may switch between the stages)."
        )
    variant = settings.get("variant", "non-gait-library")
    if variant != "unicycle":
      log.say(f"Warning: {run_dir} is a {variant!r} run, not a unicycle one.")
    reward: Reward = "clf" if settings["reward"] == "clf" else "mimic"
    log.say(f"Stage 1: the existing run {run_dir} (reward {reward}).")
    return reward

  def _check_handover(self, run_dir: Path) -> None:
    """Fail early on a missing checkpoint or an unfinished stage 1; warn when stage
    1 ran past the chosen checkpoint."""
    latest = latest_checkpoint(run_dir)
    if self.cfg.stage1.checkpoint:
      chosen = run_dir / self.cfg.stage1.checkpoint
      if not chosen.is_file():
        raise PipelineError(f"No checkpoint {chosen}.")
    elif latest is None:
      raise PipelineError(f"{run_dir} holds no model_N.pt checkpoint.")
    else:
      chosen = latest
      self._check_finished(run_dir, latest)
    handover = checkpoint_iteration(chosen)
    if handover is None:
      return
    self.log.say(
      f"Handover: {chosen.name}; stage 2 starts at iteration {handover + 1}."
    )
    last = None if latest is None else checkpoint_iteration(latest)
    if self.cfg.curves == "offset" and last is not None and handover < last:
      self.log.say(
        f"Warning: stage 1 ran on to iteration {last}, so in "
        f"{self.paths.tensorboard} its iterations after {handover} interleave with "
        "stage 2's. Compare the two run directories side by side instead."
      )

  def _check_finished(self, run_dir: Path, latest: Path) -> None:
    """Refuse to split a stage 1 whose highest-numbered checkpoint falls short of
    its last planned iteration, unless stage 2 is skipped (then warn).

    The GPU guard waits for the GPU to be free, not for stage 1 to finish, and a
    stage 1 that crashed frees it too; an unattended pipeline would then train stage
    2 for hours from a policy in the middle of its learning-rate decay. A dry run
    only warns, since a real run checks after its GPU guard, by which time a stage 1
    still training may have finished.
    """
    at = checkpoint_iteration(latest)
    last = planned_last_iteration(run_dir)
    if last is None:
      self.log.say(
        f"Warning: cannot tell whether {run_dir} finished: its params/agent.yaml "
        "gives no max_iterations."
      )
      return
    if at is None or at >= last:
      return
    problem = (
      f"{run_dir} stopped at iteration {at} of {last} (crashed, stopped or still "
      "training)."
    )
    remedy = (
      "Resume it with train and pass the resumed run as --stage1.run-dir, or pass "
      f"--stage1.checkpoint {latest.name} to split it anyway."
    )
    if self.cfg.stage2.skip:
      self.log.warn(f"{problem} Splitting it anyway, since stage 2 is skipped.")
    elif self.cfg.dry_run:
      self.log.warn(
        f"{problem} A real run checks again after its GPU guard and stops there if "
        f"the run has still not finished. {remedy}"
      )
    else:
      raise PipelineError(f"{problem} {remedy}")

  def _train(
    self,
    label: str,
    command: list[str],
    run_id: str,
    link_tensorboard: bool = False,
  ) -> Path | None:
    """Launch one training stage; return its run directory (None when dry)."""
    cfg, log = self.cfg, self.log
    env = wandb_env(cfg.name, run_id)
    link = link_tensorboard and cfg.curves == "offset"
    if cfg.dry_run:
      log.say(f"{label}: {format_command(command, env)}")
      if link:
        log.say(f"Combined TensorBoard dir: {self.paths.tensorboard}")
      return None
    self._guard()
    log.say(f"{label}: W&B run {run_id} in group {cfg.name}.")
    watcher = _RunDirWatcher()
    links = (
      _TensorBoardLinks(self.paths.tensorboard, self.stage1_run, log) if link else None
    )

    def on_line(line: str) -> None:
      watcher(line)
      if links is not None and watcher.run_dir is not None:
        links.sync(watcher.run_dir)

    code = run_child(command, env, log, on_line=on_line, pids=self.children)
    if links is not None and watcher.run_dir is not None:
      links.sync(watcher.run_dir, force=True)
    if code != 0:
      raise PipelineError(
        f"{label} failed (exit code {code}); its output is in {self.paths.log}.",
        exit_code=_exit_code(code),
      )
    if watcher.run_dir is None or not watcher.run_dir.is_dir():
      raise PipelineError(
        f"{label} finished, but announced no run directory (train's '[INFO] "
        f"Logging experiment in directory:' line; got {watcher.run_dir})."
      )
    return watcher.run_dir

  def _split(self) -> None:
    source = self.stage1_run if self.stage1_run is not None else "<stage-1 run dir>"
    out = self.paths.warm_start
    command = split_command(source, out, self.cfg.stage1.checkpoint, self.python)
    env = {"CUDA_VISIBLE_DEVICES": ""}
    if self.cfg.dry_run:
      self.log.say(f"Split: {format_command(command, env)}")
      return
    before = out.stat().st_mtime_ns if out.exists() else None
    code = run_child(command, env, self.log, pids=self.children)
    if code != 0:
      raise PipelineError(
        f"The split failed (exit code {code}); its output is in {self.paths.log}.",
        exit_code=_exit_code(code),
      )
    if not out.is_file() or out.stat().st_mtime_ns == before:
      raise PipelineError(f"The split finished without writing {out}.")

  def _summary(self, stage1_run: Path, stage2_run: Path) -> None:
    log = self.log
    pinned = "" if self.cfg.stage1.run_dir else f" (W&B run {self.stage1_id})"
    log.say(f"Done. Stage 1: {stage1_run}{pinned}.")
    log.say(f"Warm start: {self.paths.warm_start}.")
    log.say(f"Stage 2: {stage2_run} (W&B run {self.stage2_id}).")
    if self.cfg.curves == "offset":
      log.say(f"Both stages in one view: tensorboard --logdir {self.paths.tensorboard}")


class _RunDirWatcher:
  """Remembers the first run directory a step announces."""

  def __init__(self) -> None:
    self.run_dir: Path | None = None

  def __call__(self, line: str) -> None:
    if self.run_dir is None:
      self.run_dir = parse_run_dir(line)


def run_pipeline(cfg: TwoStageCfg, python: str = sys.executable) -> int:
  """Run (or, with ``dry_run``, print) the pipeline; return the exit status."""
  try:
    check_cfg(cfg)
  except PipelineError as e:
    print(f"[pipeline] Error: {e}", file=sys.stderr)
    return e.exit_code
  paths = PipelinePaths.of(cfg)
  with PipelineLog(None if cfg.dry_run else paths.log) as log:
    try:
      return _Pipeline(cfg, paths, log, python).run()
    except PipelineError as e:
      log.error(str(e))
      return e.exit_code
    except KeyboardInterrupt:
      log.error("Interrupted.")
      return 130


def main(argv: Sequence[str] | None = None) -> int:
  """The command line: ``argv`` defaults to ``sys.argv[1:]``."""
  cfg = tyro.cli(TwoStageCfg, args=argv, prog="two_stage")
  return run_pipeline(cfg)


if __name__ == "__main__":
  sys.exit(main())
