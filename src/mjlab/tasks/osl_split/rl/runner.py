"""The stage-2 runner: stage 1's loop and validation, with two agents.

Stage 2 trains the human and the OSL as separate PPO agents in one env
(:class:`~.ppo.TwoAgentPPO`). This runner is stage 1's
(:class:`~mjlab.tasks.osl_gait_library.rl.runner.GaitLibraryOnPolicyRunner`)
wherever one agent did not become two:

* **Start.** Stage 2 always starts from stage 1: ``--agent.warm-start.path`` (a
  stage-1 run, checkpoint or converted file, see :mod:`.warmstart`) gives the human
  actor, both critics, the return statistics and the iteration axis. Only the
  reward type may differ from stage 1's (the reward switch). A model or socket
  change would change the human actor's inputs or outputs, so it is refused, as is
  a network built otherwise than the stage-1 one whose weights it takes (another
  activation loads strictly and computes something else). Stage 1's recorded end
  state is compared with stage 2's settings: another CLF penalty weight is refused
  (``--agent.warm-start.allow-pen-w-mismatch`` overrides), and another twist
  weight, root-deviation limit or library release, a large human std, an
  unfinished stage-1 run or a human rate above stage 1's last one are reported.
* **Iterations.** Stage 2 continues stage 1's numbering: its first iteration is
  stage 1's last plus one, so W&B and TensorBoard steps run on from stage 1's, and
  ``Train/mean_reward`` (the team reward, which the split reward term keeps equal
  to stage 1's) continues stage 1's curve. Schedules, saves and evaluations count
  *stage-local* iterations ``k`` from 0. A resumed stage-2 run takes its stage
  start and lineage from its checkpoint, and its curriculum (schedule, channels,
  additive weights) from the checkpoint or, with ``resume_curriculum="config"``,
  from the command line; it does not replay the checkpoint's own iteration.
* **Rewards.** After every step each agent's reward is composed from the reward
  manager (:class:`AgentRewards`): its imitation channel of the split ``mimic``
  term, plus every other (additive) term times its per-agent weight. The env's own
  reward stays the team reward, which the logger receives.
* **Schedules.** Before every rollout the algorithm takes ``k`` (freeze flags,
  per-module learning rates, the blend) and the reward term takes the channels'
  group weights for ``k``. The evaluation envs' reward terms take the same weights
  before every evaluation.
* **Validation** is stage 1's protocol (one env per library clip, deterministic,
  ending early only on the 0.5 m site check), in families (``validation.inference``,
  see :class:`~..config.rl_cfg.SplitValidationCfg`): ``Validation/`` plays both
  actors' means, the deployment metric. Before the first update the zero-shot
  validation plays the fresh OSL actor (the floor it starts from) and the teacher,
  the unified stage-1 policy, at its raw mean (``ValidationTeacher/``) and
  noise-averaged (``ValidationTeacherExpected/``): they should match the stage-1
  checkpoint's own ``Validation/`` and ``ValidationExpected/``. The validation env
  is the task's validation config with the training run's settings (simulation,
  observation groups, every shared reward term's weight and params) and the
  training OSL observation read exactly, at the middle of its training lag. Per
  clip family it also reports the left and right tight arcs and pivots.
* **Stress validation** (every ``stress_interval`` stage-local iterations, and at
  zero-shot with the teacher alone) plays a copy of the training config, the OSL
  sensor noise and every randomization included, seeded by
  ``validation.stress_seed``, on 512 envs with validation's site check, and
  reports each env's first episode: ``Stress/`` (both means), the deployment proxy
  for the OSL actor, and ``StressOslSampled/`` (the OSL actor sampled), whose gap to
  it shows an OSL actor relying on its own noise.
* **Guards.** Every evaluation (validation, stress, video) is guarded: a failure,
  such as running out of GPU memory, prints a warning, its memory is released and
  training goes on. Their wall times are logged (``Perf/<evaluation>_time``), as is
  the GPU memory (also after the last save and video).
* **Logging** adds per-term policy statistics of both actors (``Policy/std/``,
  ``Policy/noise_gap/``, ``Policy/mean_outside/``, stage 1's names), each critic's
  explained variance (``Critic/``), the learning rates, the freeze and blend state
  and the channels' weights and shares.
* **Checkpoints** hold the four networks, their optimizers, both return
  normalizers and the split state (the stage's place and its curriculum), every
  ``save_interval`` stage-local iterations except the first iteration of a launch,
  and at its end. Each comes with the OSL actor alone, as ONNX plus its metadata
  (:mod:`.export`), in ``exported/`` next to it. ``play.py``'s load restores only
  the actors.
"""

from __future__ import annotations

import gc
import math
import os
import shlex
import statistics
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan
from rsl_rl.utils.log_writer import LogWriter
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import TendonEffortAction
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.osl_gait_library.config.dr import nominal_plant
from mjlab.tasks.osl_gait_library.mdp import TIGHT_ARC_WZ, GaitLibraryCommandCfg
from mjlab.tasks.osl_gait_library.rl.runner import GaitLibraryOnPolicyRunner
from mjlab.tasks.osl_mimic.rl.expected import auto_inference
from mjlab.tasks.osl_mimic.rl.runner import EvalPurpose, eval_mode
from mjlab.tasks.osl_split.config.env_cfgs import REWARD_NAME, osl_split_env_cfg
from mjlab.tasks.osl_split.mdp import osl_validation_group, split_mimic_reward

from .export import EXPORT_DIR, export_osl_policy
from .layout import AGENTS, ActionLayout
from .policy import HumanMode, OslMode, SplitInferencePolicy
from .ppo import OSL_OBS_GROUP, TwoAgentPPO
from .schedules import MODULES, ChannelSchedule, SplitSchedule
from .warmstart import (
  WarmStart,
  architecture_mismatches,
  check_compatible,
  compare_library_releases,
  env_cfg_settings,
  library_release,
  lineage_warnings,
  load_warm_start,
  provenance_lines,
)

OSL_JOINT_LABELS = ("knee", "ankle")
"""The OSL action rows in order (``OSL_TORQUE_JOINTS``, which the term preserves)."""
RETURN_WINDOW = 100
"""Completed episodes per agent behind ``Train/<agent>/mean_reward``: rsl_rl's."""
FAMILIES: dict[str, tuple[str, HumanMode, OslMode]] = {
  "mean": ("", "mean", "mean"),
  "expected": ("Expected", "expected", "mean"),
  "osl_sampled": ("OslSampled", "mean", "sampled"),
}
"""Evaluation families: ``name -> (prefix suffix, human mode, OSL mode)``. The
prefix is ``Validation`` or ``Stress`` plus the suffix (see
``SplitValidationCfg``)."""
TEACHER_FAMILIES: dict[str, tuple[str, HumanMode, OslMode]] = {
  "teacher": ("Teacher", "mean", "teacher"),
  "teacher_expected": ("TeacherExpected", "expected", "teacher"),
}
"""The zero-shot-only families (``validation.teacher``): the human and the teacher
head, i.e. the unified stage-1 policy while the human trunk is stage 1's, at its raw
mean and noise-averaged on every row (stage 1's ``Validation/`` and
``ValidationExpected/``). Stress validation plays the raw mean only."""
FAMILY_SETTINGS = (
  "inference",
  "zero_shot_inference",
  "stress_inference",
  "zero_shot_stress_inference",
)
"""The validation settings that name families."""
PEAK_KEYS = ("split_leg_contact_force",)
"""Diagnostics whose peak over an evaluation is reported too (``<key>_peak``):
the hardest leg strike, which a mean over steps and envs dilutes."""
VALIDATION_SUMMARY = (
  "early_termination_rate",
  "mean_episode_length",
  "mean_episode_return_human",
  "mean_episode_return_osl",
  "clf_V_norm",
  "err_site_abs",
  "split_err_osl_knee",
  "split_leg_contact",
  "twist_err_vx",
)
STRESS_SUMMARY = (
  "survival_rate",
  "mean_first_episode_s",
  "mean_first_episode_return_osl",
  "err_site_abs",
  "split_leg_contact",
  "twist_err_vx",
  "twist_err_wz",
)
SIDED_GROUPS = ("fwd_tight_left", "fwd_tight_right", "turn_left", "turn_right")
"""Validation groups by turn direction (``wz > 0`` turns left, counter-clockwise
from above): the prosthesis is the outer leg in a left turn, where hull strikes on
the biological leg concentrate."""
VIDEO_INFERENCE = ("auto", "mean", "expected")
"""The human's video actions (``video.inference``)."""
GIB = float(2**30)
WARM_START_HINT = (
  "Stage 2 starts from a stage-1 run: pass --agent.warm-start.path <stage-1 run "
  "directory, model_N.pt, or a file converted by `python -m "
  "mjlab.tasks.osl_split.scripts.split`>, or resume a stage-2 run with "
  "--agent.resume."
)


class AgentRewards:
  """Both agents' rewards for the step the env just took, ``(B, 2)`` in ``AGENTS``
  order (human, osl).

  The split ``mimic`` term returns the stage-1 (team) reward to mjlab and keeps
  each agent's imitation reward in ``channel_rewards``, times the term's weight
  here as the reward manager weighs the team reward. Every other active term
  (the twist reward) is additive: both agents earn it, times its
  ``additive_term_weights`` pair ``(human, osl)``, 1 each unless listed. The terms
  come from the manager's ``_step_reward``, each term's ``value x weight`` for the
  step, which is what the env's reward adds only without dt scaling; stage 1 runs
  without it, and a dt-scaled env is refused.

  Read it after ``env.step()``: the reward state survives the env's auto-resets.

  Args:
    env: The env whose reward manager to read.
    additive_term_weights: ``{term: (human, osl)}``; every key must be an active
      reward term other than ``mimic``.

  Attributes:
    term: The split reward term (the runner sets its group weights).
    additive: The additive terms with their ``(human, osl)`` multipliers.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    additive_term_weights: Mapping[str, Sequence[float]],
  ) -> None:
    if env.cfg.scale_rewards_by_dt:
      raise ValueError(
        "The agents' rewards add the reward manager's per-step term values, which "
        "sum to the env's reward only with scale_rewards_by_dt=False (stage 1's "
        "setting)."
      )
    manager = env.reward_manager
    names = list(manager.active_terms)
    if REWARD_NAME not in names:
      raise ValueError(f"The env has no {REWARD_NAME!r} reward term: {names}")
    term_cfg = manager.get_term_cfg(REWARD_NAME)
    if term_cfg.weight == 0.0:
      # The manager skips zero-weight terms, so the channels would never update.
      raise ValueError(f"The {REWARD_NAME!r} reward term needs a non-zero weight.")
    if not isinstance(getattr(term_cfg.func, "channel_rewards", None), torch.Tensor):
      raise TypeError(
        f"The {REWARD_NAME!r} reward term must be a split_mimic_reward (it has no "
        f"channel_rewards): {term_cfg.func!r}"
      )
    unknown = sorted(
      name for name in additive_term_weights if name not in names or name == REWARD_NAME
    )
    if unknown:
      additive = [name for name in names if name != REWARD_NAME]
      raise ValueError(
        f"additive_term_weights names {unknown}; the additive reward terms are "
        f"{additive}."
      )
    self.additive: dict[str, tuple[float, float]] = {}
    multipliers: list[tuple[float, float]] = []
    for name in names:
      if name == REWARD_NAME:
        multipliers.append((0.0, 0.0))  # Its channels stand in for it.
        continue
      pair = tuple(float(w) for w in additive_term_weights.get(name, (1.0, 1.0)))
      if len(pair) != len(AGENTS) or not all(math.isfinite(w) for w in pair):
        raise ValueError(
          f"additive_term_weights[{name!r}] must be a finite (human, osl) pair, got "
          f"{additive_term_weights[name]!r}."
        )
      self.additive[name] = (pair[0], pair[1])
      multipliers.append((pair[0], pair[1]))
    self.term = cast(split_mimic_reward, term_cfg.func)
    self._manager = manager
    self._weight = float(term_cfg.weight)
    self._multipliers = torch.tensor(
      multipliers, dtype=torch.float32, device=env.device
    )

  def __call__(self) -> torch.Tensor:
    step = self._manager._step_reward  # (B, terms): raw value x weight per term.
    additive = (step.unsqueeze(-1) * self._multipliers).sum(dim=1)
    return self.term.channel_rewards * self._weight + additive


class _AgentReturns:
  """Each agent's completed-episode returns, kept as rsl_rl's Logger keeps the
  team's (the last :data:`RETURN_WINDOW` episodes)."""

  def __init__(self, num_envs: int, device: str | torch.device) -> None:
    self.current = torch.zeros(num_envs, len(AGENTS), device=device)
    self.completed: dict[str, deque[float]] = {
      agent: deque(maxlen=RETURN_WINDOW) for agent in AGENTS
    }

  def add(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    self.current += rewards
    ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return
    finished = self.current[ids].cpu()
    for i, agent in enumerate(AGENTS):
      self.completed[agent].extend(finished[:, i].tolist())
    self.current[ids] = 0.0

  def means(self) -> dict[str, float]:
    return {agent: statistics.mean(r) for agent, r in self.completed.items() if r}


class SplitOnPolicyRunner(GaitLibraryOnPolicyRunner):
  """The stage-2 runner; see the module docstring.

  Construction builds the algorithm and, given ``warm_start.path``, applies the
  warm start. Without one the runner can still load a stage-2 checkpoint (as
  ``play.py`` and ``--agent.resume`` do), but cannot train from scratch.

  Attributes:
    stage_start: The global iteration of stage-local iteration 0: stage 1's last
      iteration plus one.
    lineage: The warm start's lineage (where stage 1 came from); None until a warm
      start or a stage-2 checkpoint gives one.
    reward_switch: Whether stage 2's reward type differs from stage 1's.
    schedule_iterations: The stage length every learning rate decays over.
    schedule: The freeze curriculum, learning rates and blend.
    channel_schedule: The reward channels' group weights over the stage.
  """

  alg: TwoAgentPPO  # pyright: ignore[reportIncompatibleVariableOverride]

  validation_group_keys = (
    *GaitLibraryOnPolicyRunner.validation_group_keys,
    "split_reward_human",
    "split_reward_osl",
    "split_leg_contact",
  )

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
  ) -> None:
    kwargs.pop("registry_name", None)  # Passed to tracking runners; unused here.
    for key in ("osl_actor", "osl_critic"):
      _strip_optional(train_cfg.get(key))
    super().__init__(env, train_cfg, log_dir, device)
    channels = train_cfg["channels"]
    self._additive_weights: dict[str, Sequence[float]] = dict(
      channels.get("additive_term_weights") or {}
    )
    self._rewards = AgentRewards(self.env.unwrapped, self._additive_weights)
    self._validation_rewards: AgentRewards | None = None
    self._stress_rewards: AgentRewards | None = None
    _check_families(self.validation_cfg)
    video = str(self.video_cfg.get("inference", "auto"))
    if video not in VIDEO_INFERENCE:
      raise ValueError(
        f"video.inference {video!r}: stage 2's videos play {VIDEO_INFERENCE} (the "
        "human's action; the OSL actor plays its mean)."
      )
    self._returns = _AgentReturns(self.env.num_envs, self.device)
    self.channel_schedule = ChannelSchedule.from_train_cfg(train_cfg)
    # The term refuses weights that leave one of its kernels without coordinates,
    # which only it can judge. Every weight of the anneal is a convex combination of
    # its two ends, so checking those now spares a refusal when the anneal ends.
    for weights in (self.channel_schedule.start, self.channel_schedule.end):
      self._rewards.term.check_group_weights(weights)
    self.stage_start = 0
    self.lineage: dict[str, Any] | None = None
    self._ready = False
    self._install_schedule(self._configured_schedule_iterations(), False)
    warm_start = train_cfg.get("warm_start") or {}
    if warm_start.get("path"):
      self.warm_start(
        load_warm_start(warm_start["path"], warm_start.get("checkpoint", ""))
      )

  ##
  # Warm start and schedules.
  ##

  def warm_start(self, ws: WarmStart) -> None:
    """Start stage 2 from ``ws``: the human actor, both critics (unless the reward
    switches), the return statistics, the iteration axis and the env's step
    counter.

    Stage 1's recorded end state is then compared with this run's settings
    (:meth:`_report_provenance`).

    Raises:
      ValueError: On a model or socket change, a warm start whose action layout
        or observation does not match this env, a network that takes stage-1
        weights but is configured otherwise than stage 1's (the message names
        the overrides that build stage 1's), or another CLF penalty weight than
        stage 1's without ``warm_start.allow_pen_w_mismatch``.
    """
    v = self.validation_cfg
    model, socket, reward = v["model"], v["socket"], v["reward"]
    reward_switch = check_compatible(ws, model=model, socket=socket, reward=reward)
    env = self.env.unwrapped
    layout = ActionLayout.from_action_manager(env.action_manager)
    if layout != ws.layout:
      raise ValueError(
        f"The warm start's actions {ws.layout.terms} differ from this env's "
        f"{layout.terms}."
      )
    obs_dim = int(ws.lineage["obs_dim"])
    if obs_dim != self.alg.human_actor.obs_dim:
      raise ValueError(
        f"The warm start's actor reads {obs_dim} observations; this env's human "
        f"observation has {self.alg.human_actor.obs_dim}."
      )
    _check_architecture(ws.lineage, self.cfg, reward_switch)
    self._check_pen_w(ws.lineage, reward_switch)
    self.alg.apply_warm_start(
      ws,
      reward_switch=reward_switch,
      num_envs=self.env.num_envs,
      num_steps=int(self.cfg["num_steps_per_env"]),
    )
    self.lineage = dict(ws.lineage)
    self.stage_start = int(ws.lineage["iter"]) + 1
    self.current_learning_iteration = self.stage_start
    env.common_step_counter = int(ws.lineage["common_step_counter"])
    self._install_schedule(self.schedule_iterations, reward_switch)
    self._ready = True

    variant = v.get("variant", "unicycle")
    if ws.lineage.get("variant") != variant:
      print(
        f"[WARN] The warm start was trained on the {ws.lineage.get('variant')!r} "
        f"variant; stage 2 trains on {variant!r}."
      )
    switch = (
      f", reward switch {ws.lineage['reward']} -> {reward}: fresh critics and "
      "return statistics"
      if reward_switch
      else ""
    )
    print(
      f"[INFO] Stage 2 starts at iteration {self.stage_start} from "
      f"{ws.lineage.get('stage1_checkpoint')} ({model}, {socket}{switch})."
    )
    self._report_provenance(ws.lineage)

  def _check_pen_w(self, lineage: Mapping[str, Any], reward_switch: bool) -> None:
    """Refuse a warm start trained with another CLF penalty weight, whose copied
    critics and return statistics would value another reward, unless
    ``warm_start.allow_pen_w_mismatch`` (or the reward switches: the critics then
    start fresh)."""
    stage1 = (lineage.get("stage1_env") or {}).get("clf_pen_w")
    stage2 = env_cfg_settings(getattr(self.env.unwrapped, "cfg", None))["clf_pen_w"]
    if stage1 is None or stage2 is None or math.isclose(stage1, stage2, abs_tol=1e-12):
      return
    text = (
      f"Stage 1 trained with the CLF penalty weight pen_w {stage1:g}; stage 2's is "
      f"{stage2:g}."
    )
    if reward_switch:
      print(f"[WARN] {text} The reward switches, so the critics start fresh anyway.")
      return
    allowed = bool((self.cfg.get("warm_start") or {}).get("allow_pen_w_mismatch"))
    if not allowed:
      raise ValueError(
        f"{text} The copied critics and return statistics would value another "
        "reward. Train stage 2 with stage 1's pen_w, or pass "
        "--agent.warm-start.allow-pen-w-mismatch True to start anyway."
      )
    print(
      f"[WARN] {text} Allowed (warm_start.allow_pen_w_mismatch): the copied "
      "critics and return statistics value another reward."
    )

  def _report_provenance(self, lineage: Mapping[str, Any]) -> None:
    """Print stage 1's recorded end state and how it compares with this run: the
    twist weight, root-deviation limit and library release, the human's rate
    against stage 1's last one, and the lineage's own warnings (a large human std,
    an unfinished stage-1 run)."""
    env = self.env.unwrapped
    lines = [f"[INFO] {line.strip()}" for line in provenance_lines(lineage)]
    stage1 = lineage.get("stage1_env")
    if stage1:
      stage2 = env_cfg_settings(getattr(env, "cfg", None))
      for key, label in (
        ("twist_weight", "twist weight"),
        ("root_deviation", "root-deviation limit"),
      ):
        a, b = stage1.get(key), stage2[key]
        if a is not None and b is not None and not math.isclose(a, b, abs_tol=1e-12):
          lines.append(
            f"[WARN] Stage 1 trained with the {label} {a:g}; stage 2 uses {b:g}."
          )
    else:
      lines.append(
        "[WARN] The warm start records no stage-1 env settings (converted before "
        "they were recorded, or without the run's params/env.yaml): the CLF "
        "penalty weight, twist weight and root-deviation limit are not compared. "
        "Convert the stage-1 run again, or pass its run directory to "
        "--agent.warm-start.path."
      )
    stage1_release = lineage.get("library_release") or library_release(
      lineage.get("library_dir")
    )
    compared = compare_library_releases(
      stage1_release, library_release(_library_dir(env))
    )
    if compared is not None:
      lines.append(f"[{compared[0]}] {compared[1]}")
    stage1_lr = lineage.get("stage1_lr")
    peak = self.schedule.human_peak_lr
    if stage1_lr is not None and peak > float(stage1_lr) * (1.0 + 1e-6):
      lines.append(
        f"[WARN] The human's peak rate {peak:.3g} exceeds stage 1's last rate "
        f"{float(stage1_lr):.3g}: Muon moves the warm human by about the rate per "
        "step, and a Loss/human/clip_fraction above about 0.6 means it moves too far."
      )
    lines += [f"[WARN] {warning}" for warning in lineage_warnings(lineage)]
    for line in lines:
      print(line, flush=True)

  def _configured_schedule_iterations(self) -> int:
    """``algorithm.schedule_iterations``, or the run length when it is 0."""
    iterations = int(self.cfg["algorithm"].get("schedule_iterations", 0) or 0)
    iterations = iterations or int(self.cfg["max_iterations"])
    if iterations < 1:
      raise ValueError(f"The stage needs at least one iteration, got {iterations}.")
    return iterations

  def _install_schedule(self, iterations: int, reward_switch: bool) -> None:
    """The current config's schedule over ``iterations``."""
    schedule = SplitSchedule.from_train_cfg(
      self.cfg, total_iterations=int(iterations), reward_switch=reward_switch
    )
    self._set_schedule(schedule, reward_switch)

  def _set_schedule(self, schedule: SplitSchedule, reward_switch: bool) -> None:
    self.schedule = schedule
    self.schedule_iterations = schedule.total_iterations
    self.reward_switch = bool(reward_switch)
    self.alg.set_schedule(schedule)

  def _configured_curriculum(
    self, schedule_iterations: int, reward_switch: bool
  ) -> tuple[
    SplitSchedule | None, ChannelSchedule, dict[str, Sequence[float]], str | None
  ]:
    """The curriculum the current config gives under ``reward_switch``: its
    schedule (None, with the reason, when the config cannot build one there), its
    channel schedule and its additive term weights.

    An ``algorithm.schedule_iterations`` of 0 means the run's own length, which a
    resumed run does not choose (its ``max_iterations`` counts the iterations it
    adds), so ``schedule_iterations``, the checkpoint's, stands in.
    """
    configured = int(self.cfg["algorithm"].get("schedule_iterations", 0) or 0)
    schedule, error = None, None
    try:
      schedule = SplitSchedule.from_train_cfg(
        self.cfg,
        total_iterations=configured or schedule_iterations,
        reward_switch=reward_switch,
      )
    except ValueError as err:
      error = str(err)
    additive: dict[str, Sequence[float]] = {
      name: tuple(float(w) for w in pair)
      for name, pair in (
        self.cfg["channels"].get("additive_term_weights") or {}
      ).items()
    }
    return schedule, ChannelSchedule.from_train_cfg(self.cfg), additive, error

  def _restore_curriculum(self, split: Mapping[str, Any]) -> None:
    """Install the curriculum a resumed run trains on.

    With ``resume_curriculum="checkpoint"`` (the default) it is the one the
    checkpoint was trained with: its schedule (stage length, freeze, warm-ups,
    learning rates, blend), reward switch, channel schedule and additive term
    weights. A resumed run continues the run it resumes (rsl_rl's resume). Its
    command line builds the config, and one that does not repeat the run's
    stage-2 flags would otherwise freeze an unfrozen human again or snap the OSL
    channel back to its start weights mid-stage. Every value the current config
    sets differently is listed in a warning; the checkpoint's wins.

    With ``resume_curriculum="config"`` the command line's curriculum replaces it,
    over ``algorithm.schedule_iterations`` (the checkpoint's stage length when 0),
    to retune or extend a run; the reward switch stays the checkpoint's. The
    differences are listed as applied, with each module's rate before and after
    at the first iteration the run trains, and a human the config freezes again
    is warned about.

    A checkpoint written before the curriculum was kept restores its stage length
    and reward switch only (the config's length with ``"config"``, when set).
    """
    reward_switch = bool(split.get("reward_switch", False))
    take_config = self.cfg.get("resume_curriculum", "checkpoint") == "config"
    configured = int(self.cfg["algorithm"].get("schedule_iterations", 0) or 0)
    if "schedule" not in split:
      iterations = int(split.get("schedule_iterations") or 0)
      if take_config and configured:
        iterations = configured
      iterations = iterations or self._configured_schedule_iterations()
      self._install_schedule(iterations, reward_switch)
      print(
        "[WARN] The checkpoint predates saved curricula: the current config's "
        "freeze, learning rates, blend and reward channels apply, over its "
        f"schedule of {iterations} iterations."
      )
      return
    schedule = SplitSchedule.from_saved(split["schedule"])
    channels = ChannelSchedule(**split["channel_schedule"])
    additive: dict[str, Sequence[float]] = {
      name: tuple(float(w) for w in pair)
      for name, pair in split["additive_term_weights"].items()
    }
    config_schedule, config_channels, config_additive, error = (
      self._configured_curriculum(schedule.total_iterations, reward_switch)
    )
    checkpoint = _curriculum(schedule, channels, additive)
    differences = _curriculum_differences(
      checkpoint, _curriculum(config_schedule, config_channels, config_additive)
    )
    if take_config:
      if config_schedule is None:
        raise ValueError(
          f"resume_curriculum='config', but the config cannot build a schedule: {error}"
        )
      k = self.alg.stage_iteration + 1  # The first iteration the run trains.
      lines = [*differences]
      lines += [
        f"{module} rate at k={k}: {schedule.lr(module, k):.4g} -> "
        f"{config_schedule.lr(module, k):.4g}"
        for module in MODULES
      ]
      self._install_curriculum(
        config_schedule, config_channels, config_additive, reward_switch
      )
      print(
        "[INFO] The resumed run takes the config's curriculum "
        "(resume_curriculum='config'):\n" + "\n".join(f"  {line}" for line in lines),
        flush=True,
      )
      if not schedule.human_actor_frozen(k) and config_schedule.human_actor_frozen(k):
        print(
          f"[WARN] The config freezes the human actor again at k={k} (until "
          f"{config_schedule.human_freeze}); the checkpoint had it learning.",
          flush=True,
        )
      return
    if error is not None:
      differences.insert(0, f"schedule: the config cannot build one ({error})")
    self._install_curriculum(schedule, channels, additive, reward_switch)
    if differences:
      print(
        "[WARN] The resumed run keeps its checkpoint's curriculum, which the "
        "current config does not repeat; the checkpoint's values win "
        "(--agent.resume-curriculum config takes the config's):\n"
        + "\n".join(f"  {line}" for line in differences),
        flush=True,
      )

  def _install_curriculum(
    self,
    schedule: SplitSchedule,
    channels: ChannelSchedule,
    additive: Mapping[str, Sequence[float]],
    reward_switch: bool,
  ) -> None:
    for weights in (channels.start, channels.end):
      self._rewards.term.check_group_weights(weights)
    self._rewards = AgentRewards(self.env.unwrapped, additive)
    # Rebuilt with these weights when next needed.
    self._validation_rewards = None
    self._stress_rewards = None
    self._additive_weights = dict(additive)
    self.channel_schedule = channels
    self._set_schedule(schedule, reward_switch)

  def _set_stage_iteration(self, k: int) -> None:
    """Freeze flags, learning rates and the blend for ``k``, and the channels'
    group weights."""
    self.alg.set_stage_iteration(k)
    self._rewards.term.set_group_weights(self.channel_schedule.weights(k))

  ##
  # Training loop: stage 1's, for two agents.
  ##

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    del init_at_random_ep_len  # musclemimic never randomizes episode lengths.
    if not self._ready:
      raise RuntimeError(f"Nothing to train from. {WARM_START_HINT}")
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    if total_it - self.stage_start > self.schedule_iterations:
      print(
        f"[INFO] Training to stage iteration {total_it - self.stage_start}, past "
        f"the schedule's {self.schedule_iterations}: from there the critics and "
        f"the OSL actor hold {self.schedule.lr_final_ratio:g} of their peak rates "
        f"and the human {self.schedule.human_lr_final_ratio:g} of its own."
      )
    interval = int(self.validation_cfg.get("interval", 0))
    stress_interval = int(self.validation_cfg.get("stress_interval", 0))
    save_interval = int(self.cfg["save_interval"])
    num_steps = int(self.cfg["num_steps_per_env"])
    check = self.cfg.get("check_for_nan", True)
    writer = self.logger.writer
    log_dir = str(self.logger.log_dir)
    # Zero-shot: the split policies (and the teacher) before any stage-2 update. An
    # interval of 0 turns that evaluation off, its zero-shot pass included.
    zero_shot = bool(self.validation_cfg.get("zero_shot", True))
    if zero_shot and start_it == self.stage_start and writer is not None:
      if interval > 0 and self._families("validation", zero_shot=True):
        self._evaluate(
          "zero_shot_validation", lambda i: self.validate(i, zero_shot=True), start_it
        )
      if stress_interval > 0 and self._families("stress", zero_shot=True):
        self._evaluate(
          "zero_shot_stress",
          lambda i: self.stress_validate(i, zero_shot=True),
          start_it,
        )

    frozen: tuple[bool, bool] | None = None
    for it in range(start_it, total_it):
      k = it - self.stage_start
      self._set_stage_iteration(k)
      now = (self.alg.human_actor_frozen, self.alg.osl_actor_frozen)
      if now != frozen:
        learning = [
          name
          for name, off in zip(("human actor", "OSL actor"), now, strict=True)
          if not off
        ]
        print(
          f"[Stage2] it={it} k={k}: critics"
          + "".join(f", {name}" for name in learning)
          + " learn.",
          flush=True,
        )
        frozen = now

      start = time.time()
      with torch.inference_mode():
        for _ in range(num_steps):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if check:
            check_nan(obs, rewards, dones)
          agent_rewards = self._rewards()
          obs, rewards, dones, agent_rewards = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
            agent_rewards.to(self.device),
          )
          self.alg.process_env_step(obs, agent_rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)
          self._returns.add(agent_rewards, dones)
        collect_time = time.time() - start
        start = time.time()
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      learn_time = time.time() - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.human_std,
        rnd_weight=None,
      )
      self._log_stage(it, k)

      if writer is not None and interval > 0 and (k + 1) % interval == 0:
        self._evaluate("validation", self.validate, it)
      if writer is not None and stress_interval > 0 and (k + 1) % stress_interval == 0:
        self._evaluate("stress", self.stress_validate, it)
      # Not at a launch's first iteration: that checkpoint would be the warm start
      # (or the resumed checkpoint) after one update, for minutes of video.
      periodic = save_interval > 0 and k % save_interval == 0 and it != start_it
      if writer is not None and periodic:
        self.save(os.path.join(log_dir, f"model_{it}.pt"))
        self._evaluate("video", self.record_video, it)

    if writer is not None:
      if total_it > start_it:  # Never a checkpoint of an iteration that did not run.
        path = f"model_{self.current_learning_iteration}.pt"
        self.save(os.path.join(log_dir, path))
        self._evaluate("video", self.record_video, self.current_learning_iteration)
      # Again after the last save and video: the first periodic one comes only at
      # k = save_interval, so a short run's in-loop values never include the video
      # env and its renderer.
      for tag, value in self._gpu_memory().items():
        writer.add_scalar(tag, value, self.current_learning_iteration)
      self.close_video_env()
      flush = getattr(writer, "flush", None)
      if callable(flush):
        flush()  # TensorBoard's writer only flushes every 10 s on its own.
      self.logger.stop_logging_writer()

  def _log_stage(self, it: int, k: int) -> None:
    """The stage-2 scalars rsl_rl's Logger does not write."""
    writer = self.logger.writer
    if writer is None:
      return
    alg = self.alg
    scalars: dict[str, float] = {}
    for agent, mean in self._returns.means().items():
      scalars[f"Train/{agent}/mean_reward"] = mean
    for module, lr in alg.lrs.items():
      scalars[f"LR/{module}"] = lr
    osl_std = alg.osl_std
    scalars["Policy/human/mean_std"] = float(alg.human_std.mean())
    scalars["Policy/osl/mean_std"] = float(osl_std.mean())
    for joint, std in zip(OSL_JOINT_LABELS, osl_std.tolist(), strict=False):
      scalars[f"Policy/osl/std_{joint}"] = std
    scalars["Stage2/iteration"] = k
    scalars["Stage2/human_frozen"] = float(alg.human_actor_frozen)
    scalars["Stage2/osl_actor_frozen"] = float(alg.osl_actor_frozen)
    scalars["Stage2/blend_beta"] = alg.blend_beta
    scalars["Stage2/reward_switch"] = float(self.reward_switch)
    for module, skipped in alg.skipped_steps.items():
      scalars[f"Stage2/skipped_steps/{module}"] = skipped
    term = self._rewards.term
    shares = term.effective_shares()
    for channel, groups in term.group_weights.items():
      for group, weight in groups.items():
        scalars[f"Channels/{channel}/{group}"] = weight
        scalars[f"Channels/{channel}/share_{group}"] = shares[channel][group]
    samples = (k + 1) * int(self.cfg["num_steps_per_env"]) * self.env.num_envs
    scalars["Train/total_env_steps"] = self._stage1_env_steps() + samples
    if alg.last_teacher_rms is not None:
      scalars["Train/osl/teacher_rms"] = alg.last_teacher_rms
    for key, value in alg.policy_stats.items():
      scalars[f"Policy/{key}"] = value
    for key, value in alg.critic_stats.items():
      scalars[f"Critic/{key}"] = value
    scalars.update(self._gpu_memory())
    for tag, value in scalars.items():
      writer.add_scalar(tag, value, it)

  def _gpu_memory(self) -> dict[str, float]:
    """GiB on the training device: PyTorch's peak allocation so far and its
    reservation, and the whole device's use (warp, other processes and the desktop
    included). Empty off CUDA."""
    device = torch.device(self.device)
    if device.type != "cuda" or not torch.cuda.is_available():
      return {}
    try:
      free, total = torch.cuda.mem_get_info(device)
      return {
        "Perf/gpu_max_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "Perf/gpu_reserved_gib": torch.cuda.memory_reserved(device) / GIB,
        "Perf/gpu_device_used_gib": (total - free) / GIB,
      }
    except Exception:  # noqa: BLE001 - logging must never end a run.
      return {}

  def _evaluate(self, name: str, evaluation: Callable[[int], Any], it: int) -> None:
    """Run an in-training evaluation at ``it``, guarded and timed.

    A failure (running out of GPU memory, say) prints a warning and training goes
    on, as stage 1 guards its validations; the failed attempt's memory is then
    released. The wall time is logged as ``Perf/<name>_time``, which rsl_rl's own
    timing and ETA leave out.
    """
    start = time.time()
    failed = False
    try:
      evaluation(it)
    except Exception as err:  # noqa: BLE001
      print(f"[WARN] {name} at it={it} failed (training continues): {err!r}")
      failed = True
    if failed:
      # Only now: inside the handler the traceback still holds the failed frames (a
      # half-built env, the pass's tensors), and envs hold reference cycles.
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    writer = self.logger.writer
    if writer is not None:
      writer.add_scalar(f"Perf/{name}_time", time.time() - start, it)

  def _stage1_env_steps(self) -> int:
    return int((self.lineage or {}).get("env_steps") or 0)

  ##
  # Validation.
  ##

  def _eval_env_cfg(self, purpose: EvalPurpose) -> ManagerBasedRlEnvCfg:
    """An evaluation env's config, made to score and observe what training does.

    Stress validation plays a copy of the training config itself: its OSL sensor
    noise, bias and lags, every randomization and every command-line override,
    without the run's seed (an env built with a seed re-seeds every generator, so
    its draws would follow the run's seed rather than ``validation.stress_seed``,
    and training's numpy, Python and warp generators would be reset mid-run).
    Validation is the task's validation mode, and a video its play mode on
    validation's nominal plant, each with the training run's motion library and
    its settings (:meth:`_inherit_training_settings`: the simulation, the
    observation groups with their corruption off, and every shared reward term's
    weight and params, the channels' settings included), then the training OSL
    observation read exactly (:func:`~..mdp.osl_validation_group`) at the middle
    of its training lag, so a changed history, sensor switch or lag reaches the
    OSL actor in evaluation too.
    """
    train = self.env.unwrapped.cfg
    if purpose == "stress":
      cfg = deepcopy(train)
      cfg.seed = None
      return cfg
    v = self.validation_cfg
    cfg = osl_split_env_cfg(v["model"], v["socket"], v["reward"], eval_mode(purpose))
    if purpose == "video":
      nominal_plant(cfg)
    train_motion = train.commands["motion"]
    motion = cfg.commands["motion"]
    assert isinstance(train_motion, GaitLibraryCommandCfg)
    assert isinstance(motion, GaitLibraryCommandCfg)
    motion.library_dir = train_motion.library_dir
    self._inherit_training_settings(cfg)
    train_osl = train.observations[OSL_OBS_GROUP]
    cfg.observations[OSL_OBS_GROUP] = osl_validation_group(
      train_osl, _middle_lag(train_osl)
    )
    return cfg

  def _build_eval_env(
    self, cfg: ManagerBasedRlEnvCfg, render: bool = False
  ) -> RslRlVecEnvWrapper:
    vec = super()._build_eval_env(cfg, render)
    _check_same_observations(self.env.unwrapped, vec.unwrapped)
    return vec

  def _video_inference(self, env: ManagerBasedRlEnv) -> str:
    """The human's video action (``video.inference``): ``"auto"`` is stage 1's
    rule on the human's muscle std (:func:`~mjlab.tasks.osl_mimic.rl.expected.
    auto_inference`), so the raw mean at the recipe's 0.07."""
    mode = str(self.video_cfg.get("inference", "auto"))
    if mode != "auto":
      return mode
    manager = env.action_manager
    columns = [
      column
      for term, cols in self.alg.layout.agent_terms("human")
      if isinstance(manager.get_term(term), TendonEffortAction)
      for column in range(cols.start, cols.stop)
    ]
    return auto_inference(self.alg.human_std[columns])

  @contextmanager
  def _video_policy(
    self, env: ManagerBasedRlEnv
  ) -> Iterator[Callable[[TensorDict], torch.Tensor]]:
    """The human's ``video.inference`` action and the OSL actor's mean; both
    observation normalizers stay frozen."""
    mode = cast(HumanMode, self._video_inference(env))
    yield self.alg.inference_policy(mode, env.action_manager)

  def _families(self, kind: str, zero_shot: bool) -> list[str]:
    """The families a ``"validation"`` or ``"stress"`` evaluation plays: its
    ``validation`` setting (``inference``, ``stress_inference``, or their
    ``zero_shot_`` forms), then at zero-shot the teacher's (``validation.teacher``:
    raw mean and noise-averaged in validation, raw mean in stress)."""
    setting = "inference" if kind == "validation" else "stress_inference"
    if zero_shot:
      setting = f"zero_shot_{setting}"
    families = list(_family_names(self.validation_cfg.get(setting)))
    if zero_shot and self.validation_cfg.get("teacher", True):
      families.append("teacher")
      if kind == "validation":
        families.append("teacher_expected")
    return families

  def _policy(self, family: str, env: ManagerBasedRlEnv) -> SplitInferencePolicy:
    _, human_mode, osl_mode = _family(family)
    return self.alg.inference_policy(human_mode, env.action_manager, osl_mode)

  def validate(self, it: int, zero_shot: bool = False) -> dict[str, float]:
    """Validate at global iteration ``it``: the ``validation.inference`` families,
    or at zero-shot ``validation.zero_shot_inference`` plus the teacher's. Returns
    every metric under its family's prefix (``Validation/...``,
    ``ValidationExpected/...``, ``ValidationOslSampled/...``,
    ``ValidationTeacher/...``, ``ValidationTeacherExpected/...``). The OSL-sampled
    family is seeded (``validation.seed``); training's random state is left
    untouched."""
    if self._validation_env is None:
      self._validation_env = self._make_validation_env()
      self._validation_rewards = None
    if self._validation_rewards is None:
      self._validation_rewards = AgentRewards(
        self._validation_env.unwrapped, self._additive_weights
      )
    vec, rewards = self._validation_env, self._validation_rewards
    env = vec.unwrapped
    k = max(it - self.stage_start, 0)
    rewards.term.set_group_weights(self.channel_schedule.weights(k))
    seed = int(self.validation_cfg.get("seed", 0))
    results: dict[str, float] = {}
    for family in self._families("validation", zero_shot):
      policy = self._policy(family, env)
      sampled = policy.osl_mode == "sampled"
      with _seeded(seed if sampled else None, self.device, env.device):
        metrics = self._validation_pass(vec, policy, rewards)
      prefix = "Validation" + _suffix(family)
      results.update(self._write_family(prefix, metrics, it, k, VALIDATION_SUMMARY))
    return results

  def stress_validate(self, it: int, zero_shot: bool = False) -> dict[str, float]:
    """Stress-validate at global iteration ``it`` (see the module docstring): the
    ``validation.stress_inference`` families, or at zero-shot
    ``validation.zero_shot_stress_inference`` plus the teacher, each seeded alike
    (``validation.stress_seed``: every checkpoint meets the same commands, pushes
    and draws); training's random state is left untouched. Returns every metric
    under its family's prefix (``Stress/...``, ``StressOslSampled/...``,
    ``StressTeacher/...``)."""
    seed = int(self.validation_cfg.get("stress_seed", 0))
    device = self.env.unwrapped.device
    if self._stress_env is None:
      # Its physical draws come from the stress seed, not training's random state.
      with _seeded(seed, device):
        self._stress_env = self._make_stress_env()
      self._stress_rewards = None
    if self._stress_rewards is None:
      self._stress_rewards = AgentRewards(
        self._stress_env.unwrapped, self._additive_weights
      )
    vec, rewards = self._stress_env, self._stress_rewards
    env = vec.unwrapped
    k = max(it - self.stage_start, 0)
    rewards.term.set_group_weights(self.channel_schedule.weights(k))
    results: dict[str, float] = {}
    for family in self._families("stress", zero_shot):
      policy = self._policy(family, env)
      with _seeded(seed, self.device, env.device):
        metrics = self._split_stress_pass(vec, policy, rewards)
      prefix = "Stress" + _suffix(family)
      results.update(self._write_family(prefix, metrics, it, k, STRESS_SUMMARY))
    return results

  def _write_family(
    self,
    prefix: str,
    metrics: Mapping[str, float],
    it: int,
    k: int,
    summary: Sequence[str],
  ) -> dict[str, float]:
    writer = self.logger.writer
    results: dict[str, float] = {}
    for key, value in metrics.items():
      results[f"{prefix}/{key}"] = value
      if writer is not None:
        writer.add_scalar(f"{prefix}/{key}", value, it)
    print(
      f"[{prefix}] it={it} k={k} "
      + " ".join(f"{key}={metrics[key]:.4g}" for key in summary if key in metrics),
      flush=True,
    )
    return results

  def _split_stress_pass(
    self,
    vec: RslRlVecEnvWrapper,
    policy: SplitInferencePolicy,
    rewards: AgentRewards,
  ) -> dict[str, float]:
    """Every env's first episode under ``policy``: stage 1's stress pass, plus each
    agent's first-episode return.

    ``survival_rate`` is the share of envs whose first episode was not terminated
    early, ``mean_first_episode_s`` its mean length (s) and
    ``mean_first_episode_return[_<agent>]`` its mean return; the diagnostics are
    averaged over the envs still in their first episode at each step, and those of
    :data:`PEAK_KEYS` also reported at their peak there (``<key>_peak``). The
    policy keeps both observation normalizers frozen, so no snapshot is taken.
    """
    env = vec.unwrapped
    num_steps = int(self.validation_cfg.get("stress_num_steps", 1000))
    terms = self._diagnostic_reward_terms(env, rewards)
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    failed = torch.zeros(n, dtype=torch.bool, device=env.device)
    length = torch.zeros(n, device=env.device)
    team_return = torch.zeros(n, device=env.device)
    agent_return = torch.zeros(n, len(AGENTS), device=env.device)
    sums: dict[str, torch.Tensor] = {}
    peaks: dict[str, torch.Tensor] = {}
    weighted_steps = 0
    with torch.inference_mode():
      obs, _ = vec.reset()
      for _ in range(num_steps):
        actions = policy(obs.to(self.device))
        obs, reward, dones, extras = vec.step(actions.to(vec.device))
        agent = rewards()
        truncated = extras["time_outs"].bool()
        terminated = dones.bool() & ~truncated
        weight = alive.float()
        count = weight.sum()
        if bool(count > 0):
          weighted_steps += 1
          for term in terms:
            for key, value in term.components.items():
              mean = (value.float() * weight).sum() / count
              sums[key] = sums[key] + mean if key in sums else mean
              if key in PEAK_KEYS:
                peak = (value.float().reshape(n, -1) * weight.unsqueeze(-1)).amax()
                peaks[key] = torch.maximum(peaks[key], peak) if key in peaks else peak
        team_return += reward * weight
        agent_return += agent * weight.unsqueeze(-1)
        length += weight
        failed |= alive & terminated
        alive &= ~(terminated | truncated)
    metrics = {
      key: float(value) / max(weighted_steps, 1) for key, value in sums.items()
    }
    metrics.update({f"{key}_peak": float(value) for key, value in peaks.items()})
    metrics["survival_rate"] = 1.0 - float(failed.float().mean())
    metrics["mean_first_episode_s"] = float(length.mean()) * float(env.step_dt)
    metrics["mean_first_episode_return"] = float(team_return.mean())
    for i, name in enumerate(AGENTS):
      metrics[f"mean_first_episode_return_{name}"] = float(agent_return[:, i].mean())
    return metrics

  def _diagnostic_reward_terms(
    self, env: ManagerBasedRlEnv, rewards: AgentRewards
  ) -> list[Any]:
    """Every class reward term that keeps per-step diagnostics, the split one
    first."""
    terms: list[Any] = [rewards.term]
    for name in env.reward_manager.active_terms:
      func = env.reward_manager.get_term_cfg(name).func
      if name != REWARD_NAME and isinstance(getattr(func, "components", None), dict):
        terms.append(func)
    return terms

  def _validation_groups(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Stage 1's clip families, plus the tight forward arcs and the pivots by turn
    direction (:data:`SIDED_GROUPS`; stage 1's ``fwd_tight`` and ``turn`` are each
    the union of a pair)."""
    groups = super()._validation_groups(env)
    cmd = env.command_manager.get_term("motion")
    twists = getattr(cmd, "twists", None)
    num_clips = int(getattr(cmd, "num_clips", 0) or 0)
    if twists is None or num_clips < 1:
      return groups
    twists = torch.as_tensor(twists, device=env.device)
    vx, wz = twists[:, 0], twists[:, 2]
    eps = 1e-6
    forward, pivot = vx > eps, vx.abs() < eps
    sided = (
      forward & (wz > TIGHT_ARC_WZ + eps),
      forward & (wz < -(TIGHT_ARC_WZ + eps)),
      pivot & (wz > eps),
      pivot & (wz < -eps),
    )
    clip = torch.arange(env.num_envs, device=env.device) % num_clips
    for name, of_clip in zip(SIDED_GROUPS, sided, strict=True):
      groups[name] = of_clip[clip]
    return groups

  def _validation_pass(
    self,
    vec: RslRlVecEnvWrapper,
    policy: SplitInferencePolicy,
    rewards: AgentRewards,
  ) -> dict[str, float]:
    """One validation run of ``policy``: stage 1's loop and metrics, plus each
    agent's completed-episode return (``mean_episode_return_<agent>``) and the
    peak of :data:`PEAK_KEYS` (``<key>_peak``).

    The policy keeps both observation normalizers frozen, so no snapshot is taken.
    """
    env = vec.unwrapped
    num_steps = int(self.validation_cfg.get("num_steps", 0))
    if num_steps <= 0:
      num_steps = self._default_validation_steps(env)
    terms = self._diagnostic_reward_terms(env, rewards)

    n = env.num_envs
    sums: dict[str, torch.Tensor] = {}
    peaks: dict[str, torch.Tensor] = {}
    episode_return = torch.zeros(n, device=env.device)
    agent_return = torch.zeros(n, len(AGENTS), device=env.device)
    episode_length = torch.zeros(n, device=env.device)
    finished_returns: list[torch.Tensor] = []
    finished_agent_returns: list[torch.Tensor] = []
    finished_lengths: list[torch.Tensor] = []
    num_done = 0
    num_early = 0
    groups = {g: m for g, m in self._validation_groups(env).items() if bool(m.any())}
    # Group means as weighted sums: a boolean index would sync with the host for
    # every key and group at every step.
    group_weights = {g: m.float() for g, m in groups.items()}
    group_sizes = {g: w.sum() for g, w in group_weights.items()}
    group_done = dict.fromkeys(groups, 0)
    group_early = dict.fromkeys(groups, 0)
    group_lengths: dict[str, list[torch.Tensor]] = {g: [] for g in groups}

    # Reset inside inference mode too: stepping there turns the env's state tensors
    # into inference tensors, which a later reset outside it could not update.
    with torch.inference_mode():
      obs, _ = vec.reset()
      for _ in range(num_steps):
        actions = policy(obs.to(self.device))
        obs, reward, dones, extras = vec.step(actions.to(vec.device))
        agent = rewards()
        truncated = extras["time_outs"].bool()
        terminated = dones.bool() & ~truncated
        for term in terms:
          for key, value in term.components.items():
            mean = value.float().mean()
            sums[key] = sums[key] + mean if key in sums else mean
            if key in PEAK_KEYS:
              peak = value.float().amax()
              peaks[key] = torch.maximum(peaks[key], peak) if key in peaks else peak
            if key not in self.validation_group_keys:
              continue
            per_env = value.float().reshape(n, -1).mean(dim=1)
            for group, weight in group_weights.items():
              name = f"{key}/{group}"
              mean = (per_env * weight).sum() / group_sizes[group]
              sums[name] = sums[name] + mean if name in sums else mean
        episode_return += reward
        agent_return += agent
        episode_length += 1
        done = terminated | truncated
        if done.any():
          num_done += int(done.sum())
          num_early += int(terminated.sum())
          finished_returns.append(episode_return[done].clone())
          finished_agent_returns.append(agent_return[done].clone())
          finished_lengths.append(episode_length[done].clone())
          for group, mask in groups.items():
            group_done[group] += int((done & mask).sum())
            group_early[group] += int((terminated & mask).sum())
            group_lengths[group].append(episode_length[done & mask].clone())
          episode_return[done] = 0.0
          agent_return[done] = 0.0
          episode_length[done] = 0.0

    metrics = {key: float(value) / num_steps for key, value in sums.items()}
    metrics.update({f"{key}_peak": float(value) for key, value in peaks.items()})
    metrics["early_termination_rate"] = num_early / num_done if num_done else 0.0
    if finished_lengths:
      metrics["mean_episode_length"] = float(torch.cat(finished_lengths).mean())
      metrics["mean_episode_return"] = float(torch.cat(finished_returns).mean())
      returns = torch.cat(finished_agent_returns)
      for i, agent in enumerate(AGENTS):
        metrics[f"mean_episode_return_{agent}"] = float(returns[:, i].mean())
    for group in groups:
      done_g = group_done[group]
      metrics[f"early_termination_rate/{group}"] = (
        group_early[group] / done_g if done_g else 0.0
      )
      lengths = torch.cat(group_lengths[group]) if group_lengths[group] else None
      if lengths is not None and lengths.numel():
        metrics[f"mean_episode_length/{group}"] = float(lengths.mean())
    return metrics

  def close(self) -> None:
    super().close()
    self._validation_rewards = None
    self._stress_rewards = None

  ##
  # Checkpoints, inference and export.
  ##

  def save(self, path: str, infos: dict | None = None) -> None:
    """Checkpoint everything (the four networks, optimizers, return statistics and
    split state, the curriculum included), then export the OSL actor next to it
    if ``export_osl_onnx``."""
    env = self.env.unwrapped
    infos = {
      **(infos or {}),
      "env_state": {"common_step_counter": env.common_step_counter},
      "split": {
        "stage_start": self.stage_start,
        "lineage": self.lineage,
        "schedule_iterations": self.schedule_iterations,
        "reward_switch": self.reward_switch,
        **_curriculum(self.schedule, self.channel_schedule, self._additive_weights),
      },
    }
    saved = self.alg.save()
    saved["iter"] = self.current_learning_iteration
    saved["infos"] = infos
    torch.save(saved, path)
    upload = bool(self.cfg.get("upload_model", False))
    if upload:
      self.logger.save_model(path, self.current_learning_iteration)
    if self.cfg.get("export_osl_onnx", True):
      name = f"osl_policy_{self.current_learning_iteration}"
      try:
        onnx_path = export_osl_policy(
          self.alg, env, Path(path).parent / EXPORT_DIR, name, self.lineage
        )
      except Exception as err:
        # The checkpoint holds the actor; a failed export must not end the run.
        print(f"[WARN] OSL policy export failed (training continues): {err}")
        return
      writer = self.logger.writer
      if upload and isinstance(writer, LogWriter):
        writer.save_file(str(onnx_path))
        writer.save_file(str(onnx_path.with_suffix(".json")))

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Restore a stage-2 checkpoint, with the stage start, the lineage and the
    env's step counter.

    ``load_cfg=None`` (``--agent.resume``) restores everything a resumed run trains
    on: every network, optimizer and statistic, the reward switch and the whole
    curriculum (:meth:`_restore_curriculum`). Training resumes at the iteration
    after the checkpoint's.

    A ``load_cfg`` that asks for neither ``critic`` nor ``optimizer`` (``play.py``'s
    ``{"actor": True}``) loads the two actors and the teacher head only, all
    inference needs. ``play.py`` builds the runner from the task's registered
    config, without the run's overrides, so the critics need not fit (a run
    trained with ``osl_critic_obs="full+osl"`` has a wider OSL critic). Without
    the run's critics, optimizers and statistics the runner cannot train.

    Raises:
      ValueError: For a stage-1 checkpoint, a warm-start file or anything else
        that is not a stage-2 checkpoint.
    """
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    _check_stage2_checkpoint(loaded, path)
    actors_only = _actors_only(load_cfg)
    self.alg.load(loaded, strict, actors_only=actors_only)
    infos = dict(loaded.get("infos") or {})
    split = dict(infos.get("split") or {})
    self.stage_start = int(split.get("stage_start", 0))
    lineage = split.get("lineage")
    self.lineage = dict(lineage) if lineage is not None else None
    env_state = infos.get("env_state") or {}
    if "common_step_counter" in env_state:
      counter = int(env_state["common_step_counter"])
      self.env.unwrapped.common_step_counter = counter
    if actors_only:
      self._ready = False
      return infos
    self._restore_curriculum(split)
    self.current_learning_iteration = int(loaded["iter"]) + 1
    self._ready = True
    return infos

  def get_inference_policy(  # type: ignore[override]
    self, device: str | None = None
  ) -> SplitInferencePolicy:
    """Both actors' mean actions over the env's whole action (what ``play.py``
    runs). The policy shares the actors; moving it moves them."""
    self.alg.eval_mode()
    policy = self.alg.get_policy()
    return policy if device is None else policy.to(device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export the OSL actor, the deployable one (:mod:`.export`): ``filename``
    and its ``.json`` metadata in ``path``."""
    del verbose
    export_osl_policy(
      self.alg, self.env.unwrapped, Path(path), Path(filename).stem, self.lineage
    )

  def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
    del path, filename
    raise NotImplementedError(
      "Stage 2 exports the OSL actor to ONNX (export_policy_to_onnx); the human "
      "actor stays in simulation."
    )


##
# Helpers.
##


def _family_names(families: Any) -> tuple[str, ...]:
  if not families:
    return ()
  return (families,) if isinstance(families, str) else tuple(families)


def _check_families(validation_cfg: Mapping[str, Any]) -> None:
  """Every family the validation settings name must be one of :data:`FAMILIES`
  (the teacher's play at zero-shot only, by ``validation.teacher``)."""
  for setting in FAMILY_SETTINGS:
    unknown = [
      f for f in _family_names(validation_cfg.get(setting)) if f not in FAMILIES
    ]
    if unknown:
      hint = (
        " The teacher's families play at zero-shot only (validation.teacher)."
        if any(f in TEACHER_FAMILIES for f in unknown)
        else ""
      )
      raise ValueError(
        f"validation.{setting} names unknown families {unknown}; stage 2's are "
        f"{list(FAMILIES)} (stage 1's 'sampled' is 'osl_sampled' here).{hint}"
      )


def _family(family: str) -> tuple[str, HumanMode, OslMode]:
  return TEACHER_FAMILIES[family] if family in TEACHER_FAMILIES else FAMILIES[family]


def _suffix(family: str) -> str:
  return _family(family)[0]


@contextmanager
def _seeded(seed: int | None, *devices: str | torch.device) -> Iterator[None]:
  """Seed torch's generators (the CPU's and those of the CUDA ``devices``) for the
  block and restore them after; nothing when ``seed`` is None."""
  if seed is None:
    yield
    return
  cuda = sorted(
    {torch.device(d).index or 0 for d in devices if torch.device(d).type == "cuda"}
  )
  with torch.random.fork_rng(devices=cuda):
    torch.manual_seed(seed)
    yield


def _middle_lag(group: ObservationGroupCfg) -> int:
  """The middle of the OSL buses' training lag range: validation's fixed lag
  (``OslObsCfg.validation_delay_steps``'s rule)."""
  for name in ("actpack", "loadcell"):
    term = group.terms.get(name)
    if term is not None:
      return (int(term.delay_min_lag) + int(term.delay_max_lag)) // 2
  return 0


def _strip_optional(model_cfg: dict | None) -> None:
  """Drop the None-valued optional keys rsl_rl's model configs carry, as
  ``MjlabOnPolicyRunner`` does for the actor and the critic."""
  if not model_cfg:
    return
  for key in ("cnn_cfg", "distribution_cfg"):
    if model_cfg.get(key) is None:
      model_cfg.pop(key, None)
  if model_cfg.get("rnn_type") is None:
    for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      model_cfg.pop(key, None)


def _actors_only(load_cfg: Mapping[str, Any] | None) -> bool:
  """Whether rsl_rl's ``load_cfg`` asks for less than a resume: neither ``critic``
  nor ``optimizer`` (``play.py``'s ``{"actor": True}``). None asks for all."""
  if load_cfg is None:
    return False
  return not (load_cfg.get("critic") or load_cfg.get("optimizer"))


def _check_architecture(
  lineage: Mapping[str, Any], train_cfg: Mapping[str, Any], reward_switch: bool
) -> None:
  """Every stage-2 network that takes stage-1 weights must be built as stage 1's
  was (:func:`~.warmstart.architecture_mismatches`): the human actor always, both
  critics unless the reward switches (they then start fresh, with only the
  stage-1 observation statistics), the OSL critic unless it starts fresh
  (``algorithm.osl_critic_init="fresh"``). A strict load catches a changed width,
  not a changed activation, which would load and compute something else.

  Raises:
    ValueError: Naming each difference and the ``--agent`` overrides that build
      stage 1's architecture.
  """
  networks = [("actor", "actor_cfg")]
  if not reward_switch:
    networks.append(("critic", "critic_cfg"))
    if train_cfg["algorithm"].get("osl_critic_init", "stage1") != "fresh":
      networks.append(("osl_critic", "critic_cfg"))
  lines: list[str] = []
  flags: list[str | None] = []
  for module, key in networks:
    stage1 = lineage[key]
    for mismatch in architecture_mismatches(stage1, train_cfg[module]):
      lines.append(f"  {module}.{mismatch}")
      flags.append(_override(module, mismatch.split(":", 1)[0], stage1))
  if not lines:
    return
  given = [flag for flag in flags if flag is not None]
  message = [
    f"The warm start ({lineage.get('stage1_checkpoint')}) holds networks of "
    "another architecture than this stage-2 config builds: their weights would "
    "load and compute something else.",
    *lines,
  ]
  if given:
    message.append("Build stage 1's architecture: " + " ".join(given))
  if len(given) < len(flags):
    message.append("The command line cannot set the rest: change the task's config.")
  raise ValueError("\n".join(message))


def _override(module: str, key: str, stage1: Mapping[str, Any]) -> str | None:
  """The ``--agent`` flag that sets ``module``'s dotted ``key`` to stage 1's value,
  or None when stage 1 records none or it is not a single value (a whole
  ``distribution_cfg``)."""
  value: Any = stage1
  for part in key.split("."):
    if not isinstance(value, Mapping) or part not in value:
      return None
    value = value[part]
  if value is None or isinstance(value, Mapping):
    return None
  if isinstance(value, (list, tuple)):
    value = tuple(value)  # tyro takes Python syntax for collections.
  text = value if isinstance(value, str) else repr(value)
  flag = ".".join(part.replace("_", "-") for part in ("agent", module, *key.split(".")))
  return f"--{flag} {shlex.quote(text)}"


_ADDITIVE_DEFAULT = (1.0, 1.0)
"""An additive term's ``(human, osl)`` multipliers when ``additive_term_weights``
does not list it (:class:`AgentRewards`)."""
_UNSET = object()
"""A key one curriculum lacks (:func:`_curriculum_differences`)."""


def _curriculum(
  schedule: SplitSchedule | None,
  channels: ChannelSchedule,
  additive: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
  """A curriculum as checkpoints keep it, in plain types: the schedule's fields
  (absent without one), the channel schedule and the additive term weights."""
  curriculum: dict[str, Any] = {}
  if schedule is not None:
    curriculum["schedule"] = asdict(schedule)
  curriculum["channel_schedule"] = {
    "start": {channel: dict(groups) for channel, groups in channels.start.items()},
    "end": {channel: dict(groups) for channel, groups in channels.end.items()},
    "anneal_start": channels.anneal_start,
    "anneal_iterations": channels.anneal_iterations,
  }
  curriculum["additive_term_weights"] = {
    name: [float(w) for w in pair] for name, pair in additive.items()
  }
  return curriculum


def _curriculum_differences(
  checkpoint: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
  """``"<key>: checkpoint <value>, config <value>"`` for every dotted key of two
  :func:`_curriculum` dicts whose values differ, over the sections both have. An
  additive term that one side does not list counts as :data:`_ADDITIVE_DEFAULT`
  there, as the runner counts it."""
  names = sorted(
    {*checkpoint["additive_term_weights"], *config["additive_term_weights"]}
  )

  def flat(curriculum: Mapping[str, Any]) -> dict[str, Any]:
    additive = curriculum["additive_term_weights"]
    return _flat(
      {
        **{key: curriculum[key] for key in checkpoint if key in config},
        "additive_term_weights": {
          name: list(additive.get(name, _ADDITIVE_DEFAULT)) for name in names
        },
      }
    )

  ours, theirs = flat(checkpoint), flat(config)
  return [
    f"{key}: checkpoint {_shown(ours, key)}, config {_shown(theirs, key)}"
    for key in dict.fromkeys([*ours, *theirs])
    if ours.get(key, _UNSET) != theirs.get(key, _UNSET)
  ]


def _shown(flat: Mapping[str, Any], key: str) -> str:
  return repr(flat[key]) if key in flat else "unset"


def _flat(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
  """``tree``'s leaves under dotted keys."""
  flat: dict[str, Any] = {}
  for key, value in tree.items():
    if isinstance(value, Mapping):
      flat.update(_flat(value, f"{prefix}{key}."))
    else:
      flat[f"{prefix}{key}"] = value
  return flat


def _check_stage2_checkpoint(loaded: Any, path: str) -> None:
  if isinstance(loaded, dict) and "human_actor_state_dict" in loaded:
    return
  if isinstance(loaded, dict) and "format" in loaded:
    raise ValueError(
      f"{path} is a stage-2 warm-start file, not a checkpoint: start a run from it "
      f"with --agent.warm-start.path {path}."
    )
  if isinstance(loaded, dict) and "actor_state_dict" in loaded:
    raise ValueError(
      f"{path} is a stage-1 checkpoint. Stage 2 starts from one with "
      f"--agent.warm-start.path {path} (or its run directory); --agent.resume and "
      "play load stage-2 checkpoints only."
    )
  raise ValueError(f"{path} is not a stage-2 checkpoint. {WARM_START_HINT}")


def _library_dir(env: ManagerBasedRlEnv) -> str | None:
  commands = getattr(env.cfg, "commands", None) or {}
  motion = commands.get("motion") if isinstance(commands, Mapping) else None
  return getattr(motion, "library_dir", None) or None


def _check_same_observations(
  train: ManagerBasedRlEnv, other: ManagerBasedRlEnv
) -> None:
  """The validation env must feed the networks what the training env does."""
  ours = train.observation_manager.group_obs_dim
  theirs = other.observation_manager.group_obs_dim
  groups = ("actor", OSL_OBS_GROUP)
  differ = [g for g in groups if ours.get(g) != theirs.get(g)]
  if differ:
    detail = ", ".join(f"{g}: {ours.get(g)} vs {theirs.get(g)}" for g in differ)
    raise ValueError(
      "The validation env's observations differ from the training env's "
      f"({detail}). Every evaluation env copies the training env's observation "
      f"groups (validation with their corruption off and the {OSL_OBS_GROUP!r} "
      "group rebuilt clean), so the difference comes from the scene they read: "
      "the model, its sensors or the motion library."
    )
