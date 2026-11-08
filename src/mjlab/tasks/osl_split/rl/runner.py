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
  activation loads strictly and computes something else).
* **Iterations.** Stage 2 continues stage 1's numbering: its first iteration is
  stage 1's last plus one, so W&B and TensorBoard steps run on from stage 1's, and
  ``Train/mean_reward`` (the team reward, which the split reward term keeps equal
  to stage 1's) continues stage 1's curve. Schedules, saves and validations count
  *stage-local* iterations ``k`` from 0. A resumed stage-2 run takes its stage
  start, lineage and whole curriculum (schedule, channels, additive weights) from
  its checkpoint, and does not replay the checkpoint's own iteration.
* **Rewards.** After every step each agent's reward is composed from the reward
  manager (:class:`AgentRewards`): its imitation channel of the split ``mimic``
  term, plus every other (additive) term times its per-agent weight. The env's own
  reward stays the team reward, which the logger receives.
* **Schedules.** Before every rollout the algorithm takes ``k`` (freeze flags,
  per-module learning rates, the blend) and the reward term takes the channels'
  group weights for ``k``. The validation env's reward term takes the same
  weights before every validation.
* **Validation** is stage 1's protocol (one env per library clip, deterministic,
  ending early only on the 0.5 m site check) in two families: ``Validation/``
  with both actors' mean action, and ``ValidationExpected/`` with the human's
  noise-averaged action (its raw mean ends every MSK episode early). Both run once
  before the first update (zero-shot, logged at the first stage-2 iteration) and
  then every ``interval`` stage-local iterations. The validation env is the task's
  validation config with the training env's imitation-reward params and OSL
  observation (its corruption off), so it scores the reward being trained and
  feeds the OSL actor what it trains on.
* **Checkpoints** hold the four networks, their optimizers, both return
  normalizers and the split state (the stage's place and its curriculum). Each
  comes with the OSL actor alone, as ONNX plus its metadata (:mod:`.export`), in
  ``exported/`` next to it. ``play.py``'s load restores only the actors.
"""

from __future__ import annotations

import math
import os
import shlex
import statistics
import time
from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan
from rsl_rl.utils.log_writer import LogWriter

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_gait_library.rl.runner import GaitLibraryOnPolicyRunner
from mjlab.tasks.osl_split.config.env_cfgs import REWARD_NAME, osl_split_env_cfg
from mjlab.tasks.osl_split.mdp import osl_validation_group, split_mimic_reward

from .export import EXPORT_DIR, export_osl_policy
from .layout import AGENTS, ActionLayout
from .policy import HumanMode, SplitInferencePolicy
from .ppo import OSL_OBS_GROUP, TwoAgentPPO
from .schedules import ChannelSchedule, SplitSchedule
from .warmstart import (
  WarmStart,
  architecture_mismatches,
  check_compatible,
  load_warm_start,
)

OSL_JOINT_LABELS = ("knee", "ankle")
"""The OSL action rows in order (``OSL_TORQUE_JOINTS``, which the term preserves)."""
RETURN_WINDOW = 100
"""Completed episodes per agent behind ``Train/<agent>/mean_reward``: rsl_rl's."""
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

    Raises:
      ValueError: On a model or socket change, a warm start whose action layout
        or observation does not match this env, or a network that takes stage-1
        weights but is configured otherwise than stage 1's (the message names
        the overrides that build stage 1's).
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
    stage1_library = ws.lineage.get("library_dir")
    library = _library_dir(env)
    if stage1_library and library and Path(stage1_library) != Path(library):
      print(
        f"[WARN] Stage 1 trained on the library {stage1_library}; stage 2 uses "
        f"{library}."
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
  ) -> tuple[dict[str, Any], str | None]:
    """The curriculum the current config gives under ``reward_switch``, to compare
    with a checkpoint's, and why it has no schedule when the config cannot build
    one there (the checkpoint's replaces it anyway, so that is reported, not
    raised).

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
    additive = self.cfg["channels"].get("additive_term_weights") or {}
    channels = ChannelSchedule.from_train_cfg(self.cfg)
    return _curriculum(schedule, channels, additive), error

  def _restore_curriculum(self, split: Mapping[str, Any]) -> None:
    """Resume on the curriculum the checkpoint was trained with: its schedule
    (stage length, freeze, warm-ups, learning rates, blend), reward switch, channel
    schedule and additive term weights.

    A resumed run continues the run it resumes (rsl_rl's resume). Its command line
    builds the config, and one that does not repeat the run's stage-2 flags would
    otherwise freeze an unfrozen human again or snap the OSL channel back to its
    start weights mid-stage. Every value the current config sets differently is
    listed in a warning; the checkpoint's wins. A checkpoint written before the
    curriculum was kept restores its stage length and reward switch only.
    """
    reward_switch = bool(split.get("reward_switch", False))
    if "schedule" not in split:
      iterations = int(split.get("schedule_iterations") or 0)
      iterations = iterations or self._configured_schedule_iterations()
      self._install_schedule(iterations, reward_switch)
      print(
        "[WARN] The checkpoint predates saved curricula: the current config's "
        "freeze, learning rates, blend and reward channels apply, over its "
        f"schedule of {iterations} iterations."
      )
      return
    schedule = SplitSchedule(**split["schedule"])
    channels = ChannelSchedule(**split["channel_schedule"])
    additive: dict[str, Sequence[float]] = {
      name: tuple(float(w) for w in pair)
      for name, pair in split["additive_term_weights"].items()
    }
    configured, error = self._configured_curriculum(
      schedule.total_iterations, reward_switch
    )
    checkpoint = _curriculum(schedule, channels, additive)
    differences = _curriculum_differences(checkpoint, configured)
    if error is not None:
      differences.insert(0, f"schedule: the config cannot build one ({error})")

    for weights in (channels.start, channels.end):
      self._rewards.term.check_group_weights(weights)
    self._rewards = AgentRewards(self.env.unwrapped, additive)
    self._validation_rewards = None  # Rebuilt with these weights when next needed.
    self._additive_weights = additive
    self.channel_schedule = channels
    self._set_schedule(schedule, reward_switch)
    if differences:
      print(
        "[WARN] The resumed run keeps its checkpoint's curriculum, which the "
        "current config does not repeat; the checkpoint's values win:\n"
        + "\n".join(f"  {line}" for line in differences),
        flush=True,
      )

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
        f"[WARN] Training to stage iteration {total_it - self.stage_start}, past "
        f"the schedule's {self.schedule_iterations}: every learning rate is 0 "
        "from there."
      )
    interval = int(self.validation_cfg.get("interval", 0))
    save_interval = int(self.cfg["save_interval"])
    num_steps = int(self.cfg["num_steps_per_env"])
    check = self.cfg.get("check_for_nan", True)
    writer = self.logger.writer
    log_dir = str(self.logger.log_dir)
    # Zero-shot: the split policies before any stage-2 update (interval 0 turns
    # every validation off, this one included).
    zero_shot = bool(self.validation_cfg.get("zero_shot", True)) and interval > 0
    if zero_shot and start_it == self.stage_start and writer is not None:
      self.validate(start_it)

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
        self.validate(it)
      if writer is not None and save_interval > 0 and k % save_interval == 0:
        self.save(os.path.join(log_dir, f"model_{it}.pt"))

    if writer is not None:
      if total_it > start_it:  # Never a checkpoint of an iteration that did not run.
        path = f"model_{self.current_learning_iteration}.pt"
        self.save(os.path.join(log_dir, path))
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
    for tag, value in scalars.items():
      writer.add_scalar(tag, value, it)

  def _stage1_env_steps(self) -> int:
    return int((self.lineage or {}).get("env_steps") or 0)

  ##
  # Validation.
  ##

  def _make_validation_env(self) -> RslRlVecEnvWrapper:
    """The task's validation env, made to score and observe what training does.

    From the training env's config it takes the motion library, the imitation
    reward's params (the channels' settings, e.g. ``owner_penalties``, decide the
    per-agent rewards and diagnostics validation reports) and the OSL observation
    group with its corruption off (:func:`~..mdp.osl_validation_group`), so a
    changed history or sensor switch reaches the OSL actor in validation too. The
    params depend on the reward type only, never on the mode.
    """
    v = self.validation_cfg
    train = self.env.unwrapped.cfg
    cfg = osl_split_env_cfg(v["model"], v["socket"], v["reward"], mode="validation")
    train_motion = train.commands["motion"]
    motion = cfg.commands["motion"]
    assert isinstance(train_motion, GaitLibraryCommandCfg)
    assert isinstance(motion, GaitLibraryCommandCfg)
    motion.library_dir = train_motion.library_dir
    clips = len(list(Path(motion.library_dir).glob("*.npz")))
    cfg.scene.num_envs = int(v.get("num_envs", 0)) or clips
    cfg.rewards[REWARD_NAME].params = deepcopy(train.rewards[REWARD_NAME].params)
    cfg.observations[OSL_OBS_GROUP] = osl_validation_group(
      train.observations[OSL_OBS_GROUP]
    )
    vec = RslRlVecEnvWrapper(
      ManagerBasedRlEnv(cfg=cfg, device=self.env.unwrapped.device)
    )
    _check_same_observations(self.env.unwrapped, vec.unwrapped)
    return vec

  def validate(self, it: int) -> dict[str, float]:
    """Both validation families at global iteration ``it``; returns their metrics
    under their prefixes (``Validation/...``, ``ValidationExpected/...``)."""
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
    families: list[tuple[str, HumanMode]] = [("Validation", "mean")]
    if self.validation_cfg.get("expected", True):
      families.append(("ValidationExpected", "expected"))
    writer = self.logger.writer
    results: dict[str, float] = {}
    for prefix, mode in families:
      policy = self.alg.inference_policy(mode, env.action_manager)
      metrics = self._validation_pass(vec, policy, rewards)
      for key, value in metrics.items():
        results[f"{prefix}/{key}"] = value
        if writer is not None:
          writer.add_scalar(f"{prefix}/{key}", value, it)
      summary = (
        "early_termination_rate",
        "mean_episode_length",
        "mean_episode_return_human",
        "mean_episode_return_osl",
        "clf_V_norm",
        "err_site_abs",
        "split_err_osl_knee",
        "twist_err_vx",
      )
      print(
        f"[{prefix}] it={it} k={k} "
        + " ".join(f"{key}={metrics[key]:.4g}" for key in summary if key in metrics),
        flush=True,
      )
    return results

  def _validation_pass(
    self,
    vec: RslRlVecEnvWrapper,
    policy: SplitInferencePolicy,
    rewards: AgentRewards,
  ) -> dict[str, float]:
    """One validation run of ``policy``: stage 1's loop and metrics, plus each
    agent's completed-episode return (``mean_episode_return_<agent>``).

    The policy keeps both observation normalizers frozen, so no snapshot is taken.
    """
    env = vec.unwrapped
    num_steps = int(self.validation_cfg.get("num_steps", 0))
    if num_steps <= 0:
      num_steps = self._default_validation_steps(env)
    # Every class reward term that keeps per-step diagnostics, the split one first.
    terms: list[Any] = [rewards.term]
    for name in env.reward_manager.active_terms:
      func = env.reward_manager.get_term_cfg(name).func
      if name != REWARD_NAME and isinstance(getattr(func, "components", None), dict):
        terms.append(func)

    n = env.num_envs
    sums: dict[str, torch.Tensor] = {}
    episode_return = torch.zeros(n, device=env.device)
    agent_return = torch.zeros(n, len(AGENTS), device=env.device)
    episode_length = torch.zeros(n, device=env.device)
    finished_returns: list[torch.Tensor] = []
    finished_agent_returns: list[torch.Tensor] = []
    finished_lengths: list[torch.Tensor] = []
    num_done = 0
    num_early = 0
    groups = {g: m for g, m in self._validation_groups(env).items() if bool(m.any())}
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
            if key not in self.validation_group_keys:
              continue
            for group, mask in groups.items():
              name = f"{key}/{group}"
              mean = value[mask].float().mean()
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
  stage-1 observation statistics). A strict load catches a changed width, not a
  changed activation, which would load and compute something else.

  Raises:
    ValueError: Naming each difference and the ``--agent`` overrides that build
      stage 1's architecture.
  """
  networks = [("actor", "actor_cfg")]
  if not reward_switch:
    networks += [("critic", "critic_cfg"), ("osl_critic", "critic_cfg")]
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
      f"({detail}). Validation takes the {OSL_OBS_GROUP!r} group from the "
      "training env but builds the human's 'actor' group from the task's "
      "defaults, so a changed 'actor' observation must change the task's config."
    )
