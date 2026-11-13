"""The stage-1.5 runner: stage 2's, with the human learning beside the OSL controller.

Stage 1.5 is stage 2 with the learned OSL actor replaced by the classical walking
controller, which the env's ``osl_motors`` action term runs from the prosthesis's
own sensors (:mod:`mjlab.tasks.osl_fsm.mdp.actions`). Only the human learns, through
:class:`~.ppo.HumanOnlyPPO`, stage 2's human path. So this runner is stage 2's
(:class:`~mjlab.tasks.osl_split.rl.runner.SplitOnPolicyRunner`): the warm start from
stage 1 with its checks, stage 1's iteration axis, the human's reward (the split
reward's human channel plus the twist term), the stage-local schedules, validation
and stress validation by clip family, the guards and the checkpoints with their
curriculum. It changes only what the missing OSL actor, or the controller in its
place, changes:

* **Settings** that need an OSL actor are refused before anything is built
  (:func:`~..config.rl_cfg.check_fsm_runner_cfg`): the ``osl_sampled`` family, the
  teacher's families, the blend and the OSL export. The evaluations play the
  human's raw mean (``Validation/``, ``Stress/``) or noise-averaged action beside
  the controller; at zero-shot that is the stage-1 human beside it, the floor stage
  1.5 starts from.
* **Evaluation envs run the training controller.** Validation and videos are
  rebuilt from the task, so each takes the training env's ``osl_fsm`` settings and
  its whole ``osl_motors`` term; stress validation copies the training config.
  The ``osl`` group is the clean one in every mode, and the OSL motors' delay zero.
  Validation and videos start every episode in early stance, as osl_deploy does:
  they start at frame 0, the OSL's touchdown, where the training default's pose
  guess reads an almost unloaded foot and guesses swing (676 of 779 clips), a start
  deployment never makes. Stress keeps training's initial state, since it plays the
  training distribution of start frames.
* **Diagnostics**: the controller's per-step diagnostics (the share of steps in
  each state, cycles, commanded torques, the load in body weights) are averaged
  with the reward terms' under ``Validation/`` and ``Stress/``.
* **Logging** is the human's: no OSL actor std, freeze, blend or teacher.
  ``Train/mean_reward`` (rsl_rl's) is the team reward, which includes the
  controller's motor-torque penalty (the split reward charges it to the OSL
  channel alone); the human trains on ``Train/human/mean_reward``.
* **Checkpoints** also record the controller settings the env ran
  (``infos["osl_fsm"]``, :func:`~..config.env_cfgs.fsm_settings`): a policy only
  meets the controller it learned beside. A resume under another controller (rules,
  swing exit, body mass, the term's ``controller``) is refused, naming the flags; the
  initial state is an episode-start convention and may differ. Play, which takes no
  ``--env`` overrides, builds the checkpoint's controller
  (:meth:`FsmSplitRunner.configure_env_for_checkpoint`) and, like validation, starts
  every episode in early stance. Nothing is exported next to them: the stage-1.5
  bundle exporter is ``python -m mjlab.tasks.osl_fsm.export``.
* **A reward switch** (another reward type than the warm start's) starts the critic
  fresh, as in stage 2, so the human may not learn before the fresh critic's
  warm-up ends (``freeze.human_actor_iterations`` at least
  ``freeze.reward_switch_critic_warmup_iterations``): it would otherwise follow
  advantages of a random value function. Without a switch the human learns from
  the first iteration.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from rsl_rl.env import VecEnv

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.osl_fsm.config.env_cfgs import (
  FSM_FLAGS,
  OslFsmEnvCfg,
  configure_fsm,
  fsm_settings,
  osl_fsm_env_cfg,
  sync_osl_fsm,
)
from mjlab.tasks.osl_fsm.config.rl_cfg import check_fsm_runner_cfg
from mjlab.tasks.osl_fsm.mdp import OslFsmActionCfg
from mjlab.tasks.osl_gait_library.config.dr import nominal_plant
from mjlab.tasks.osl_gait_library.mdp import GaitLibraryCommandCfg
from mjlab.tasks.osl_mimic.rl.runner import EvalPurpose, eval_mode
from mjlab.tasks.osl_mimic.touch_sites import load_checkpoint
from mjlab.tasks.osl_split.rl.layout import OSL_ACTION_TERM
from mjlab.tasks.osl_split.rl.runner import AgentRewards, SplitOnPolicyRunner
from mjlab.tasks.osl_split.rl.warmstart import WarmStart

from .ppo import ALGORITHM, HumanOnlyPPO, _hint

FSM_INFO_KEY = "osl_fsm"
"""The key of a checkpoint's ``infos`` that holds the controller settings."""
SETTING_FLAGS: dict[str, str] = {
  **FSM_FLAGS,
  "controller": "--env.actions.osl-motors.controller",
}
"""The controller settings a checkpoint records, each with the flag that sets it:
``--env.osl-fsm.*``, and the term's ``controller`` (``"fsm"``, or ``"policy"`` for
the policy's own OSL rows)."""
REQUIRED_SETTINGS: tuple[str, ...] = tuple(
  key for key in SETTING_FLAGS if key != "initial_state"
)
"""The settings a checkpoint and the env must agree on: everything that makes the
controller the human learned beside. The initial state only decides how an episode
starts, so a run may resume, and play may run, under another."""
EVAL_INITIAL_STATE = "e_stance"
"""How validation, video and play episodes start: osl_deploy's convention."""
EXPORT_HINT = (
  "Stage 1.5 has no OSL network to export: the human's partner is the walking "
  "controller. Its deployment bundle (the human actor with the controller's "
  "settings) is the stage-1.5 exporter's: `python -m mjlab.tasks.osl_fsm.export "
  "<run>/model_<N>.pt`, run with `python -m osl_deploy.osl_fsm BUNDLE --rules "
  "<rules>`."
)


class FsmSplitRunner(SplitOnPolicyRunner):
  """The stage-1.5 runner; see the module docstring.

  Construction refuses settings that need an OSL actor, then builds the algorithm
  and, given ``warm_start.path``, applies the warm start, as stage 2's does.
  """

  alg: HumanOnlyPPO  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
  ) -> None:
    check_fsm_runner_cfg(train_cfg)
    if "critic" in train_cfg:
      # Stage 2's warm start checks the OSL critic's block against stage 1's critic.
      # Stage 1.5 builds no OSL critic, so the block is the human critic's, which
      # the same check covers: it cannot refuse a warm start on its own.
      train_cfg["osl_critic"] = deepcopy(train_cfg["critic"])
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
    print(
      f"[INFO] Stage 1.5: the OSL runs the walking controller "
      f"({describe_fsm(fsm_settings(self.env))}); only the human learns.",
      flush=True,
    )

  def warm_start(self, ws: WarmStart) -> None:
    """Stage 2's warm start, then the reward-switch check (module docstring).

    Raises:
      ValueError: On stage 2's refusals, or under a reward switch when the human
        would learn before the fresh critic's warm-up ends.
    """
    super().warm_start(ws)
    schedule = self.schedule
    if self.reward_switch and schedule.human_freeze < schedule.critic_warmup:
      raise ValueError(
        f"The reward switches ({ws.lineage.get('reward')} -> "
        f"{self.validation_cfg.get('reward')}), so the critic starts fresh and warms "
        f"up for {schedule.critic_warmup} iterations, but the human would learn from "
        f"iteration {schedule.human_freeze}: it would follow the advantages of a "
        "random value function. Hold it for the warm-up with "
        f"--agent.freeze.human-actor-iterations {schedule.critic_warmup}, or train "
        "with the warm start's reward type."
      )

  ##
  # Validation.
  ##

  def _eval_env_cfg(self, purpose: EvalPurpose) -> ManagerBasedRlEnvCfg:
    """An evaluation env's config, running the training env's controller.

    Stress validation plays a copy of the training config without the run's seed,
    as stage 2's. Validation is the task's validation mode and a video its play
    mode on validation's nominal plant, each rebuilt from the task with the
    training env's controller: its ``osl_fsm`` settings and its whole
    ``osl_motors`` term (the controller choice and the clip too; nothing in it is
    mode-specific), except that their episodes start in early stance
    (:data:`EVAL_INITIAL_STATE`, see the module docstring). They also take the
    training env's motion library and its settings
    (:meth:`_inherit_training_settings`: the simulation, the observation groups, the
    clean ``osl`` group among them, every shared reward term's weight and params,
    the touch sites). The task's OSL motor delay is zero in every mode.

    Raises:
      TypeError: If the training env's config is not a stage-1.5 one.
    """
    train = self.env.unwrapped.cfg
    if purpose == "stress":
      cfg = deepcopy(train)
      cfg.seed = None
      return cfg
    if not isinstance(train, OslFsmEnvCfg):
      raise TypeError(f"Not a stage-1.5 training config: {type(train).__name__}.")
    v = self.validation_cfg
    fsm = replace(train.osl_fsm, initial_state=EVAL_INITIAL_STATE)
    cfg = osl_fsm_env_cfg(
      v["model"], v["socket"], v["reward"], eval_mode(purpose), fsm=fsm
    )
    cfg.actions = {
      **cfg.actions,
      OSL_ACTION_TERM: deepcopy(train.actions[OSL_ACTION_TERM]),
    }
    sync_osl_fsm(cfg)  # The copied term takes the evaluation's initial state.
    if purpose == "video":
      nominal_plant(cfg)
    train_motion = train.commands["motion"]
    motion = cfg.commands["motion"]
    assert isinstance(train_motion, GaitLibraryCommandCfg)
    assert isinstance(motion, GaitLibraryCommandCfg)
    motion.library_dir = train_motion.library_dir
    self._inherit_training_settings(cfg)
    return cfg

  def _diagnostic_reward_terms(
    self, env: ManagerBasedRlEnv, rewards: AgentRewards
  ) -> list[Any]:
    """Stage 2's (every class reward term that keeps per-step diagnostics, the
    split one first), then the controller's action term, whose diagnostics
    (:data:`~mjlab.tasks.osl_fsm.mdp.FSM_DIAGNOSTICS`) the evaluations average
    alike. The term keeps them through a reset, as the reward terms keep theirs, so
    each episode's last step counts too."""
    terms = super()._diagnostic_reward_terms(env, rewards)
    manager = env.action_manager
    if OSL_ACTION_TERM in manager.active_terms:
      term = manager.get_term(OSL_ACTION_TERM)
      if isinstance(getattr(term, "components", None), dict):
        terms.append(term)
    return terms

  ##
  # Logging.
  ##

  def _log_stage(self, it: int, k: int) -> None:
    """The human's scalars that rsl_rl's Logger does not write, under stage 2's
    names: its return, the two learning rates, its std, the stage's place, the
    skipped optimizer steps, the channels' weights and shares, the env steps since
    stage 1 began, the policy and critic statistics and the GPU memory. No OSL
    actor, freeze, blend or teacher scalars: there are none.

    rsl_rl's ``Train/mean_reward`` is the team reward, which includes the
    controller's motor-torque penalty; the human trains on
    ``Train/human/mean_reward`` (its channel plus the twist term)."""
    writer = self.logger.writer
    if writer is None:
      return
    alg = self.alg
    scalars: dict[str, float] = {}
    human = self._returns.means().get("human")
    if human is not None:
      scalars["Train/human/mean_reward"] = human
    for module, lr in alg.lrs.items():
      scalars[f"LR/{module}"] = lr
    scalars["Policy/human/mean_std"] = float(alg.human_std.mean())
    scalars["Stage2/iteration"] = k
    scalars["Stage2/human_frozen"] = float(alg.human_actor_frozen)
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
    for key, value in alg.policy_stats.items():
      scalars[f"Policy/{key}"] = value
    for key, value in alg.critic_stats.items():
      scalars[f"Critic/{key}"] = value
    scalars.update(self._gpu_memory())
    for tag, value in scalars.items():
      writer.add_scalar(tag, value, it)

  ##
  # Checkpoints and export.
  ##

  def save(self, path: str, infos: dict | None = None) -> None:
    """Stage 2's checkpoint (the human's networks, optimizers, return statistics,
    the stage's place and curriculum, the touch sites), plus the controller
    settings the env ran under ``infos["osl_fsm"]``."""
    infos = {**(infos or {}), FSM_INFO_KEY: fsm_settings(self.env)}
    super().save(path, infos)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Restore a stage-1.5 checkpoint as stage 2's load does (everything for a
    resume; the actor alone for ``play.py``'s ``load_cfg``), once it is known to
    have been trained beside the controller this env runs.

    Raises:
      ValueError: For a checkpoint that is not a stage-1.5 one (a stage-2 or
        stage-1 checkpoint, a warm-start file: the message says which), one that
        records no controller settings, or one trained under other controller
        settings than this env's (naming the flags that build its controller), or
        on stage 2's own refusals (the touch sites).
    """
    loaded = load_checkpoint(path)  # Memory-mapped: only the settings are read.
    kind = loaded.get("algorithm") if isinstance(loaded, Mapping) else None
    if kind != ALGORITHM:
      raise ValueError(f"{path} is not a stage-1.5 checkpoint.{_hint(loaded)}")
    stored = recorded_fsm_settings(loaded)
    del loaded
    if stored is None:
      raise ValueError(
        f"{path} records no OSL controller settings (infos[{FSM_INFO_KEY!r}]): not "
        "a checkpoint of a stage-1.5 run."
      )
    check_fsm_settings(stored, fsm_settings(self.env), path)
    return super().load(path, load_cfg, strict, map_location)

  @classmethod
  def configure_env_for_checkpoint(
    cls, env_cfg: ManagerBasedRlEnvCfg, checkpoint: str | Path
  ) -> None:
    """Make ``env_cfg`` build what ``checkpoint`` was trained with, before the env
    exists: its OSL touch sites (stage 2's), then its controller, the
    ``--env.osl-fsm.*`` settings and the term's ``controller`` recorded under
    ``infos["osl_fsm"]``, with episodes starting in early stance as in validation
    and deployment (:data:`EVAL_INITIAL_STATE`). ``play`` calls this; it takes no
    ``--env`` overrides.

    A checkpoint that records no controller settings leaves the config's as they
    are, with a warning: it is no stage-1.5 checkpoint, and loading it will say what
    it is instead.

    Raises:
      TypeError: If ``env_cfg`` is not a stage-1.5 env config.
      ValueError: For recorded settings the controller refuses, or stage 2's
        touch-site refusals.
    """
    super().configure_env_for_checkpoint(env_cfg, checkpoint)
    stored = recorded_fsm_settings(load_checkpoint(checkpoint))
    if stored is None:
      print(
        f"[WARN] {checkpoint} records no OSL controller settings (infos"
        f"[{FSM_INFO_KEY!r}]): not a stage-1.5 checkpoint. The env keeps the task's "
        "controller."
      )
      return
    configure_fsm(env_cfg, {**stored, "initial_state": EVAL_INITIAL_STATE})
    term = env_cfg.actions[OSL_ACTION_TERM]
    controller = stored.get("controller")
    if isinstance(term, OslFsmActionCfg) and controller not in (None, term.controller):
      replaced = replace(term, controller=controller)
      env_cfg.actions = {**env_cfg.actions, OSL_ACTION_TERM: replaced}
    print(
      f"[INFO] OSL controller: {describe_fsm(stored)} (from the checkpoint); "
      f"episodes start in {EVAL_INITIAL_STATE}, as in deployment."
    )

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    del path, filename, verbose
    raise NotImplementedError(EXPORT_HINT)

  def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
    del path, filename
    raise NotImplementedError(EXPORT_HINT)


##
# Helpers.
##


def recorded_fsm_settings(checkpoint: Any) -> dict[str, Any] | None:
  """The controller settings a loaded checkpoint records (``infos["osl_fsm"]``), or
  None when it records none."""
  infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
  settings = infos.get(FSM_INFO_KEY) if isinstance(infos, Mapping) else None
  return dict(settings) if isinstance(settings, Mapping) else None


def fsm_differences(
  stored: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
  """``{setting: (stored value, current value)}`` for each of
  :data:`SETTING_FLAGS` the two controller settings differ in. Both are
  :func:`~..config.env_cfgs.fsm_settings` dicts, whose ``swing_load_exit`` is
  resolved, so the sdk rules' default and an explicit True agree."""
  return {
    key: (stored.get(key), current.get(key))
    for key in SETTING_FLAGS
    if stored.get(key) != current.get(key)
  }


def check_fsm_settings(
  stored: Mapping[str, Any], current: Mapping[str, Any], path: str | Path
) -> None:
  """Refuse a checkpoint (``path``, whose recorded settings are ``stored``) trained
  beside another controller than the env runs (``current``): one that differs in
  any of :data:`REQUIRED_SETTINGS`. Another initial state alone is reported, not
  refused.

  Raises:
    ValueError: Listing each difference and the flags that build the checkpoint's
      controller: all of its ``--env.osl-fsm.*`` values, since what one of them
      resolves to depends on another (the swing exit's default on the rules), and
      the term's ``controller`` when that differs.
  """
  differences = fsm_differences(stored, current)
  if not any(key in REQUIRED_SETTINGS for key in differences):
    if differences:
      ours, theirs = differences["initial_state"]
      print(
        f"[INFO] {path} trained with the initial state {ours!r}; this env starts "
        f"episodes in {theirs!r}."
      )
    return
  lines = [
    f"  {key}: checkpoint {ours!r}, this env {theirs!r}"
    for key, (ours, theirs) in differences.items()
  ]
  keys = [*FSM_FLAGS, *(["controller"] if "controller" in differences else [])]
  flags = " ".join(f"{SETTING_FLAGS[key]} {stored.get(key)}" for key in keys)
  raise ValueError(
    f"{path} was trained beside another OSL controller than this env runs:\n"
    + "\n".join(lines)
    + "\nIts human learned to walk beside its own controller. Relaunch with "
    + f"{flags}."
  )


def describe_fsm(settings: Mapping[str, Any]) -> str:
  """A controller's settings in one line, for the logs."""
  mass = settings.get("body_mass")
  exit_ = "on" if settings.get("swing_load_exit") else "off"
  weight = settings.get("body_weight_n")
  weight_text = "?" if weight is None else f"{float(weight):.1f}"
  return (
    f"{settings.get('rules')} rules, swing load exit {exit_}, body weight "
    f"{weight_text} N ({'nominal' if mass is None else 'set'}), initial state "
    f"{settings.get('initial_state')}, controller {settings.get('controller')}"
  )
