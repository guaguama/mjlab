"""The on-policy runner for musclemimic-style imitation, with in-training validation.

musclemimic validates every ``num_updates // validation.num`` updates (61 for 1B
steps and 100 validations): 100 envs run the deterministic policy from frame 0 for
one clip length, ending episodes only on a 0.5 m mean site deviation or the clip's
end, with auto-reset. The policy's observation statistics keep updating during the
rollout and are restored afterwards. The metrics are the training diagnostics,
averaged over every validation step, plus the early-termination rate and the
completed episodes' length and return. They are logged under ``Validation/``.

Two other departures from rsl_rl's loop, both musclemimic's: initial episode lengths
are never randomized, and the rollout uses the algorithm's own normalization modes.

Every checkpoint also gets a video, as musclemimic records at its validations: one
env of the task's play mode on the nominal plant (the reference ghost on, where the
task draws one) runs the deterministic policy for ``video.num_steps`` control steps
at the control rate. It is written next to the checkpoint as
``videos/model_<it>.mp4`` and uploaded through the run's logger (W&B: ``video`` at
the checkpoint's iteration). A failed recording never ends the run.

Beyond musclemimic's:

* validation can play several deterministic actions, each under its own prefix
  (``validation.inference``): the raw mean (``Validation/``), the noise-averaged
  action ``E[clip(mu + sigma eps)]`` (``ValidationExpected/``, see
  :mod:`.expected`) or the stochastic policy (``ValidationSampled/``);
* a stress validation (``validation.stress_interval``) plays the training
  distribution itself with deterministic actions, ending episodes only on
  validation's site check, and reports each env's first episode (``Stress/``,
  ``StressExpected/``): the survival rate is the share of envs that last the whole
  episode;
* the algorithm's ``policy_stats`` (std per action term, the std cap, the noise gap)
  are logged under ``Policy/`` every iteration;
* videos play ``video.inference`` (the noise-averaged action for muscle policies by
  default).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

import mediapy as media
import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan
from rsl_rl.utils.log_writer import LogWriter
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import TendonEffortAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.osl_mimic.mdp import MimicMotionCommandCfg, mimic_reward

from .expected import action_bounds, expected_clipped_gaussian
from .models import MuscleMimicMLPModel, gaussian_std
from .ppo import MuscleMimicPPO

EvalPurpose = Literal["validation", "video", "stress"]
"""What an evaluation env is built for: validation runs the task's validation mode,
videos its play mode, stress validation its training mode (with validation's site
check as the only failure)."""
Inference = Literal["mean", "expected", "sampled"]
VALIDATION_PREFIXES: dict[str, str] = {
  "mean": "Validation",
  "expected": "ValidationExpected",
  "sampled": "ValidationSampled",
}
STRESS_PREFIXES: dict[str, str] = {
  "mean": "Stress",
  "expected": "StressExpected",
  "sampled": "StressSampled",
}


EvalMode = Literal["validation", "play", "train"]
_EVAL_MODES: dict[str, EvalMode] = {
  "validation": "validation",
  "video": "play",
  "stress": "train",
}


def eval_mode(purpose: EvalPurpose) -> EvalMode:
  """The task mode an evaluation env of ``purpose`` is built in."""
  return _EVAL_MODES[purpose]


def inference_modes(modes: Any, default: tuple[str, ...]) -> tuple[str, ...]:
  """Validated inference modes (a string, or a sequence of strings)."""
  if not modes:
    return default
  modes = (modes,) if isinstance(modes, str) else tuple(modes)
  unknown = sorted(set(modes) - set(VALIDATION_PREFIXES))
  if unknown:
    raise ValueError(
      f"Unknown inference modes {unknown}; valid: {list(VALIDATION_PREFIXES)}"
    )
  return modes


def drives_muscles(env: ManagerBasedRlEnv) -> bool:
  manager = env.action_manager
  return any(
    isinstance(manager.get_term(name), TendonEffortAction)
    for name in manager.active_terms
  )


class OslMimicOnPolicyRunner(MjlabOnPolicyRunner):
  alg: MuscleMimicPPO

  validation_group_keys: tuple[str, ...] = ()
  """Diagnostics also reported per validation group (``<key>/<group>``), next to
  each group's early-termination rate and episode length."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs,
  ) -> None:
    kwargs.pop("registry_name", None)  # Passed to tracking runners; unused here.
    self.validation_cfg: dict = dict(train_cfg.get("validation") or {})
    self.video_cfg: dict = dict(train_cfg.get("video") or {})
    super().__init__(env, train_cfg, log_dir, device)
    self._validation_env: RslRlVecEnvWrapper | None = None
    self._stress_env: RslRlVecEnvWrapper | None = None
    self._video_env: RslRlVecEnvWrapper | None = None

  ##
  # Training loop: rsl_rl's, plus validation.
  ##

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    del init_at_random_ep_len  # musclemimic never randomizes episode lengths.
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    interval = int(self.validation_cfg.get("interval", 0))
    stress_interval = int(self.validation_cfg.get("stress_interval", 0))
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)
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
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )
      self._log_policy_stats(it)

      # musclemimic validates after every interval-th update.
      if interval > 0 and (it + 1) % interval == 0 and self.logger.writer is not None:
        self._guarded(self.validate, it)
      if (
        stress_interval > 0
        and (it + 1) % stress_interval == 0
        and self.logger.writer is not None
      ):
        self._guarded(self.stress_validate, it)

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]
        self.record_video(it)

    if self.logger.writer is not None:
      it = self.current_learning_iteration
      self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]
      self.record_video(it)
      self.close_video_env()
      self.logger.stop_logging_writer()

  ##
  # Validation.
  ##

  def _eval_env_cfg(self, purpose: EvalPurpose) -> ManagerBasedRlEnvCfg:
    """The task's config for an evaluation env, with the training env's motion and
    settings (see :meth:`_inherit_training_settings`). Stress validation plays the
    training config itself."""
    from mjlab.tasks.osl_mimic.config.env_cfgs import osl_mimic_env_cfg

    if purpose == "stress":
      return deepcopy(self.env.unwrapped.cfg)
    v = self.validation_cfg
    cfg = osl_mimic_env_cfg(v["model"], v["socket"], v["reward"], eval_mode(purpose))
    train_motion = self.env.unwrapped.cfg.commands["motion"]
    motion = cfg.commands["motion"]
    assert isinstance(train_motion, MimicMotionCommandCfg)
    assert isinstance(motion, MimicMotionCommandCfg)
    motion.motion_file = train_motion.motion_file
    self._inherit_training_settings(cfg)
    return cfg

  def _inherit_training_settings(self, cfg: ManagerBasedRlEnvCfg) -> None:
    """Give an evaluation config, rebuilt from the task, the settings a run may have
    overridden on the command line: the simulation (solver, contact model), the
    observation groups (the policy's input; their noise off) and every shared reward
    term's weight and parameters. Everything mode-specific (commands, terminations,
    randomization) stays the evaluation mode's own."""
    train = self.env.unwrapped.cfg
    cfg.sim = deepcopy(train.sim)
    cfg.observations = deepcopy(train.observations)
    for group in cfg.observations.values():
      group.enable_corruption = False
    for name, term in cfg.rewards.items():
      if name in train.rewards:
        term.weight = train.rewards[name].weight
        term.params = deepcopy(train.rewards[name].params)

  def _guarded(self, validation: Callable[[int], Any], it: int) -> None:
    """Run an in-training validation; a failure is reported, never fatal."""
    try:
      validation(it)
    except Exception as err:  # noqa: BLE001
      name = getattr(validation, "__name__", "validation")
      print(f"[WARN] {name} at it={it} failed (training continues): {err!r}")

  def _validation_num_envs(self) -> int:
    return int(self.validation_cfg.get("num_envs", 100))

  def _build_eval_env(
    self, cfg: ManagerBasedRlEnvCfg, render: bool = False
  ) -> RslRlVecEnvWrapper:
    env = ManagerBasedRlEnv(
      cfg=cfg,
      device=self.env.unwrapped.device,
      render_mode="rgb_array" if render else None,
    )
    return RslRlVecEnvWrapper(env)

  def _make_validation_env(self) -> RslRlVecEnvWrapper:
    cfg = self._eval_env_cfg("validation")
    cfg.scene.num_envs = self._validation_num_envs()
    return self._build_eval_env(cfg)

  def _make_stress_env(self) -> RslRlVecEnvWrapper:
    from mjlab.tasks.osl_mimic.config.env_cfgs import stress_env_cfg

    v = self.validation_cfg
    cfg = stress_env_cfg(
      self._eval_env_cfg("stress"),
      num_steps=int(v.get("stress_num_steps", 1000)),
      num_envs=int(v.get("stress_num_envs", 512)),
    )
    return self._build_eval_env(cfg)

  def _default_validation_steps(self, env: ManagerBasedRlEnv) -> int:
    """One clip length."""
    return int(env.command_manager.get_term("motion").num_frames)  # type: ignore[attr-defined]

  def _validation_groups(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """(B,) boolean masks of validation envs reported separately. None here."""
    del env
    return {}

  def _inference_policy(
    self, mode: str, env: ManagerBasedRlEnv
  ) -> Callable[[TensorDict], torch.Tensor]:
    """``obs -> action`` for an inference mode: the raw mean, the noise-averaged
    action (see :mod:`.expected`), or a sample of the stochastic policy. The actor's
    normalization mode is left as the caller set it."""
    policy = cast(MuscleMimicMLPModel, self.alg.get_policy())
    if mode == "mean":
      return policy
    if mode == "sampled":
      return lambda obs: policy(obs, stochastic_output=True)
    if mode != "expected":
      raise ValueError(f"Unknown inference mode {mode!r}")
    lo, hi = action_bounds(env.action_manager, device=self.device)

    def expected(obs: TensorDict) -> torch.Tensor:
      return expected_clipped_gaussian(policy(obs), gaussian_std(policy), lo, hi)

    return expected

  def _write_metrics(
    self, prefix: str, metrics: dict[str, float], it: int, summary: tuple[str, ...]
  ) -> dict[str, float]:
    writer = self.logger.writer
    if writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, it)
    print(
      f"[{prefix}] it={it} "
      + " ".join(f"{k}={metrics[k]:.4g}" for k in summary if k in metrics),
      flush=True,
    )
    return {f"{prefix}/{key}": value for key, value in metrics.items()}

  def _log_policy_stats(self, it: int) -> None:
    writer = self.logger.writer
    if writer is None:
      return
    for key, value in getattr(self.alg, "policy_stats", {}).items():
      writer.add_scalar(f"Policy/{key}", value, it)

  def _diagnostic_terms(self, env: ManagerBasedRlEnv) -> list[Any]:
    """Every class reward term that keeps per-step diagnostics, mimic_reward first."""
    terms: list[Any] = [
      cast(mimic_reward, env.reward_manager.get_term_cfg("mimic").func)
    ]
    for name in env.reward_manager.active_terms:
      func = env.reward_manager.get_term_cfg(name).func
      if name != "mimic" and isinstance(getattr(func, "components", None), dict):
        terms.append(func)
    return terms

  def validate(self, it: int) -> dict[str, float]:
    """One validation per ``validation.inference`` mode, each logged under its
    prefix; returns every metric, prefixed."""
    if self._validation_env is None:
      self._validation_env = self._make_validation_env()
    vec = self._validation_env
    summary = ("early_termination_rate", "mean_episode_length", "clf_V_norm")
    summary += ("err_site_abs", "twist_err_vx", "twist_reward")
    results: dict[str, float] = {}
    modes = inference_modes(self.validation_cfg.get("inference"), ("mean",))
    for mode in modes:
      metrics = self._validation_run(vec, mode)
      results.update(
        self._write_metrics(VALIDATION_PREFIXES[mode], metrics, it, summary)
      )
    return results

  def _validation_run(self, vec: RslRlVecEnvWrapper, mode: str) -> dict[str, float]:
    """musclemimic's validation run of one inference mode: every env from its
    start, auto-reset, the observation statistics updating during the run and
    restored after it."""
    env = vec.unwrapped
    policy = cast(MuscleMimicMLPModel, self.alg.get_policy())
    act = self._inference_policy(mode, env)
    normalizer = policy.obs_normalizer
    snapshot = {k: v.clone() for k, v in normalizer.state_dict().items()}

    num_steps = int(self.validation_cfg.get("num_steps", 0))
    if num_steps <= 0:
      num_steps = self._default_validation_steps(env)
    terms = self._diagnostic_terms(env)

    n = env.num_envs
    sums: dict[str, torch.Tensor] = {}
    episode_return = torch.zeros(n, device=env.device)
    episode_length = torch.zeros(n, device=env.device)
    finished_returns: list[torch.Tensor] = []
    finished_lengths: list[torch.Tensor] = []
    num_done = 0
    num_early = 0
    groups = {g: m for g, m in self._validation_groups(env).items() if bool(m.any())}
    group_done = dict.fromkeys(groups, 0)
    group_early = dict.fromkeys(groups, 0)
    group_lengths: dict[str, list[torch.Tensor]] = {g: [] for g in groups}

    policy.set_norm_mode("update")
    try:
      # Reset inside inference mode too: stepping there turns the env's state
      # tensors into inference tensors, which a later reset outside it could not
      # update.
      with torch.inference_mode():
        obs, _ = vec.reset()
        for _ in range(num_steps):
          actions = act(obs.to(self.device))
          obs, reward, dones, extras = vec.step(actions.to(vec.device))
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
          episode_length += 1
          done = terminated | truncated
          if done.any():
            num_done += int(done.sum())
            num_early += int((terminated & ~truncated).sum())
            finished_returns.append(episode_return[done].clone())
            finished_lengths.append(episode_length[done].clone())
            for group, mask in groups.items():
              group_done[group] += int((done & mask).sum())
              group_early[group] += int((terminated & ~truncated & mask).sum())
              group_lengths[group].append(episode_length[done & mask].clone())
            episode_return[done] = 0.0
            episode_length[done] = 0.0
    finally:
      normalizer.load_state_dict(snapshot)
      policy.set_norm_mode("update")

    metrics = {key: float(value) / num_steps for key, value in sums.items()}
    metrics["early_termination_rate"] = num_early / num_done if num_done else 0.0
    if finished_lengths:
      metrics["mean_episode_length"] = float(torch.cat(finished_lengths).mean())
      metrics["mean_episode_return"] = float(torch.cat(finished_returns).mean())
    for group in groups:
      done_g = group_done[group]
      metrics[f"early_termination_rate/{group}"] = (
        group_early[group] / done_g if done_g else 0.0
      )
      lengths = torch.cat(group_lengths[group]) if group_lengths[group] else None
      if lengths is not None and lengths.numel():
        metrics[f"mean_episode_length/{group}"] = float(lengths.mean())
    return metrics

  def stress_validate(self, it: int) -> dict[str, float]:
    """One stress validation per ``validation.stress_inference`` mode (see the
    module docstring), each seeded alike; returns every metric, prefixed."""
    seed = int(self.validation_cfg.get("stress_seed", 0))
    device = torch.device(self.env.unwrapped.device)
    cuda = [device.index or 0] if device.type == "cuda" else []
    if self._stress_env is None:
      # Its physical draws come from the stress seed, not training's random state.
      with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(seed)
        self._stress_env = self._make_stress_env()
    vec = self._stress_env
    summary = ("survival_rate", "mean_first_episode_s", "err_site_abs")
    summary += ("twist_err_vx", "twist_err_wz", "clf_V_norm")
    results: dict[str, float] = {}
    modes = inference_modes(self.validation_cfg.get("stress_inference"), ("expected",))
    for mode in modes:
      with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(seed)
        metrics = self._stress_pass(vec, mode)
      results.update(self._write_metrics(STRESS_PREFIXES[mode], metrics, it, summary))
    return results

  def _stress_pass(self, vec: RslRlVecEnvWrapper, mode: str) -> dict[str, float]:
    """Every env's first episode under one inference mode. ``survival_rate`` is the
    share of envs whose first episode was not terminated early;
    ``mean_first_episode_s`` its mean length (s); the diagnostics are averaged over
    the envs still in their first episode at each step."""
    env = vec.unwrapped
    policy = cast(MuscleMimicMLPModel, self.alg.get_policy())
    act = self._inference_policy(mode, env)
    normalizer = policy.obs_normalizer
    snapshot = {k: v.clone() for k, v in normalizer.state_dict().items()}
    num_steps = int(self.validation_cfg.get("stress_num_steps", 1000))
    terms = self._diagnostic_terms(env)

    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    failed = torch.zeros(n, dtype=torch.bool, device=env.device)
    length = torch.zeros(n, device=env.device)
    sums: dict[str, torch.Tensor] = {}
    weighted_steps = 0
    policy.set_norm_mode("update")
    try:
      with torch.inference_mode():
        obs, _ = vec.reset()
        for _ in range(num_steps):
          actions = act(obs.to(self.device))
          obs, _, dones, extras = vec.step(actions.to(vec.device))
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
          length += weight
          failed |= alive & terminated
          alive &= ~(terminated | truncated)
    finally:
      normalizer.load_state_dict(snapshot)
      policy.set_norm_mode("update")

    metrics = {
      key: float(value) / max(weighted_steps, 1) for key, value in sums.items()
    }
    metrics["survival_rate"] = 1.0 - float(failed.float().mean())
    metrics["mean_first_episode_s"] = float(length.mean()) * float(env.step_dt)
    return metrics

  ##
  # Checkpoint videos.
  ##

  def _make_video_env(self) -> RslRlVecEnvWrapper:
    cfg = self._eval_env_cfg("video")
    cfg.scene.num_envs = 1
    cfg.viewer.width = int(self.video_cfg.get("width", 640))
    cfg.viewer.height = int(self.video_cfg.get("height", 480))
    return self._build_eval_env(cfg, render=True)

  def _video_inference(self, env: ManagerBasedRlEnv) -> str:
    mode = str(self.video_cfg.get("inference", "auto"))
    if mode == "auto":
      return "expected" if drives_muscles(env) else "mean"
    return mode

  @contextmanager
  def _video_policy(
    self, env: ManagerBasedRlEnv
  ) -> Iterator[Callable[[TensorDict], torch.Tensor]]:
    """The deterministic policy of ``video.inference`` (the raw mean, or the
    noise-averaged action), with its observation statistics frozen."""
    policy = cast(MuscleMimicMLPModel, self.alg.get_policy())
    policy.set_norm_mode("frozen")
    try:
      yield self._inference_policy(self._video_inference(env), env)
    finally:
      policy.set_norm_mode("update")

  def record_video(self, it: int) -> Path | None:
    """Record the checkpoint of iteration ``it`` (see the module docstring) and
    upload it. Returns the video's path, or None when disabled or failed."""
    if not self.video_cfg.get("enabled", True) or self.logger.log_dir is None:
      return None
    path = Path(self.logger.log_dir) / "videos" / f"model_{it}.mp4"
    if path.exists():  # This checkpoint already has its video (a final re-save).
      return path
    num_steps = int(self.video_cfg.get("num_steps", 3000))
    start = time.time()
    try:
      if self._video_env is None:
        self._video_env = self._make_video_env()
      vec = self._video_env
      env = vec.unwrapped
      path.parent.mkdir(parents=True, exist_ok=True)
      device = torch.device(env.device)
      cuda = [device.index or 0] if device.type == "cuda" else []
      # The video env's draws are seeded, and training's random state is restored.
      with (
        torch.random.fork_rng(devices=cuda),
        torch.inference_mode(),
        self._video_policy(env) as policy,
      ):
        torch.manual_seed(int(self.video_cfg.get("seed", 0)))
        obs, _ = vec.reset()
        frame = env.render()
        assert frame is not None
        fps = round(1.0 / env.step_dt)
        with media.VideoWriter(path, shape=frame.shape[:2], fps=fps) as video:
          video.add_image(frame)
          for _ in range(num_steps - 1):
            actions = policy(obs.to(self.device))
            obs, _, _, _ = vec.step(actions.to(vec.device))
            frame = env.render()
            assert frame is not None
            video.add_image(frame)
    except Exception as err:
      path.unlink(missing_ok=True)
      print(f"[WARN] Video of model_{it} failed (training continues): {err}")
      return None
    writer = self.logger.writer
    if isinstance(writer, LogWriter):
      writer.save_video(path, it)
    print(f"[Video] it={it} {path} ({time.time() - start:.0f} s)", flush=True)
    return path

  def close_video_env(self) -> None:
    """Release the video env and its renderer. Training calls it after the last
    video: an EGL renderer left to the garbage collector at exit raises there."""
    if self._video_env is not None:
      self._video_env.close()
      self._video_env = None

  def close(self) -> None:
    if self._validation_env is not None:
      self._validation_env.close()
      self._validation_env = None
    if self._stress_env is not None:
      self._stress_env.close()
      self._stress_env = None
    self.close_video_env()
