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

Every checkpoint also exports the actor, as mjlab's stock runners do: the run
directory's ``<run>.onnx`` (overwritten at each save) with its deployment metadata
embedded and next to it as ``<run>.json`` (see :mod:`.export`). A deployment bundle
for a chosen checkpoint comes from ``mjlab.tasks.osl_gait_library.export``.
"""

from __future__ import annotations

import os
import time
from typing import cast

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan
from rsl_rl.utils.log_writer import LogWriter

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.osl_mimic.mdp import MimicMotionCommandCfg, mimic_reward

from .export import export_policy
from .models import MuscleMimicMLPModel
from .ppo import MuscleMimicPPO


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
    super().__init__(env, train_cfg, log_dir, device)
    self._validation_env: RslRlVecEnvWrapper | None = None

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

      # musclemimic validates after every interval-th update.
      if interval > 0 and (it + 1) % interval == 0 and self.logger.writer is not None:
        self.validate(it)

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")  # type: ignore[arg-type]
      )
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    """Checkpoint, then export the actor (ONNX + metadata) into the run directory.

    A failed export never ends the run: the checkpoint holds the actor."""
    super().save(path, infos)
    _, _, onnx_path = self._get_export_paths(path)
    try:
      export_policy(self.env.unwrapped, self.alg.get_policy(), onnx_path)
    except Exception as err:
      print(f"[WARN] ONNX export failed (training continues): {err}")
      return
    writer = self.logger.writer
    if self.cfg.get("upload_model", False) and isinstance(writer, LogWriter):
      writer.save_file(str(onnx_path))
      writer.save_file(str(onnx_path.with_suffix(".json")))

  ##
  # Validation.
  ##

  def _make_validation_env(self) -> RslRlVecEnvWrapper:
    from mjlab.tasks.osl_mimic.config.env_cfgs import osl_mimic_env_cfg

    v = self.validation_cfg
    cfg = osl_mimic_env_cfg(v["model"], v["socket"], v["reward"], mode="validation")
    cfg.scene.num_envs = int(v.get("num_envs", 100))
    train_motion = self.env.unwrapped.cfg.commands["motion"]
    motion = cfg.commands["motion"]
    assert isinstance(train_motion, MimicMotionCommandCfg)
    assert isinstance(motion, MimicMotionCommandCfg)
    motion.motion_file = train_motion.motion_file
    return RslRlVecEnvWrapper(
      ManagerBasedRlEnv(cfg=cfg, device=self.env.unwrapped.device)
    )

  def _default_validation_steps(self, env: ManagerBasedRlEnv) -> int:
    """One clip length."""
    return int(env.command_manager.get_term("motion").num_frames)  # type: ignore[attr-defined]

  def _validation_groups(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """(B,) boolean masks of validation envs reported separately. None here."""
    del env
    return {}

  def validate(self, it: int) -> dict[str, float]:
    if self._validation_env is None:
      self._validation_env = self._make_validation_env()
    vec = self._validation_env
    env = vec.unwrapped
    policy = cast(MuscleMimicMLPModel, self.alg.get_policy())
    normalizer = policy.obs_normalizer
    snapshot = {k: v.clone() for k, v in normalizer.state_dict().items()}

    num_steps = int(self.validation_cfg.get("num_steps", 0))
    if num_steps <= 0:
      num_steps = self._default_validation_steps(env)
    # Every class reward term that keeps per-step diagnostics (mimic_reward first).
    terms = [cast(mimic_reward, env.reward_manager.get_term_cfg("mimic").func)]
    for name in env.reward_manager.active_terms:
      func = env.reward_manager.get_term_cfg(name).func
      if name != "mimic" and isinstance(getattr(func, "components", None), dict):
        terms.append(func)

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
    # Reset inside inference mode too: stepping there turns the env's state tensors
    # into inference tensors, which a later reset outside it could not update.
    with torch.inference_mode():
      obs, _ = vec.reset()
      for _ in range(num_steps):
        actions = policy(obs.to(self.device))  # Deterministic: the distribution mean.
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
    writer = self.logger.writer
    if writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(f"Validation/{key}", value, it)
    summary = ("early_termination_rate", "mean_episode_length", "clf_V_norm")
    summary += ("err_site_abs", "twist_err_vx", "twist_reward")
    print(
      f"[Validation] it={it} "
      + " ".join(f"{k}={metrics[k]:.4g}" for k in summary if k in metrics),
      flush=True,
    )
    return metrics

  def close(self) -> None:
    if self._validation_env is not None:
      self._validation_env.close()
      self._validation_env = None
