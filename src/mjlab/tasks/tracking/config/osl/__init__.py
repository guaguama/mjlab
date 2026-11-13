from pathlib import Path

import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Socket
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.osl_mimic.touch_sites import (
  CHECKPOINT_KEY,
  check_checkpoint_file,
  configure_for_checkpoint,
  env_touch_sites,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import osl_fullbody_tracking_env_cfg


class OslTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
  """Native mjlab checkpoints, which also record the OSL touch sites the env
  simulates (and refuse to load into one with others); mixed-actuator ONNX export
  is deferred."""

  def save(self, path: str, infos=None):
    """``MjlabOnPolicyRunner.save`` plus the OSL touch-site version."""
    env = self.env.unwrapped
    infos = {
      **(infos or {}),
      "env_state": {"common_step_counter": env.common_step_counter},
    }
    saved = self.alg.save()
    saved["iter"] = self.current_learning_iteration
    saved["infos"] = infos
    saved[CHECKPOINT_KEY] = env_touch_sites(env)
    torch.save(saved, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Refuses a checkpoint trained with other OSL touch sites (one that records
    none trained with ``original``), then loads it."""
    check_checkpoint_file(path, self.env.unwrapped)
    return super().load(path, load_cfg, strict, map_location)

  @classmethod
  def configure_env_for_checkpoint(
    cls, env_cfg: ManagerBasedRlEnvCfg, checkpoint: str | Path
  ) -> None:
    """Build the OSL touch sites ``checkpoint`` was trained with (for play)."""
    configure_for_checkpoint(env_cfg, checkpoint)


# task id -> (socket, experiment name). The musculoskeletal model trains with
# musclemimic's recipe instead (tasks/osl_mimic): G1's rewards give it a negative
# per-step reward, under which ending the episode early is optimal.
_VARIANTS: dict[str, tuple[Socket, str]] = {
  "Mjlab-Tracking-Flat-OSL-Fullbody-ROBOT": ("rigid", "osl_fullbody_tracking"),
  "Mjlab-Tracking-Flat-OSL-Fullbody-ROBOT-Compliant": (
    "compliant",
    "osl_fullbody_compliant_tracking",
  ),
}

for _task_id, (_socket, _experiment) in _VARIANTS.items():
  _rl_cfg = unitree_g1_tracking_ppo_runner_cfg()
  _rl_cfg.experiment_name = _experiment
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_fullbody_tracking_env_cfg(_socket),
    play_env_cfg=osl_fullbody_tracking_env_cfg(_socket, play=True),
    rl_cfg=_rl_cfg,
    runner_cls=OslTrackingOnPolicyRunner,
  )
