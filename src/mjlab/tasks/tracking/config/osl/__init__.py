from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import osl_fullbody_tracking_env_cfg


class OslTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
  """Native mjlab checkpoints; mixed-actuator ONNX export is deferred."""

  def save(self, path: str, infos=None):
    MjlabOnPolicyRunner.save(self, path, infos)


# task id -> (model, socket, experiment name)
_VARIANTS: dict[str, tuple[Model, Socket, str]] = {
  "Mjlab-Tracking-Flat-OSL-Fullbody": ("robot", "rigid", "osl_fullbody_tracking"),
  "Mjlab-Tracking-Flat-OSL-Fullbody-Compliant": (
    "robot",
    "compliant",
    "osl_fullbody_compliant_tracking",
  ),
  "Mjlab-Tracking-Flat-OSL-Fullbody-MSK": ("msk", "rigid", "osl_fullbody_msk_tracking"),
  "Mjlab-Tracking-Flat-OSL-Fullbody-MSK-Compliant": (
    "msk",
    "compliant",
    "osl_fullbody_msk_compliant_tracking",
  ),
}

for _task_id, (_model, _socket, _experiment) in _VARIANTS.items():
  _rl_cfg = unitree_g1_tracking_ppo_runner_cfg()
  _rl_cfg.experiment_name = _experiment
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_fullbody_tracking_env_cfg(_model, _socket),
    play_env_cfg=osl_fullbody_tracking_env_cfg(_model, _socket, play=True),
    rl_cfg=_rl_cfg,
    runner_cls=OslTrackingOnPolicyRunner,
  )
