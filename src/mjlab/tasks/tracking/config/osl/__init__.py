from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Socket
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import osl_fullbody_tracking_env_cfg


class OslTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
  """Native mjlab checkpoints; mixed-actuator ONNX export is deferred."""

  def save(self, path: str, infos=None):
    MjlabOnPolicyRunner.save(self, path, infos)


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
