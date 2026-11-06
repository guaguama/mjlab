"""Task registration: the gait-library tasks, rigid and compliant socket.

The robot model with MimicReward (``Mjlab-Gait-Library-OSL-Fullbody``) and the
musculoskeletal model with CLFReward (``-MSK-CLF``). Train with, e.g.::

  uv run train Mjlab-Gait-Library-OSL-Fullbody \\
    --env.commands.motion.library-dir <100 Hz clipper gait library folder>
"""

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import GaitLibraryOnPolicyRunner
from .env_cfgs import osl_gait_library_env_cfg
from .rl_cfg import osl_gait_library_runner_cfg

# task id -> (model, socket, reward, experiment name)
_VARIANTS: dict[str, tuple[Model, Socket, Reward, str]] = {
  "Mjlab-Gait-Library-OSL-Fullbody": (
    "robot",
    "rigid",
    "mimic",
    "osl_fullbody_gait_library",
  ),
  "Mjlab-Gait-Library-OSL-Fullbody-Compliant": (
    "robot",
    "compliant",
    "mimic",
    "osl_fullbody_gait_library_compliant",
  ),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF": (
    "msk",
    "rigid",
    "clf",
    "osl_fullbody_gait_library_msk_clf",
  ),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF-Compliant": (
    "msk",
    "compliant",
    "clf",
    "osl_fullbody_gait_library_msk_clf_compliant",
  ),
}

for _task_id, (_model, _socket, _reward, _experiment) in _VARIANTS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_gait_library_env_cfg(_model, _socket, _reward, mode="train"),
    play_env_cfg=osl_gait_library_env_cfg(_model, _socket, _reward, mode="play"),
    rl_cfg=osl_gait_library_runner_cfg(_model, _socket, _reward, _experiment),
    runner_cls=GaitLibraryOnPolicyRunner,
  )
