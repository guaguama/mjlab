"""Task registration: the round-1 MSK gait-library task, rigid and compliant socket.

Train with, e.g.::

  uv run train Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF \\
    --env.commands.motion.library-dir <100 Hz clipper gait library folder>
"""

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Socket
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import GaitLibraryOnPolicyRunner
from .env_cfgs import osl_gait_library_env_cfg
from .rl_cfg import osl_gait_library_runner_cfg

_VARIANTS: dict[str, tuple[Socket, str]] = {
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF": (
    "rigid",
    "osl_fullbody_gait_library_msk_clf",
  ),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF-Compliant": (
    "compliant",
    "osl_fullbody_gait_library_msk_clf_compliant",
  ),
}

for _task_id, (_socket, _experiment) in _VARIANTS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_gait_library_env_cfg(_socket, mode="train"),
    play_env_cfg=osl_gait_library_env_cfg(_socket, mode="play"),
    rl_cfg=osl_gait_library_runner_cfg(_socket, _experiment),
    runner_cls=GaitLibraryOnPolicyRunner,
  )
