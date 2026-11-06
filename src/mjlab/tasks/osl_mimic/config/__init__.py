"""Task registration: musclemimic's three osl_fullbody configs plus a robot CLF.

Each comes with the rigid and the compliant socket. Train with, e.g.::

  uv run train Mjlab-Mimic-OSL-Fullbody-MSK-CLF \\
    --env.commands.motion.motion-file <100 Hz clipper --format mjlab clip>
"""

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import OslMimicOnPolicyRunner
from .env_cfgs import Reward, osl_mimic_env_cfg
from .rl_cfg import osl_mimic_runner_cfg

# task id -> (model, socket, reward, experiment name)
_VARIANTS: dict[str, tuple[Model, Socket, Reward, str]] = {}
for _model, _model_tag in (("robot", ""), ("msk", "-MSK")):
  for _reward, _reward_tag in (("mimic", ""), ("clf", "-CLF")):
    for _socket, _socket_tag in (("rigid", ""), ("compliant", "-Compliant")):
      _task_id = f"Mjlab-Mimic-OSL-Fullbody{_model_tag}{_reward_tag}{_socket_tag}"
      _experiment = "_".join(
        part
        for part in (
          "osl_fullbody_mimic",
          "msk" if _model == "msk" else "",
          "clf" if _reward == "clf" else "",
          "compliant" if _socket == "compliant" else "",
        )
        if part
      )
      _VARIANTS[_task_id] = (_model, _socket, _reward, _experiment)

for _task_id, (_model, _socket, _reward, _experiment) in _VARIANTS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_mimic_env_cfg(_model, _socket, _reward, mode="train"),
    play_env_cfg=osl_mimic_env_cfg(_model, _socket, _reward, mode="play"),
    rl_cfg=osl_mimic_runner_cfg(_model, _socket, _reward, _experiment),
    runner_cls=OslMimicOnPolicyRunner,
  )
