"""Task registration: the gait-library tasks.

The forward-walking tasks of round 1: the robot model with MimicReward
(``Mjlab-Gait-Library-OSL-Fullbody``) and the musculoskeletal model with CLFReward
(``-MSK-CLF``), rigid and compliant socket. The unicycle tasks, with domain
randomization: every combination of model (robot, ``-MSK``), reward (MimicReward,
``-CLF``) and socket (rigid, ``-Compliant``). Train with, e.g.::

  uv run train Mjlab-Gait-Library-OSL-Fullbody \\
    --env.commands.motion.library-dir <100 Hz clipper forward library folder>
  uv run train Mjlab-Gait-Library-Unicycle-OSL-Fullbody-MSK-CLF \\
    --env.commands.motion.library-dir <100 Hz clipper unicycle library folder>
"""

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import GaitLibraryOnPolicyRunner
from .env_cfgs import Variant, osl_gait_library_env_cfg
from .rl_cfg import osl_gait_library_runner_cfg

# task id -> (model, socket, reward, variant, experiment name)
_VARIANTS: dict[str, tuple[Model, Socket, Reward, Variant, str]] = {
  "Mjlab-Gait-Library-OSL-Fullbody": (
    "robot",
    "rigid",
    "mimic",
    "walk_fwd",
    "osl_fullbody_gait_library",
  ),
  "Mjlab-Gait-Library-OSL-Fullbody-Compliant": (
    "robot",
    "compliant",
    "mimic",
    "walk_fwd",
    "osl_fullbody_gait_library_compliant",
  ),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF": (
    "msk",
    "rigid",
    "clf",
    "walk_fwd",
    "osl_fullbody_gait_library_msk_clf",
  ),
  "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF-Compliant": (
    "msk",
    "compliant",
    "clf",
    "walk_fwd",
    "osl_fullbody_gait_library_msk_clf_compliant",
  ),
}
_MODELS: tuple[tuple[Model, str], ...] = (("robot", ""), ("msk", "-MSK"))
_REWARDS: tuple[tuple[Reward, str], ...] = (("mimic", ""), ("clf", "-CLF"))
_SOCKETS: tuple[tuple[Socket, str], ...] = (("rigid", ""), ("compliant", "-Compliant"))
for _model, _model_tag in _MODELS:
  for _reward, _reward_tag in _REWARDS:
    for _socket, _socket_tag in _SOCKETS:
      _tags = _model_tag + _reward_tag + _socket_tag
      _task_id = f"Mjlab-Gait-Library-Unicycle-OSL-Fullbody{_tags}"
      _experiment = "_".join(
        part
        for part in (
          "osl_fullbody_unicycle",
          "msk" if _model == "msk" else "",
          "clf" if _reward == "clf" else "",
          "compliant" if _socket == "compliant" else "",
        )
        if part
      )
      _VARIANTS[_task_id] = (_model, _socket, _reward, "unicycle", _experiment)

for _task_id, (_model, _socket, _reward, _variant, _experiment) in _VARIANTS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_gait_library_env_cfg(_model, _socket, _reward, "train", _variant),
    play_env_cfg=osl_gait_library_env_cfg(_model, _socket, _reward, "play", _variant),
    rl_cfg=osl_gait_library_runner_cfg(_model, _socket, _reward, _experiment, _variant),
    runner_cls=GaitLibraryOnPolicyRunner,
  )
