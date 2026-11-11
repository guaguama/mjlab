"""Task registration: stair climbing on one reference clip, stage 1 (unified policy).

Every combination of model (``-ROBOT``, ``-MSK``), reward (MimicReward, ``-CLF``) and
socket (rigid, ``-Compliant``). A clip is clipper's 100 Hz ``--format mjlab`` NPZ
with TERRA's terrain sidecar ``<clip>_terrain.json`` beside it. Train with, e.g.::

  uv run train Mjlab-Stairs-OSL-Fullbody-MSK-CLF \\
    --env.commands.motion.motion-file <stairs>/terra_upstairs04_poses.npz

where ``<stairs>`` is clipper's ``outputs/mjlab/osl_fullbody/stairs``.

``--env.commands.motion.terrain-file <json>`` overrides the sidecar. Play takes no
``--env`` overrides, so it reads the clip from ``MJLAB_OSL_STAIRS_MOTION``::

  MJLAB_OSL_STAIRS_MOTION=<clip.npz> uv run play \\
    Mjlab-Stairs-OSL-Fullbody-ROBOT --checkpoint-file <model_N.pt>
"""

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import MODEL_TAGS, Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import OslStairsOnPolicyRunner
from .env_cfgs import osl_stairs_env_cfg
from .rl_cfg import osl_stairs_runner_cfg

_MODELS: tuple[tuple[Model, str], ...] = tuple(MODEL_TAGS.items())
_REWARDS: tuple[tuple[Reward, str], ...] = (("mimic", ""), ("clf", "-CLF"))
_SOCKETS: tuple[tuple[Socket, str], ...] = (("rigid", ""), ("compliant", "-Compliant"))

# task id -> (model, socket, reward, experiment name)
_VARIANTS: dict[str, tuple[Model, Socket, Reward, str]] = {}
for _model, _model_tag in _MODELS:
  for _reward, _reward_tag in _REWARDS:
    for _socket, _socket_tag in _SOCKETS:
      _task_id = f"Mjlab-Stairs-OSL-Fullbody{_model_tag}{_reward_tag}{_socket_tag}"
      _experiment = "_".join(
        part
        for part in (
          "osl_fullbody_stairs",
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
    env_cfg=osl_stairs_env_cfg(_model, _socket, _reward, mode="train"),
    play_env_cfg=osl_stairs_env_cfg(_model, _socket, _reward, mode="play"),
    rl_cfg=osl_stairs_runner_cfg(_model, _socket, _reward, _experiment),
    runner_cls=OslStairsOnPolicyRunner,
  )
