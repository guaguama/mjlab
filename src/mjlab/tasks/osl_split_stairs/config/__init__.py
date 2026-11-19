"""Task registration: stage 2 on the stairs clip.

Every combination of model (``-ROBOT``, ``-MSK``), reward (MimicReward, ``-CLF``) and
socket (rigid, ``-Compliant``), as stairs stage 1's tasks
(``Mjlab-Stairs-OSL-Fullbody...``). A stairs stage-2 task starts from a stairs
stage-1 run of the same model, socket and OSL touch sites, on the same clip, as
unicycle stage 2 starts from a unicycle run (the reward may differ: the critics then
start fresh), e.g.::

  uv run train Mjlab-Split-Stairs-OSL-Fullbody-MSK-CLF-Compliant \\
    --env.commands.motion.motion-file <stairs>/terra_upstairs04_poses_optimized.npz \\
    --env.osl-touch-sites <the stage-1 run's> \\
    --agent.warm-start.path <stairs stage-1 run directory> \\
    --agent.warm-start.checkpoint model_<N>.pt

where ``<stairs>`` is clipper's ``outputs/mjlab/osl_fullbody/stairs``. The OSL
actor's sensors are unicycle stage 2's, with the same switches
(``--env.observations.osl...``; the heel contact needs a stage-1 run on the
``feet_extended`` or ``heel_extended`` touch sites). The reward is stairs stage
1's: the warm start refuses a stage-1 run trained with other reward settings (the
CLF decrease condition, the regularizers;
``--agent.warm-start.allow-reward-mismatch True`` starts anyway). Stage 2 continues
stage 1's iteration numbers in its own run directory
(``osl_fullbody_split_stairs[_msk][_clf][_compliant]``). Resume with
``--agent.resume``; play a checkpoint, which builds its own touch sites and OSL
sensors, with the clip in ``MJLAB_OSL_STAIRS_MOTION`` (play takes no env
overrides)::

  MJLAB_OSL_STAIRS_MOTION=<clip.npz> uv run play \\
    Mjlab-Split-Stairs-OSL-Fullbody-MSK-CLF-Compliant \\
    --checkpoint-file <stage-2 run>/model_<N>.pt
"""

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import MODEL_TAGS, Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_split.config import SPLIT_TASKS
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import StairsSplitRunner
from .env_cfgs import osl_split_stairs_env_cfg
from .rl_cfg import osl_split_stairs_runner_cfg

TASK_PREFIX = "Mjlab-Split-Stairs-OSL-Fullbody"
EXPERIMENT_PREFIX = "osl_fullbody_split_stairs"


def stairs_split_task_id(model: Model, socket: Socket, reward: Reward) -> str:
  """``TASK_PREFIX`` plus the model, reward and socket tags, in stage 1's order."""
  tags = MODEL_TAGS[model] + ("-CLF" if reward == "clf" else "")
  return TASK_PREFIX + tags + ("-Compliant" if socket == "compliant" else "")


def stairs_split_experiment_name(model: Model, socket: Socket, reward: Reward) -> str:
  parts = (
    EXPERIMENT_PREFIX,
    "msk" if model == "msk" else "",
    "clf" if reward == "clf" else "",
    "compliant" if socket == "compliant" else "",
  )
  return "_".join(part for part in parts if part)


# Unicycle stage 2's combinations. Importing them also completes unicycle stage 2's
# registration before this package's runner imports its runner (osl_split's config
# and runner import each other).
STAIRS_SPLIT_TASKS: dict[str, tuple[Model, Socket, Reward]] = {
  stairs_split_task_id(model, socket, reward): (model, socket, reward)
  for model, socket, reward in SPLIT_TASKS.values()
}
"""Task id -> (model, socket, reward)."""

for _task_id, (_model, _socket, _reward) in STAIRS_SPLIT_TASKS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_split_stairs_env_cfg(_model, _socket, _reward, "train"),
    play_env_cfg=osl_split_stairs_env_cfg(_model, _socket, _reward, "play"),
    rl_cfg=osl_split_stairs_runner_cfg(
      _model, _socket, _reward, stairs_split_experiment_name(_model, _socket, _reward)
    ),
    runner_cls=StairsSplitRunner,
  )
