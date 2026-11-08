"""Task registration: the stage-2 split tasks.

Every combination of model (robot, ``-MSK``), reward (MimicReward, ``-CLF``) and
socket (rigid, ``-Compliant``) over the unicycle library, as stage 1's unicycle tasks
(``Mjlab-Gait-Library-Unicycle-OSL-Fullbody...``). A split task starts from a
stage-1 run of the same model and socket; the reward may differ (the reward switch:
the critics then start fresh). Convert the stage-1 checkpoint once on CPU, then
train, e.g.::

  uv run python -m mjlab.tasks.osl_split.scripts.split \\
    logs/rsl_rl/osl_fullbody_unicycle_msk_clf_compliant/<run> --out ws.pt
  uv run train Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant \\
    --env.commands.motion.library-dir <100 Hz clipper unicycle library folder> \\
    --agent.warm-start.path ws.pt

``--agent.warm-start.path`` also takes the stage-1 run directory or checkpoint itself,
converted in memory. Stage 2 continues stage 1's iteration numbers in its own run
directory (``osl_fullbody_split_unicycle[_msk][_clf][_compliant]``), with its
iteration counts (``--agent.max-iterations``, ``--agent.freeze.*``) local to stage 2.
Resume a stage-2 run with ``--agent.resume`` as usual. Play a stage-2 checkpoint,
both actors at their mean action, with the library in ``MJLAB_OSL_UNICYCLE_LIBRARY``
(play takes no env overrides)::

  uv run play Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant \\
    --checkpoint-file <stage-2 run>/model_<N>.pt

Every checkpoint comes with the OSL actor alone, the deployable part, as
``exported/osl_policy_<N>.onnx`` plus its metadata ``.json``. ``uv run --no-sync
python src/mjlab/tasks/osl_split/scripts/two_stage.py`` (a file, not ``-m``: see its
docstring) runs stage 1, the conversion and stage 2 in sequence.
"""

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import SplitOnPolicyRunner
from .env_cfgs import osl_split_env_cfg
from .rl_cfg import osl_split_runner_cfg

TASK_PREFIX = "Mjlab-Split-Unicycle-OSL-Fullbody"
EXPERIMENT_PREFIX = "osl_fullbody_split_unicycle"


def split_task_id(model: Model, socket: Socket, reward: Reward) -> str:
  """``TASK_PREFIX`` plus the model, reward and socket tags, in stage 1's order."""
  tags = ("-MSK" if model == "msk" else "") + ("-CLF" if reward == "clf" else "")
  return TASK_PREFIX + tags + ("-Compliant" if socket == "compliant" else "")


def split_experiment_name(model: Model, socket: Socket, reward: Reward) -> str:
  parts = (
    EXPERIMENT_PREFIX,
    "msk" if model == "msk" else "",
    "clf" if reward == "clf" else "",
    "compliant" if socket == "compliant" else "",
  )
  return "_".join(part for part in parts if part)


_MODELS: tuple[Model, ...] = ("robot", "msk")
_REWARDS: tuple[Reward, ...] = ("mimic", "clf")
_SOCKETS: tuple[Socket, ...] = ("rigid", "compliant")
SPLIT_TASKS: dict[str, tuple[Model, Socket, Reward]] = {
  split_task_id(model, socket, reward): (model, socket, reward)
  for model in _MODELS
  for reward in _REWARDS
  for socket in _SOCKETS
}
"""Task id -> (model, socket, reward)."""

for _task_id, (_model, _socket, _reward) in SPLIT_TASKS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_split_env_cfg(_model, _socket, _reward, "train"),
    play_env_cfg=osl_split_env_cfg(_model, _socket, _reward, "play"),
    rl_cfg=osl_split_runner_cfg(
      _model, _socket, _reward, split_experiment_name(_model, _socket, _reward)
    ),
    runner_cls=SplitOnPolicyRunner,
  )
