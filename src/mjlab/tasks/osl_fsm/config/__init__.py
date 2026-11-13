"""Task registration: the stage-1.5 tasks.

Every combination of model (``-ROBOT``, ``-MSK``), reward (MimicReward, ``-CLF``) and
socket (rigid, ``-Compliant``) over the unicycle library, as stage 2's split tasks.
A stage-1.5 task starts from a stage-1 run of the same model, socket and OSL touch
sites, as stage 2 does (the reward may differ: the critic then starts fresh), e.g.::

  uv run train Mjlab-FSM-Unicycle-OSL-Fullbody-MSK-CLF-Compliant \\
    --env.commands.motion.library-dir <100 Hz clipper unicycle library folder> \\
    --env.osl-touch-sites <the stage-1 run's> \\
    --agent.warm-start.path <stage-1 run directory, model_N.pt or converted file>

The OSL controller's settings are ``--env.osl-fsm.rules`` (``myosuite``, the
default, or ``sdk``), ``--env.osl-fsm.swing-load-exit`` (the sdk rules only),
``--env.osl-fsm.body-mass`` (kg; the model's own by default) and
``--env.osl-fsm.initial-state`` (``pose``, the default, or ``e_stance``). Stage 1.5
continues stage 1's iteration numbers in its own run directory
(``osl_fullbody_fsm_unicycle[_msk][_clf][_compliant]``), its iteration counts local
to it. Resume with ``--agent.resume`` and the run's controller flags; play a
checkpoint, which builds its own controller and touch sites, with the library in
``MJLAB_OSL_UNICYCLE_LIBRARY`` (play takes no env overrides)::

  uv run play Mjlab-FSM-Unicycle-OSL-Fullbody-MSK-CLF-Compliant \\
    --checkpoint-file <stage-1.5 run>/model_<N>.pt
"""

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import MODEL_TAGS, Model, Socket
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_split.config import SPLIT_TASKS
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import FsmSplitRunner
from .env_cfgs import osl_fsm_env_cfg
from .rl_cfg import osl_fsm_runner_cfg

TASK_PREFIX = "Mjlab-FSM-Unicycle-OSL-Fullbody"
EXPERIMENT_PREFIX = "osl_fullbody_fsm_unicycle"


def fsm_task_id(model: Model, socket: Socket, reward: Reward) -> str:
  """``TASK_PREFIX`` plus the model, reward and socket tags, in stage 1's order."""
  tags = MODEL_TAGS[model] + ("-CLF" if reward == "clf" else "")
  return TASK_PREFIX + tags + ("-Compliant" if socket == "compliant" else "")


def fsm_experiment_name(model: Model, socket: Socket, reward: Reward) -> str:
  parts = (
    EXPERIMENT_PREFIX,
    "msk" if model == "msk" else "",
    "clf" if reward == "clf" else "",
    "compliant" if socket == "compliant" else "",
  )
  return "_".join(part for part in parts if part)


# Stage 2's combinations. Importing them also completes stage 2's registration
# before this package's runner imports stage 2's (the walker reaches osl_fsm first).
FSM_TASKS: dict[str, tuple[Model, Socket, Reward]] = {
  fsm_task_id(model, socket, reward): (model, socket, reward)
  for model, socket, reward in SPLIT_TASKS.values()
}
"""Task id -> (model, socket, reward)."""

for _task_id, (_model, _socket, _reward) in FSM_TASKS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=osl_fsm_env_cfg(_model, _socket, _reward, "train"),
    play_env_cfg=osl_fsm_env_cfg(_model, _socket, _reward, "play"),
    rl_cfg=osl_fsm_runner_cfg(
      _model, _socket, _reward, fsm_experiment_name(_model, _socket, _reward)
    ),
    runner_cls=FsmSplitRunner,
  )
