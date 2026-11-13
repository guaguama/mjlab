"""Runner settings for the stage-1.5 tasks: stage 2's, for the human alone.

Stage 1.5 trains stage 2's human actor beside the OSL walking controller, so its
settings are stage 2's (:func:`~mjlab.tasks.osl_split.config.rl_cfg.
osl_split_runner_cfg`): the warm start, the human's rate, ramp and held std, the
critic's ramp and floor, the PPO constants, validation and stress validation, the
checkpoint cadence. They differ only where the missing OSL actor changes them:

* ``algorithm`` (:class:`FsmPpoCfg`) is :class:`~mjlab.tasks.osl_fsm.rl.ppo.
  HumanOnlyPPO`, stage 2's human path. Stage 2's OSL settings (``osl_*``) must keep
  their defaults: here they would do nothing;
* ``freeze.human_actor_iterations`` is 0, so the human learns from the first
  iteration, its rate ramping up over stage 2's 50 iterations.
  ``freeze.critic_warmup_iterations`` stays 10, which here only ramps the critic's
  rate;
* the blend and the OSL actor's ONNX export are off: there is no OSL actor to blend
  the teacher into or to export;
* ``channels``: all four ends at the stage-1 weights, so both channels stay the
  stage-1 reward throughout. The human trains on its own channel (plus the twist
  term); the OSL channel only reports;
* ``validation`` (:class:`FsmValidationCfg`): the human's raw mean beside the
  controller in every evaluation, the zero-shot ones included, and no teacher;
* ``obs_groups``: the human actor and its critic read the ``actor`` group. The env
  keeps the ``osl`` group, the deployment contract, but no network reads it;
* ``osl_actor`` and ``osl_critic`` build nothing. The runner sets ``osl_critic`` to
  the human critic's block, since stage 2's warm start checks that block against
  stage 1's critic.

:func:`check_fsm_runner_cfg` refuses the settings that need an OSL actor, when a
config is constructed (on the command line, before the env is built) and again when
the runner starts. Iteration counts are stage-local, as in stage 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket
from mjlab.tasks.osl_fsm.rl.ppo import OSL_ONLY_KNOBS, check_osl_knobs
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_split.config.rl_cfg import (
  ChannelWeightsCfg,
  SplitBlendCfg,
  SplitChannelsCfg,
  SplitFreezeCfg,
  SplitPpoCfg,
  SplitRunnerCfg,
  SplitValidationCfg,
  osl_split_runner_cfg,
)
from mjlab.tasks.osl_split.rl.runner import FAMILY_SETTINGS

FSM_ITERATIONS = 1000
"""Default stage-1.5 length."""
SAVE_INTERVAL = 250
"""Stage-local iterations between checkpoints."""
FAMILIES = ("mean", "expected")
"""The evaluation families stage 1.5 plays: the human's raw mean or its
noise-averaged action, beside the controller. Stage 2's ``osl_sampled`` samples an
OSL actor, and its teacher families play the unified policy's OSL head; stage 1.5
has neither."""
OBS_GROUPS: dict[str, tuple[str, ...]] = {"actor": ("actor",), "critic": ("actor",)}
"""rsl_rl observation sets: the human actor and its critic read the stage-1
``actor`` group, as in stage 2."""
WANDB_TAGS = ("fsm", "stage1.5")
"""Stage 1.5's tags, in place of stage 2's ``split`` after its model, reward and
library tags."""
VALIDATION_FAMILY_FIELDS = (*FAMILY_SETTINGS, "teacher")
"""The validation settings :class:`FsmValidationCfg` sets itself; the factory takes
every other one from stage 2."""


@dataclass
class FsmPpoCfg(SplitPpoCfg):
  """:class:`~mjlab.tasks.osl_fsm.rl.ppo.HumanOnlyPPO`: stage 2's settings, of which
  the human's are used.

  Stage 2's OSL settings (``osl_critic_lr``, ``osl_actor_lr``, ``osl_clip_param``,
  ``osl_entropy_coef``, ``osl_value_loss_coef``, ``osl_critic_obs``,
  ``osl_critic_init``) have no network to act on, so any value but stage 2's default
  is refused, as stage 1's knobs are.
  """

  class_name: str = "mjlab.tasks.osl_fsm.rl.ppo:HumanOnlyPPO"

  def __post_init__(self) -> None:
    super().__post_init__()
    # On the command line this fails before the env is built; construct_algorithm
    # checks the runner's dict again.
    check_osl_knobs({name: getattr(self, name) for name in OSL_ONLY_KNOBS})


@dataclass
class FsmValidationCfg(SplitValidationCfg):
  """Stage 1.5's in-training evaluations: stage 2's protocol, with the human beside
  the controller.

  Every family the settings name must be one of :data:`FAMILIES`: ``"mean"`` (the
  human's raw mean, ``Validation/`` and ``Stress/``, the deployment metric) or
  ``"expected"`` (its noise-averaged action). The zero-shot evaluations, before the
  first update, play the stage-1 human beside the controller: the floor stage 1.5
  starts from, in validation and in stress validation alike.
  """

  inference: tuple[str, ...] = ("mean",)
  zero_shot_inference: tuple[str, ...] = ("mean",)
  teacher: bool = False
  """Refused in stage 1.5 (keep False): the teacher is the unified policy's OSL
  head, and the controller drives the OSL here."""
  stress_inference: tuple[str, ...] = ("mean",)
  zero_shot_stress_inference: tuple[str, ...] = ("mean",)


@dataclass
class FsmRunnerCfg(SplitRunnerCfg):
  """Stage 2's runner settings with stage 1.5's changes (see the module docstring).

  The defaults are stage 1.5's own wherever a stage-2 default would be refused (the
  blend, the export, the families, the teacher), so a bare ``FsmRunnerCfg()`` is a
  valid config; :func:`osl_fsm_runner_cfg` builds each task's.
  """

  # Narrower types than the base fields (pyright's
  # reportIncompatibleVariableOverride), so tyro and the runner see the subclasses.
  algorithm: FsmPpoCfg = field(default_factory=FsmPpoCfg)  # pyright: ignore
  validation: FsmValidationCfg = field(  # pyright: ignore
    default_factory=FsmValidationCfg
  )
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: dict(OBS_GROUPS)
  )
  freeze: SplitFreezeCfg = field(
    default_factory=lambda: SplitFreezeCfg(human_actor_iterations=0)
  )
  blend: SplitBlendCfg = field(default_factory=lambda: SplitBlendCfg(enabled=False))
  """Refused in stage 1.5 (keep ``enabled`` False): there is no OSL actor to blend
  the teacher into."""
  channels: SplitChannelsCfg = field(
    default_factory=lambda: SplitChannelsCfg(osl_end=ChannelWeightsCfg())
  )
  export_osl_onnx: bool = False
  """Refused in stage 1.5 (keep False): there is no OSL actor to export. The
  stage-1.5 bundle exporter is ``python -m mjlab.tasks.osl_fsm.export``."""

  def __post_init__(self) -> None:
    check_fsm_runner_cfg(
      {
        "validation": asdict(self.validation),
        "blend": asdict(self.blend),
        "export_osl_onnx": self.export_osl_onnx,
      }
    )


def check_fsm_runner_cfg(train_cfg: Mapping[str, Any]) -> None:
  """Refuse the runner settings that need an OSL actor.

  ``train_cfg`` is the runner's ``asdict`` of a :class:`FsmRunnerCfg` (only its
  ``validation``, ``blend`` and ``export_osl_onnx`` are read).

  Raises:
    ValueError: Naming each refused setting's flag: a validation family other than
      :data:`FAMILIES`, the teacher, the blend or the OSL export.
  """
  validation = train_cfg.get("validation") or {}
  lines: list[str] = []
  for setting in FAMILY_SETTINGS:
    families = validation.get(setting) or ()
    families = (families,) if isinstance(families, str) else tuple(families)
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
      flag = "--agent.validation." + setting.replace("_", "-")
      lines.append(
        f"{flag} names {unknown}: stage 1.5 plays the human's {list(FAMILIES)} "
        "beside the controller, with no OSL actor to sample and no teacher."
      )
  if validation.get("teacher", False):
    lines.append(
      "--agent.validation.teacher True: the teacher is the unified policy's OSL "
      "head, and the controller drives the OSL; the zero-shot evaluations already "
      "play the stage-1 human beside it."
    )
  if (train_cfg.get("blend") or {}).get("enabled", False):
    lines.append(
      "--agent.blend.enabled True: there is no OSL actor to blend the teacher into."
    )
  if train_cfg.get("export_osl_onnx", False):
    lines.append(
      "--agent.export-osl-onnx True: there is no OSL actor to export (the stage-1.5 "
      "bundle exporter is `python -m mjlab.tasks.osl_fsm.export`)."
    )
  if lines:
    raise ValueError(
      "Stage 1.5 has no OSL actor (the walking controller drives the OSL), so these "
      "settings are refused:\n" + "\n".join(f"  {line}" for line in lines)
    )


def osl_fsm_runner_cfg(
  model: Model, socket: Socket, reward: Reward, experiment_name: str
) -> FsmRunnerCfg:
  """Stage 2's runner settings for this task, with stage 1.5's changes (see the
  module docstring)."""
  base = osl_split_runner_cfg(model, socket, reward, experiment_name)
  changed = {
    "algorithm",
    "validation",
    "obs_groups",
    "freeze",
    "blend",
    "channels",
    "export_osl_onnx",
    "wandb_tags",
    "max_iterations",
    "save_interval",
  }
  shared = {
    f.name: getattr(base, f.name) for f in fields(base) if f.name not in changed
  }
  algorithm = FsmPpoCfg(
    **{
      f.name: getattr(base.algorithm, f.name)
      for f in fields(base.algorithm)
      if f.name != "class_name"
    }
  )
  validation = FsmValidationCfg(
    **{
      f.name: getattr(base.validation, f.name)
      for f in fields(base.validation)
      if f.name not in VALIDATION_FAMILY_FIELDS
    }
  )
  channels = base.channels
  return FsmRunnerCfg(
    **shared,
    algorithm=algorithm,
    validation=validation,
    obs_groups=dict(OBS_GROUPS),
    freeze=replace(base.freeze, human_actor_iterations=0),
    blend=replace(base.blend, enabled=False),
    # The OSL channel ends where it starts, at the stage-1 weights: no anneal.
    channels=replace(channels, osl_end=replace(channels.osl_start)),
    export_osl_onnx=False,
    wandb_tags=(*(tag for tag in base.wandb_tags if tag != "split"), *WANDB_TAGS),
    max_iterations=FSM_ITERATIONS,
    save_interval=SAVE_INTERVAL,
  )
