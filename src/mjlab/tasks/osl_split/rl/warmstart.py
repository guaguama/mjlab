"""The stage-1 to stage-2 warm start: the unified policy, split.

Stage 1 trains one actor over every action row. Its output layer is a plain linear
map from the trunk's last features to the action means, so dropping the two OSL rows
(and their two log-stds) leaves an actor that computes the human's actions *exactly*
as the unified one did, from the same observations and the same normalizer
statistics. Stage 2's human observation is the stage-1 ``actor`` group unchanged, so
nothing else needs surgery. A :class:`WarmStart` holds:

* ``human_actor``: the unified actor's state dict with its output layer and log-std
  restricted to the human's rows. Those are ``ActionLayout.rows("human")``, *not* a
  prefix or suffix in general: the musculoskeletal layout puts the OSL first. The std
  is copied verbatim, and stage 2 holds it fixed by default: the human's mean
  commands were learned at that std (0.07 at the end of a stage-1 recipe run, where
  the raw mean is the gait), so another std changes what they do;
* ``human_critic``: the stage-1 critic, verbatim. Stage 2 copies it into both
  critics, unless the reward type switches;
* ``teacher_head``: the two OSL rows of the output layer and their log-stds: the
  unified policy's OSL controller, on the human trunk's penultimate features (stage
  2's teacher blend and zero-shot teacher check);
* ``return_stats``: the reward normalizer's running return mean, variance and count.
  The stage-1 critic learned returns divided by that standard deviation, so a copied
  critic needs it to keep its scale; stage 2 resets the count (about 2e9 at the end
  of a recipe run) so the statistics can follow the new reward channels;
* ``lineage``: where it came from, as plain types (also written as YAML): the run,
  checkpoint, iteration and step counters, the task, the action layout, both
  architectures, the stds and the W&B run, and stage 1's state at its end: its last
  learning rate and std cap, the env settings it trained with (CLF penalty weight,
  twist weight, root-deviation limit, with the env stages then in force), the
  library release and whether the run had finished. Stage 2 compares them with its
  own settings when it starts (warnings, and a refusal for another ``pen_w``).

What is *not* carried: the optimizer state and its update counter. Stage 1 runs one
MuonAdam over actor and critic on its own schedule (the recipe decays to 10% of the
rate and holds); stage 2 gives every module its own optimizer and schedule.

Between the stages the reward type may switch (CLF to MimicReward or back; stage 2
then retrains its critics from scratch), but the model, the socket and the OSL
foot's touch sites may not: each changes the human actor's inputs or outputs
(:func:`check_compatible`). Nor may the architecture of a module that takes
stage-1 weights: another activation or layer-norm epsilon loads strictly but
computes something else (:func:`architecture_mismatches`).

Everything here runs on CPU and never touches CUDA. A stage-1 checkpoint (about
0.4 GB, half of it optimizer state) is loaded memory-mapped, and the verification
rebuilds both networks from the run's ``params/agent.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast, get_args

import torch
import torch.nn.functional as F
import yaml
from tensordict import TensorDict

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  LEGACY_TOUCH_SITES,
  Model,
  Socket,
  TouchSites,
  check_touch_sites,
)
from mjlab.tasks.osl_gait_library.export import read_config
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward, _actions
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_mimic.rl.ppo import geometric_schedule
from mjlab.tasks.osl_mimic.touch_sites import (
  CLI_FLAG,
  ENV_FIELD,
  ResolvedTouchSites,
  load_checkpoint,
  run_touch_sites,
)
from mjlab.tasks.osl_split.scripts.two_stage import planned_last_iteration

from .layout import ActionLayout

WARM_START_FORMAT = "mjlab.osl_split.warm_start/1"
"""Stored under ``"format"`` in every warm-start file; bump it on layout changes."""
LINEAGE_KEYS: tuple[str, ...] = (
  "format",
  "stage1_run_dir",
  "stage1_checkpoint",
  "iter",
  "common_step_counter",
  "env_steps",
  "model",
  "socket",
  "reward",
  "variant",
  "experiment_name",
  "obs_dim",
  "action_terms",
  "human_rows",
  "osl_rows",
  "actor_cfg",
  "critic_cfg",
  "num_updates_done",
  "human_std",
  "osl_std",
  "wandb_run_id",
  "created",
  "verified_max_abs_diff",
)
"""Keys every lineage carries."""
OPTIONAL_LINEAGE_KEYS: tuple[str, ...] = (
  "library_dir",
  "library_release",
  "stage1_lr",
  "stage1_std_cap",
  "stage1_env",
  "stage1_planned_last_iteration",
  ENV_FIELD,
)
"""Keys converted files record beyond :data:`LINEAGE_KEYS` (older files lack some,
and they load all the same):

* ``library_dir``: the stage-1 motion library (``None`` when the run's
  ``params/env.yaml`` lacks it); ``library_release``: its ``{"release",
  "sha256"}`` (:func:`library_release`), when readable where it was converted;
* ``stage1_lr``: the rate of stage 1's last update; ``stage1_std_cap``: the std cap
  its final std was projected under (None without a cap);
* ``stage1_env``: the env settings of stage 1's last update
  (:func:`stage1_env_settings`), None without ``params/env.yaml``;
* ``stage1_planned_last_iteration``: the iteration the run was set to end at (None
  when its settings do not say);
* ``osl_touch_sites``: the OSL foot touch sites stage 1 trained with
  (``touch_sites.run_touch_sites`` of the stage-1 checkpoint). A file converted
  before 2026-11-13 lacks it; :func:`lineage_touch_sites` then asks the stage-1
  checkpoint the lineage names."""
HUMAN_STD_WARNING = 0.15
"""A median human std above this draws a warning: stage 2 holds it fixed, and a
policy trained at a large std leans on its noise (the recipe ends at 0.07)."""
ENV_STAGE_SETTINGS = {
  "rewards.twist.weight": "twist_weight",
  "terminations.root_deviation.threshold": "root_deviation",
}
"""Stage-1 env-stage paths and the ``stage1_env`` setting each overrides."""
VERIFY_ATOL = 1e-5
"""Bound on the split's deviation from the unified actor. The split is exact up to
float32 summation order in the output layer (1.2e-6 on the MSK-CLF-compliant run's
``model_610.pt``)."""
MODEL_CLASS = "mjlab.tasks.osl_mimic.rl.models:MuscleMimicMLPModel"
"""The only stage-1 network class the converter understands."""
LOG_STD = "distribution.log_std_param"
"""The actor's state-independent log-std (rsl_rl ``GaussianDistribution``,
``std_type="log"``)."""

_STAGE1_KEYS = (
  "actor_state_dict",
  "critic_state_dict",
  "muscle_mimic_state",
  "iter",
  "infos",
)
_CHECKPOINT_NAME = re.compile(r"model_(\d+)\.pt")
_LINEAR_WEIGHT = re.compile(r"mlp\.(\d+)\.weight")
_PLAIN_NAME = re.compile(r"[A-Za-z0-9_]+")
_ARCHITECTURE_KEYS = (
  "hidden_dims",
  "activation",
  "layer_norm",
  "layer_norm_eps",
  "obs_normalization",
)
_DISTRIBUTION_KEYS = ("std_type", "std_range")
_UNSET = object()
"""A model-config key one side does not set (:func:`architecture_mismatches`)."""


class _AgentYamlLoader(yaml.SafeLoader):
  """``yaml.SafeLoader`` plus the ``!!python/tuple`` tag that ``dump_yaml`` writes
  for tuples (hidden dims, observation groups, W&B tags); nothing else unsafe."""


def _construct_tuple(loader: yaml.SafeLoader, node: yaml.SequenceNode) -> tuple:
  return tuple(loader.construct_sequence(node, deep=True))


_AgentYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)


##
# The warm start.
##


@dataclass
class WarmStart:
  """What stage 2 starts from; see the module docstring.

  Tensors are CPU float32 (the return statistics 0-dim). ``teacher_head`` holds
  ``weight`` (OSL actions x last hidden width), ``bias`` and ``log_std`` (OSL
  actions), in layout order (knee, then ankle). ``lineage`` holds plain types only;
  :data:`LINEAGE_KEYS` lists what it carries.
  """

  human_actor: dict[str, torch.Tensor]
  """A ``MuscleMimicMLPModel`` state dict with ``layout.dim("human")`` outputs."""
  human_critic: dict[str, torch.Tensor]
  """The stage-1 critic's state dict, verbatim."""
  teacher_head: dict[str, torch.Tensor]
  return_stats: dict[str, torch.Tensor]
  """``mean``, ``var`` and ``count`` of the stage-1 return normalizer, 0-dim."""
  lineage: dict[str, Any]

  @property
  def layout(self) -> ActionLayout:
    """The stage-1 action layout, which stage 2's action manager must reproduce."""
    return ActionLayout.from_list(self.lineage["action_terms"])

  def save(self, path: str | Path) -> None:
    """Writes the warm start (``torch.save``) and its lineage as YAML beside it
    (:func:`lineage_yaml_path`). The file loads with ``weights_only=True``."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lineage = _plain(self.lineage)
    payload = {
      "format": WARM_START_FORMAT,
      "human_actor": _cpu(self.human_actor),
      "human_critic": _cpu(self.human_critic),
      "teacher_head": _cpu(self.teacher_head),
      "return_stats": _cpu(self.return_stats),
      "lineage": lineage,
    }
    # Replace atomically: a half-written file would only fail at stage 2's start.
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    lineage_yaml_path(path).write_text(
      yaml.safe_dump(lineage, sort_keys=False, default_flow_style=None, width=88)
    )

  @classmethod
  def load(cls, path: str | Path) -> WarmStart:
    """Reads a file written by :meth:`save`, on CPU."""
    path = Path(path).expanduser()
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "format" not in data:
      raise ValueError(
        f"{path} is not a warm-start file. Convert a stage-1 run with "
        "`python -m mjlab.tasks.osl_split.scripts.split`, or pass the run itself "
        "to load_warm_start()."
      )
    if data["format"] != WARM_START_FORMAT:
      raise ValueError(
        f"{path} has format {data['format']!r}; this code reads "
        f"{WARM_START_FORMAT!r}. Convert the stage-1 run again."
      )
    lineage = dict(data["lineage"])
    missing = [key for key in LINEAGE_KEYS if key not in lineage]
    if missing:
      raise ValueError(f"{path}: the lineage lacks {missing}.")
    return cls(
      human_actor=dict(data["human_actor"]),
      human_critic=dict(data["human_critic"]),
      teacher_head=dict(data["teacher_head"]),
      return_stats=dict(data["return_stats"]),
      lineage=lineage,
    )

  def summary(self) -> str:
    """A few lines on where the warm start came from and what it holds."""
    lin = self.lineage
    terms = ", ".join(f"{name} {dim}" for name, dim in self.layout.terms)
    hs = lin["human_std"]
    osl_std = ", ".join(f"{s:.3g}" for s in lin["osl_std"])
    stats = {k: float(v) for k, v in self.return_stats.items()}
    diff = lin["verified_max_abs_diff"]
    return "\n".join(
      (
        f"Warm start from {lin['stage1_checkpoint']}",
        f"  stage 1: iteration {lin['iter']}, {lin['num_updates_done']} updates, "
        f"{lin['env_steps']:,} env steps, common_step_counter "
        f"{lin['common_step_counter']}; stage 2 starts at iteration "
        f"{lin['iter'] + 1}",
        f"  task: model {lin['model']}, socket {lin['socket']}, reward "
        f"{lin['reward']}, variant {lin['variant']} ({lin['experiment_name']})",
        f"  W&B run: {lin['wandb_run_id'] or 'not found'}",
        f"  observation: {lin['obs_dim']} dims; actions: {terms}",
        f"  human rows {_format_rows(lin['human_rows'])}: std min {hs['min']:.3g}, "
        f"median {hs['median']:.3g}, max {hs['max']:.3g}",
        f"  OSL rows {_format_rows(lin['osl_rows'])} (teacher head): std {osl_std}",
        f"  return statistics: mean {stats['mean']:.4g}, var {stats['var']:.4g}, "
        f"count {stats['count']:.4g}",
        "  verification: "
        + ("skipped" if diff is None else f"max |split - unified| {diff:.2g}"),
        *provenance_lines(lin),
        *(f"[WARN] {warning}" for warning in lineage_warnings(lin)),
      )
    )


def lineage_yaml_path(path: str | Path) -> Path:
  """Where :meth:`WarmStart.save` writes the lineage: ``<path>.yaml``."""
  path = Path(path).expanduser()
  return path.with_name(f"{path.name}.yaml")


##
# Conversion.
##


def resolve_stage1_checkpoint(source: str | Path, checkpoint: str = "") -> Path:
  """The stage-1 checkpoint ``source`` names.

  A file is used as is. A run directory resolves to ``checkpoint`` inside it, or
  else to its highest-numbered ``model_<N>.pt``, compared as numbers (``model_610``
  beats ``model_70``, which sorts last alphabetically).
  """
  path = Path(source).expanduser()
  if path.is_file():
    if checkpoint and Path(checkpoint).name != path.name:
      raise ValueError(
        f"{path} is a checkpoint file; checkpoint={checkpoint!r} only selects a "
        "file inside a run directory."
      )
    return path
  if not path.is_dir():
    raise FileNotFoundError(f"No stage-1 run directory or checkpoint at {path}.")
  if checkpoint:
    ckpt = path / checkpoint
    if not ckpt.is_file():
      raise FileNotFoundError(f"No checkpoint {checkpoint!r} in {path}.")
    return ckpt
  numbered: list[tuple[int, str, Path]] = []
  for entry in path.iterdir():
    match = _CHECKPOINT_NAME.fullmatch(entry.name)
    if match is not None and entry.is_file():
      numbered.append((int(match.group(1)), entry.name, entry))
  if not numbered:
    raise FileNotFoundError(f"No model_<N>.pt checkpoint in {path}.")
  return max(numbered)[2]


def default_output_path(ckpt: Path) -> Path:
  """``<run_dir>/warm_start/model_<N>.pt`` for ``<run_dir>/model_<N>.pt``.

  Beside the run it came from, but in a subdirectory, so neither
  :func:`resolve_stage1_checkpoint` nor a stage-1 ``--agent.resume`` (which match
  ``model_*.pt`` in the run directory itself) can mistake it for a checkpoint.
  """
  ckpt = Path(ckpt).expanduser()
  return ckpt.parent / "warm_start" / f"{ckpt.stem}.pt"


def build_warm_start(
  source: str | Path, checkpoint: str = "", verify: bool = True
) -> WarmStart:
  """Converts a stage-1 run (or one of its checkpoints) into a :class:`WarmStart`.

  The model, socket, reward and variant come from the run's ``params/agent.yaml``
  (its ``validation`` block), the architectures from its ``actor`` and ``critic``
  blocks. ``params/env.yaml`` cannot be loaded back as a config (its spec_fn is a
  closure); it is read as plain data for the motion library and stage 1's env
  settings (:func:`stage1_env_settings`). ``verify`` rebuilds the unified and the
  human actor on CPU and checks the split on random observations
  (:func:`verify_warm_start`); it costs a few seconds.
  """
  ckpt = resolve_stage1_checkpoint(source, checkpoint)
  run_dir = ckpt.parent
  data = _load_checkpoint(ckpt)
  _check_stage1_checkpoint(data, ckpt)
  agent = _read_agent_cfg(run_dir)
  model, socket, reward, variant = _stage1_task(agent, run_dir)
  actor: Mapping[str, torch.Tensor] = data["actor_state_dict"]
  critic: Mapping[str, torch.Tensor] = data["critic_state_dict"]
  state = data["muscle_mimic_state"]

  first, last = _linear_layers(actor, "actor", ckpt)
  obs_dim = int(actor[f"{first}.weight"].shape[1])
  out_dim = int(actor[f"{last}.weight"].shape[0])
  log_std = actor.get(LOG_STD)
  if log_std is None or tuple(log_std.shape) != (out_dim,):
    raise ValueError(
      f"{ckpt}: the actor needs a {LOG_STD} of shape ({out_dim},) "
      "(GaussianDistribution with std_type='log'), as stage 2's human actor has."
    )
  critic_first, critic_last = _linear_layers(critic, "critic", ckpt)
  critic_in = int(critic[f"{critic_first}.weight"].shape[1])
  critic_out = int(critic[f"{critic_last}.weight"].shape[0])
  if critic_in != obs_dim or critic_out != 1:
    raise ValueError(
      f"{ckpt}: the critic maps {critic_in} inputs to {critic_out}, but stage 2's "
      f"critics read the actor's {obs_dim}-dim observation and output one value."
    )

  layout = ActionLayout.for_model(model, out_dim)
  _check_layout(layout, model)
  human_rows, osl_rows = layout.rows("human"), layout.rows("osl")
  human_index = torch.tensor(human_rows, dtype=torch.long)
  osl_index = torch.tensor(osl_rows, dtype=torch.long)
  sliced = {f"{last}.weight", f"{last}.bias", LOG_STD}
  human_actor = {
    key: value.index_select(0, human_index) if key in sliced else value.detach().clone()
    for key, value in actor.items()
  }
  teacher_head = {
    "weight": actor[f"{last}.weight"].index_select(0, osl_index),
    "bias": actor[f"{last}.bias"].index_select(0, osl_index),
    "log_std": log_std.index_select(0, osl_index),
  }
  std = log_std.detach().double().exp()
  human_std = std.index_select(0, human_index).tolist()
  env_state = (data.get("infos") or {}).get("env_state") or {}
  if "common_step_counter" not in env_state:
    raise ValueError(f"{ckpt} has no infos['env_state']['common_step_counter'].")

  lineage: dict[str, Any] = {
    "format": WARM_START_FORMAT,
    "stage1_run_dir": str(run_dir.resolve()),
    "stage1_checkpoint": str(ckpt.resolve()),
    "iter": int(data["iter"]),
    "common_step_counter": int(env_state["common_step_counter"]),
    "env_steps": round(float(state["return_count"])),
    "model": model,
    "socket": socket,
    ENV_FIELD: run_touch_sites(ckpt, data).value,
    "reward": reward,
    "variant": variant,
    "experiment_name": agent.get("experiment_name"),
    "obs_dim": obs_dim,
    "action_terms": layout.to_list(),
    "human_rows": human_rows,
    "osl_rows": osl_rows,
    "actor_cfg": _plain(agent["actor"]),
    "critic_cfg": _plain(agent["critic"]),
    "num_updates_done": int(state["num_updates_done"]),
    "human_std": {
      "min": min(human_std),
      "median": statistics.median(human_std),
      "max": max(human_std),
    },
    "osl_std": std.index_select(0, osl_index).tolist(),
    "wandb_run_id": find_wandb_run_id(run_dir),
    "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    "verified_max_abs_diff": None,
    "library_dir": _library_dir(run_dir),
  }
  algorithm = agent.get("algorithm") or {}
  updates = int(state["num_updates_done"])
  lineage["library_release"] = library_release(lineage["library_dir"])
  lineage["stage1_lr"] = stage1_learning_rate(algorithm, max(updates - 1, 0))
  lineage["stage1_std_cap"] = stage1_std_cap(algorithm, updates)
  lineage["stage1_env"] = stage1_env_settings(run_dir, agent, updates)
  try:
    lineage["stage1_planned_last_iteration"] = planned_last_iteration(run_dir)
  except Exception:  # noqa: BLE001 - provenance only, never a reason to fail.
    lineage["stage1_planned_last_iteration"] = None
  ws = WarmStart(
    human_actor=human_actor,
    human_critic={key: value.detach().clone() for key, value in critic.items()},
    teacher_head=teacher_head,
    return_stats={
      name: _scalar(state[f"return_{name}"]) for name in ("mean", "var", "count")
    },
    lineage=lineage,
  )
  if verify:
    lineage["verified_max_abs_diff"] = verify_warm_start(ws, actor)
  return ws


def verify_warm_start(
  ws: WarmStart,
  stage1_actor: Mapping[str, torch.Tensor],
  *,
  atol: float = VERIFY_ATOL,
  num_samples: int = 64,
  seed: int = 0,
) -> float:
  """Checks the split against the unified actor and returns the largest deviation.

  Both actors are rebuilt as ``MuscleMimicMLPModel`` from the lineage's
  ``actor_cfg``, stage 1's architecture (strict loads), and run on observations
  drawn around the stage-1 normalizer's statistics:

  * ``human(x)`` must equal ``unified(x)[:, human_rows]``;
  * the teacher head on the human trunk's penultimate features must equal
    ``unified(x)[:, osl_rows]``;
  * the log-stds must be the unified ones' rows, exactly.

  Raises ``RuntimeError`` otherwise. CPU only, whatever the default device. This
  checks the split against stage 1's network, not that stage 2 builds the same one
  from its own config: :func:`architecture_mismatches` compares the two.
  """
  lineage = ws.lineage
  human_rows: list[int] = list(lineage["human_rows"])
  osl_rows: list[int] = list(lineage["osl_rows"])
  obs_dim = int(lineage["obs_dim"])
  kwargs = _model_kwargs(lineage["actor_cfg"])
  groups = {"actor": ["actor"]}
  with torch.device("cpu"):
    obs = TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1])
    unified = MuscleMimicMLPModel(
      obs, groups, "actor", len(human_rows) + len(osl_rows), **kwargs
    )
    unified.load_state_dict(dict(stage1_actor), strict=True)
    human = MuscleMimicMLPModel(obs, groups, "actor", len(human_rows), **kwargs)
    human.load_state_dict(ws.human_actor, strict=True)
    unified.set_norm_mode("frozen")
    human.set_norm_mode("frozen")

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, obs_dim, generator=generator)
    mean = stage1_actor.get("obs_normalizer.mean")
    var = stage1_actor.get("obs_normalizer.var")
    if mean is not None and var is not None:
      x = mean.float() + var.float().clamp(min=0.0).sqrt() * x
    batch = TensorDict({"actor": x}, batch_size=[num_samples])
    with torch.inference_mode():
      full = unified(batch)
      split = human(batch)
      features = human.mlp[:-1](human.get_latent(batch))
      teacher = F.linear(features, ws.teacher_head["weight"], ws.teacher_head["bias"])
  human_diff = float((split - full[:, human_rows]).abs().max())
  teacher_diff = float((teacher - full[:, osl_rows]).abs().max())
  log_std = stage1_actor[LOG_STD]
  std_exact = torch.equal(ws.human_actor[LOG_STD], log_std[human_rows]) and (
    torch.equal(ws.teacher_head["log_std"], log_std[osl_rows])
  )
  diff = max(human_diff, teacher_diff)
  if not diff <= atol or not std_exact:  # Also catches NaN.
    raise RuntimeError(
      "Warm-start verification failed: max |human - unified| = "
      f"{human_diff:.3g}, max |teacher - unified OSL| = {teacher_diff:.3g} "
      f"(atol {atol:g}); log-std rows "
      + ("exact." if std_exact else "differ from the unified actor's.")
    )
  return diff


def load_warm_start(path: str | Path, checkpoint: str = "") -> WarmStart:
  """What stage 2's ``--agent.warm-start.path`` points at, as a :class:`WarmStart`.

  A converted file (it carries ``"format"``) is loaded; a stage-1 run directory or
  checkpoint is converted in memory (and verified), which takes a few seconds more.
  """
  source = Path(path).expanduser()
  if source.is_file() and _is_warm_start_file(source):
    if checkpoint:
      raise ValueError(
        f"{source} is a warm-start file; checkpoint={checkpoint!r} only selects a "
        "file inside a stage-1 run directory."
      )
    return WarmStart.load(source)
  return build_warm_start(source, checkpoint)


def resolve_lineage_touch_sites(lineage: Mapping[str, Any]) -> ResolvedTouchSites:
  """The OSL foot touch sites stage 1 trained with, and where they were read.

  A lineage records them since 2026-11-13. One without them was converted by
  earlier code, which may have read a stage-1 run that did record them, so the
  stage-1 checkpoint it names decides when it is readable here
  (``touch_sites.run_touch_sites``: its key, else ``"original"``); ``"original"``
  otherwise.

  Raises:
    ValueError: For an unknown value, or a stage-1 checkpoint that contradicts its
      run's env.yaml.
  """
  value = lineage.get(ENV_FIELD)
  if value is not None:
    return ResolvedTouchSites(
      check_touch_sites(value, f"the warm start's {ENV_FIELD}"), "its lineage"
    )
  stage1 = lineage.get("stage1_checkpoint")
  if stage1 and Path(stage1).is_file():
    try:
      payload = load_checkpoint(stage1)
    except Exception:  # noqa: BLE001 - unreadable here: fall back to "original".
      payload = None
    if payload is not None:
      resolved = run_touch_sites(stage1, payload)
      return ResolvedTouchSites(
        resolved.value,
        f"not in its lineage, so from its stage-1 checkpoint {stage1}: "
        f"{resolved.source}",
      )
  return ResolvedTouchSites(
    LEGACY_TOUCH_SITES,
    "not recorded: the warm start was converted before the setting, and its "
    "stage-1 checkpoint is not readable here",
  )


def lineage_touch_sites(lineage: Mapping[str, Any]) -> TouchSites:
  """The OSL foot touch sites stage 1 trained with
  (:func:`resolve_lineage_touch_sites`)."""
  return resolve_lineage_touch_sites(lineage).value


def check_compatible(
  ws: WarmStart, *, model: str, socket: str, reward: str, touch_sites: str
) -> bool:
  """Whether stage 2 (``model``, ``socket``, ``reward``, OSL ``touch_sites``) may
  start from ``ws``.

  Returns True when the reward type differs from stage 1's (the reward switch: the
  copied critics would value the wrong reward, so stage 2 starts them fresh), False
  when it is the same. Raises ``ValueError`` on a model, socket or touch-site
  change: the human actor is the unified actor transplanted exactly, and each change
  would alter its inputs (for the socket, joints the stage-1 normalizer has never
  seen; for the touch sites, an ``r_foot`` reading it never saw; its single count
  would never give either statistics) or its outputs.
  """
  _check_choice("model", model, get_args(Model))
  _check_choice("socket", socket, get_args(Socket))
  _check_choice("reward", reward, get_args(Reward))
  check_touch_sites(touch_sites, "stage 2's OSL touch sites")
  lineage = ws.lineage
  source = lineage.get("stage1_run_dir", "the stage-1 run")
  if lineage["model"] != model:
    raise ValueError(
      f"The warm start ({source}) drives the {lineage['model']!r} model, but this "
      f"stage-2 task drives {model!r}. The human actor is the unified actor minus "
      "its OSL rows, so its observations and actions "
      f"({lineage['action_terms']}) would both change: the model cannot switch "
      "between stages. Train stage 2 on the stage-1 run's model."
    )
  if lineage["socket"] != socket:
    raise ValueError(
      f"The warm start ({source}) was trained with the {lineage['socket']!r} "
      f"socket, but this stage-2 task uses the {socket!r} socket. The compliant "
      "socket's four joints sit inside the human's joint_pos and joint_vel "
      "observations, so its inputs would change (and the stage-1 normalizer's "
      "single count would never give new inputs statistics): the socket cannot "
      "switch between stages. Only the reward may."
    )
  stage1_touch = resolve_lineage_touch_sites(lineage)
  if stage1_touch.value != touch_sites:
    raise ValueError(
      f"The warm start ({source}) was trained with the OSL touch sites "
      f"{stage1_touch.value!r} ({stage1_touch.source}), but this stage-2 task "
      f"builds {touch_sites!r}. The human actor reads the foot touch sensors, so "
      "its r_foot input would read what stage 1 never saw (and the stage-1 "
      "normalizer's single count would never adapt to it): the touch sites cannot "
      f"switch between stages. Train stage 2 with {CLI_FLAG} {stage1_touch.value}."
    )
  return lineage["reward"] != reward


def architecture_mismatches(
  lineage_cfg: Mapping[str, Any], stage2_cfg: Mapping[str, Any]
) -> list[str]:
  """How a stage-2 model config differs from the stage-1 one whose weights its
  module takes, one line per key; empty when they agree.

  ``lineage_cfg`` is a lineage's ``actor_cfg`` or ``critic_cfg`` (the run's
  agent.yaml block), ``stage2_cfg`` the stage-2 block of the module that receives
  those weights. A strict state-dict load refuses another depth, width, layer norm
  or normalizer, though with a cryptic error. It cannot see another
  ``activation``, ``layer_norm_eps`` or ``std_range``: the transplant would load
  and silently compute something else.

  Compared: ``hidden_dims`` (tuples and lists alike), ``activation``,
  ``layer_norm``, ``layer_norm_eps`` and ``obs_normalization``, where a key only
  one side sets differs; whether a distribution is configured (a missing
  ``distribution_cfg`` counts as None, since the runners drop a None one); and the
  distribution's ``std_type`` and ``std_range`` where both set them (stage 1's
  actor sets no ``std_range``). Each line reads ``<key>: stage 1 <value>, stage 2
  <value>``, the key dotted below the model block (``distribution_cfg.std_type``).
  """
  pairs = [
    (key, lineage_cfg.get(key, _UNSET), stage2_cfg.get(key, _UNSET))
    for key in _ARCHITECTURE_KEYS
  ]
  stage1_dist = lineage_cfg.get("distribution_cfg")
  stage2_dist = stage2_cfg.get("distribution_cfg")
  if isinstance(stage1_dist, Mapping) and isinstance(stage2_dist, Mapping):
    pairs += [
      (f"distribution_cfg.{key}", stage1_dist[key], stage2_dist[key])
      for key in _DISTRIBUTION_KEYS
      if key in stage1_dist and key in stage2_dist
    ]
  else:  # Equal when neither configures one.
    pairs.append(("distribution_cfg", stage1_dist, stage2_dist))
  return [
    f"{key}: stage 1 {_show(stage1)}, stage 2 {_show(stage2)}"
    for key, stage1, stage2 in pairs
    if _as_lists(stage1) != _as_lists(stage2)
  ]


def find_wandb_run_id(
  run_dir: str | Path, wandb_root: Path | None = None
) -> str | None:
  """The id of the W&B run that logged ``run_dir``, or None. Best effort.

  rsl_rl's W&B writer passes the run directory as ``config={"log_dir": ...}`` to
  ``wandb.init``. W&B echoes that config into ``<wandb>/run-*/logs/debug.log`` as a
  Python repr (``'log_dir': '<path>'``) and, for finished runs, writes it to
  ``files/config.yaml`` (a top-level ``log_dir:``, then ``value: <path>``). Only
  that entry is matched, with the exact resolved path, newest run first. A bare
  path would do for neither: rsl_rl also logs the whole train config to both files
  and W&B the command line to ``config.yaml``, so a later stage-2 run started from
  this one (``--agent.warm-start.path``, its directory or a file inside it) names
  it too. ``wandb_root`` defaults to the first ``wandb`` directory in ``run_dir``'s
  ancestors (``<repo>/wandb`` for ``<repo>/logs/...``), then the current
  directory's.
  """
  target = Path(run_dir).expanduser().resolve()
  # Neither matches a longer path (``<run>_restart``): a quote or the line's end
  # must follow.
  init_entry = f"'log_dir': {str(target)!r}"
  config_entry = re.compile(
    rf"^log_dir:[ \t]*\n[ \t]+value:[ \t]*{re.escape(str(target))}[ \t]*$",
    re.MULTILINE,
  )
  roots = [Path(wandb_root)] if wandb_root is not None else _wandb_roots(target)
  for root in roots:
    runs = [
      run
      for glob in ("run-*", "offline-run-*")
      for run in root.glob(glob)
      if run.is_dir()
    ]
    runs.sort(key=lambda run: run.name.removeprefix("offline-"), reverse=True)
    for run in runs:
      if init_entry in _read_text(run / "logs" / "debug.log") or config_entry.search(
        _read_text(run / "files" / "config.yaml")
      ):
        return run.name.rsplit("-", 1)[-1]
  return None


##
# Provenance: stage 1's state at its end.
##


def stage1_learning_rate(algorithm: Mapping[str, Any], update: int) -> float | None:
  """The rate ``MuscleMimicPPO`` trained ``update`` with, from an agent.yaml
  ``algorithm`` block (its ``scheduled_learning_rate``); None without a rate."""
  try:
    base = float(algorithm["learning_rate"])
  except (KeyError, TypeError, ValueError):
    return None
  updates = int(algorithm.get("lr_schedule_updates") or 0)
  if algorithm.get("lr_schedule", "linear") == "constant" or updates <= 0:
    return base
  done = max(update - int(algorithm.get("lr_schedule_offset") or 0), 0)
  fraction = max(1.0 - done / updates, 0.0)
  final = float(algorithm.get("lr_final_ratio") or 0.0)
  return base * (final + (1.0 - final) * fraction)


def stage1_std_cap(algorithm: Mapping[str, Any], update: int) -> float | None:
  """``MuscleMimicPPO.std_cap(update)`` from an agent.yaml ``algorithm`` block; None
  without a cap."""
  start = float(algorithm.get("std_cap_start") or 0.0)
  if start <= 0.0:
    return None
  end = float(algorithm.get("std_cap_end") or 0.0) or start
  return geometric_schedule(
    update,
    start,
    end,
    int(algorithm.get("std_cap_begin_update") or 0),
    int(algorithm.get("std_cap_end_update") or 0),
  )


def env_stages_in_force(
  stages: Iterable[Sequence[Any]], update: int
) -> dict[str, float]:
  """The staged values ``OslMimicOnPolicyRunner.apply_env_stages`` sets for
  ``update``, by path; a path whose stages have all ended is absent (the env config's
  own value holds)."""
  stages = [(int(u), str(path), float(v)) for u, path, v in stages]
  out: dict[str, float] = {}
  for path in dict.fromkeys(path for _, path, _ in stages):
    active = sorted((u, v) for u, p, v in stages if p == path and update < u)
    if active:
      out[path] = active[0][1]
  return out


def stage1_env_settings(
  run_dir: Path, agent: Mapping[str, Any], num_updates_done: int
) -> dict[str, Any] | None:
  """The env settings stage 1's last update trained with: ``clf_pen_w`` (None
  under MimicReward), ``twist_weight`` and ``root_deviation`` (the training
  limit), from ``params/env.yaml`` with the env stages then in force on top
  (recorded as ``env_stages_in_force``). None without a readable env.yaml."""
  try:
    env = read_config(run_dir / "params" / "env.yaml")
  except (OSError, yaml.YAMLError, ValueError):
    return None
  if not isinstance(env, Mapping):
    return None
  settings: dict[str, Any] = {
    "clf_pen_w": _number(_dig(env, "rewards", "mimic", "params", "clf", "pen_w")),
    "twist_weight": _number(_dig(env, "rewards", "twist", "weight")),
    "root_deviation": _number(
      _dig(env, "terminations", "root_deviation", "params", "threshold")
    ),
  }
  stages = env_stages_in_force(agent.get("env_stages") or (), num_updates_done - 1)
  for path, value in stages.items():
    if path in ENV_STAGE_SETTINGS:
      settings[ENV_STAGE_SETTINGS[path]] = value
  settings["env_stages_in_force"] = stages
  return settings


def env_cfg_settings(cfg: Any) -> dict[str, float | None]:
  """:func:`stage1_env_settings`' settings of a live env config (None where it
  has none: a MimicReward ``clf_pen_w``, a mode without the root-deviation
  limit)."""

  def term(section: str, name: str) -> Any:
    terms = getattr(cfg, section, None)
    return terms.get(name) if isinstance(terms, Mapping) else None

  mimic, twist, root = (
    term("rewards", "mimic"),
    term("rewards", "twist"),
    term("terminations", "root_deviation"),
  )
  clf = (getattr(mimic, "params", None) or {}).get("clf")
  pen_w = clf.get("pen_w") if isinstance(clf, Mapping) else getattr(clf, "pen_w", None)
  threshold = (getattr(root, "params", None) or {}).get("threshold")
  return {
    "clf_pen_w": _number(pen_w),
    "twist_weight": _number(getattr(twist, "weight", None)),
    "root_deviation": _number(threshold),
  }


def library_release(library_dir: str | Path | None) -> dict[str, str | None] | None:
  """A gait library's identity: ``release`` (``library.json``'s, else the folder
  name) and ``sha256`` (of ``library.json``; None where it cannot be read). None
  without a library."""
  if not library_dir:
    return None
  folder = Path(library_dir).expanduser()
  try:
    data = (folder / "library.json").read_bytes()
  except OSError:
    return {"release": folder.name or None, "sha256": None}
  name = None
  try:
    loaded = json.loads(data)
    name = loaded.get("release") if isinstance(loaded, Mapping) else None
  except ValueError:
    pass
  return {
    "release": str(name) if name else folder.name or None,
    "sha256": hashlib.sha256(data).hexdigest(),
  }


def compare_library_releases(
  stage1: Mapping[str, Any] | None, stage2: Mapping[str, Any] | None
) -> tuple[str, str] | None:
  """``("INFO" | "WARN", message)`` comparing two :func:`library_release` values, or
  None when either is unknown. The content hash decides where both have one (a
  cluster run's library sits at another path than this machine's copy); the release
  names otherwise."""
  if not stage1 or not stage2:
    return None
  names = (stage1.get("release"), stage2.get("release"))
  hashes = (stage1.get("sha256"), stage2.get("sha256"))
  if hashes[0] and hashes[1]:
    if hashes[0] == hashes[1]:
      return "INFO", f"Stage 1 and stage 2 use the same library release {names[1]}."
    if names[0] == names[1]:
      # library.json records its conversion's time and input path, so a release
      # converted again differs in bytes even when its clips do not.
      return (
        "WARN",
        f"Stage 1 and stage 2 use library {names[1]}, but their library.json "
        f"differ ({hashes[0][:12]} vs {hashes[1][:12]}): the same release name, "
        "converted again or edited. Check that its clips are the same.",
      )
    return (
      "WARN",
      f"Stage 1 trained on library {names[0]} (library.json {hashes[0][:12]}), "
      f"stage 2 uses {names[1]} ({hashes[1][:12]}): another library.",
    )
  if names[0] == names[1]:
    return (
      "INFO",
      f"Stage 1 and stage 2 use library {names[1]} (by name: stage 1's "
      "library.json was not readable when converted).",
    )
  return "WARN", f"Stage 1 trained on library {names[0]}, stage 2 uses {names[1]}."


def lineage_warnings(lineage: Mapping[str, Any]) -> list[str]:
  """What a warm start's lineage alone shows to be risky: a large human std (stage
  2 holds it fixed) and an unfinished stage-1 run."""
  warnings: list[str] = []
  median = (lineage.get("human_std") or {}).get("median")
  if median is not None and float(median) > HUMAN_STD_WARNING:
    warnings.append(
      f"The human's median std is {float(median):.3g}, above {HUMAN_STD_WARNING}: "
      "stage 2 holds it fixed, and a policy trained at a large std leans on its "
      "noise, so its raw mean may not walk (the stage-1 recipe ends at 0.07)."
    )
  planned = lineage.get("stage1_planned_last_iteration")
  if planned is not None and int(lineage["iter"]) < int(planned):
    warnings.append(
      f"The warm start is iteration {lineage['iter']} of a stage-1 run set to end at "
      f"{planned}: the run is unfinished (still training, stopped or crashed), or "
      "an earlier checkpoint was chosen."
    )
  return warnings


def provenance_lines(lineage: Mapping[str, Any]) -> list[str]:
  """Stage 1's recorded end state, for summaries."""
  touch = resolve_lineage_touch_sites(lineage)
  lines: list[str] = [
    f"  OSL touch sites: {touch.value}"
    + ("" if ENV_FIELD in lineage else f" ({touch.source})")
  ]
  if "stage1_lr" in lineage or "stage1_std_cap" in lineage:
    lr, cap = lineage.get("stage1_lr"), lineage.get("stage1_std_cap")
    shown_cap = "none" if cap is None and "stage1_std_cap" in lineage else _shown(cap)
    lines.append(
      f"  stage 1 at its end: learning rate {_shown(lr)}, std cap {shown_cap}"
    )
  env = lineage.get("stage1_env")
  if env:
    stages = env.get("env_stages_in_force") or {}
    lines.append(
      f"  stage-1 env: CLF pen_w {_shown(env.get('clf_pen_w'))}, twist weight "
      f"{_shown(env.get('twist_weight'))}, root deviation "
      f"{_shown(env.get('root_deviation'))}"
      + (f" (env stages in force: {stages})" if stages else "")
    )
  release = lineage.get("library_release")
  if release:
    digest = release.get("sha256")
    lines.append(
      f"  library: {release.get('release')}"
      + (f" (library.json sha256 {digest[:12]})" if digest else "")
    )
  return lines


##
# Helpers.
##


def _dig(tree: Any, *keys: str) -> Any:
  for key in keys:
    if not isinstance(tree, Mapping) or key not in tree:
      return None
    tree = tree[key]
  return tree


def _number(value: Any) -> float | None:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  return float(value) if math.isfinite(value) else None


def _shown(value: Any) -> str:
  return "not recorded" if value is None else f"{value:.3g}"


def _read_agent_cfg(run_dir: Path) -> dict[str, Any]:
  path = run_dir / "params" / "agent.yaml"
  if not path.is_file():
    raise FileNotFoundError(
      f"{path} not found. The converter reads the stage-1 model, socket, reward and "
      "architecture from the run's params/agent.yaml, so the checkpoint must sit "
      "in its run directory."
    )
  with path.open() as f:
    agent = yaml.load(f, Loader=_AgentYamlLoader)  # A SafeLoader, plus tuples.
  if not isinstance(agent, dict):
    raise ValueError(f"{path} is not a mapping.")
  return agent


def _stage1_task(
  agent: Mapping[str, Any], run_dir: Path
) -> tuple[Model, Socket, Reward, str | None]:
  validation = agent.get("validation")
  if not isinstance(validation, Mapping) or not {"actor", "critic"} <= set(agent):
    raise ValueError(
      f"{run_dir}/params/agent.yaml lacks the actor, critic or validation block: "
      "not an osl_mimic or osl_gait_library run."
    )
  model = validation.get("model")
  socket = validation.get("socket")
  reward = validation.get("reward")
  _check_choice("model", model, get_args(Model))
  _check_choice("socket", socket, get_args(Socket))
  _check_choice("reward", reward, get_args(Reward))
  variant = validation.get("variant")
  return (
    cast(Model, model),
    cast(Socket, socket),
    cast(Reward, reward),
    None if variant is None else str(variant),
  )


def _check_choice(name: str, value: object, choices: Iterable[object]) -> None:
  choices = tuple(choices)
  if value not in choices:
    raise ValueError(f"Unknown {name} {value!r}; expected one of {choices}.")


def _load_checkpoint(path: Path) -> Any:
  """CPU and memory-mapped: only the networks are read, never the optimizer."""
  return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _is_warm_start_file(path: Path) -> bool:
  data = _load_checkpoint(path)
  return isinstance(data, dict) and "format" in data


def _check_stage1_checkpoint(data: Any, path: Path) -> None:
  if not isinstance(data, dict):
    raise ValueError(f"{path} is not a checkpoint dictionary.")
  if "format" in data:
    raise ValueError(
      f"{path} is already a warm-start file ({data['format']}): pass it to stage 2 "
      "(--agent.warm-start.path) as is."
    )
  if "human_actor_state_dict" in data:
    raise ValueError(
      f"{path} is a stage-2 checkpoint: continue it with --agent.resume, not as a "
      "warm start."
    )
  missing = [key for key in _STAGE1_KEYS if key not in data]
  if missing:
    raise ValueError(
      f"{path} is not a stage-1 (MuscleMimicPPO) checkpoint: it lacks {missing}."
    )


def _linear_layers(
  state: Mapping[str, torch.Tensor], what: str, path: Path
) -> tuple[str, str]:
  """The key prefixes of the first and the last linear layer (``mlp.<i>``)."""
  ids = []
  for key, value in state.items():
    match = _LINEAR_WEIGHT.fullmatch(key)
    if match is not None and value.ndim == 2:
      ids.append(int(match.group(1)))
  if not ids:
    raise ValueError(f"{path}: the {what} has no mlp.<i>.weight matrices.")
  return f"mlp.{min(ids)}", f"mlp.{max(ids)}"


def _check_layout(layout: ActionLayout, model: Model) -> None:
  """``ActionLayout.for_model`` against the action terms osl_mimic really builds.

  The order must match exactly. A term listing plain actuator names (no regex) also
  fixes its dimension, which catches a robot run labelled with the wrong model.
  """
  terms = _actions(model)
  names = [name for name, _ in layout.terms]
  if names != list(terms):
    raise ValueError(
      f"osl_mimic's {model!r} action terms are {list(terms)}, but the stage-1 "
      f"layout assumes {names}: update layout.STAGE1_TERM_ORDER."
    )
  for name, dim in layout.terms:
    targets: tuple[str, ...] = tuple(getattr(terms[name], "actuator_names", ()))
    if not targets or not all(_PLAIN_NAME.fullmatch(t) for t in targets):
      continue  # A regular expression (the muscles): counting needs the model.
    if len(targets) != dim:
      raise ValueError(
        f"The checkpoint gives {name!r} {dim} action rows, but the {model!r} "
        f"model's {name!r} term drives {len(targets)} actuators."
      )


def _model_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
  """``MuscleMimicMLPModel`` keyword arguments from an agent.yaml model block,
  stripped as ``MjlabOnPolicyRunner`` strips them."""
  kwargs = dict(cfg)
  class_name = kwargs.pop("class_name", MODEL_CLASS)
  if class_name != MODEL_CLASS:
    raise ValueError(f"Stage-1 model class {class_name!r}; expected {MODEL_CLASS}.")
  if kwargs.pop("cnn_cfg", None) is not None or kwargs.get("rnn_type") is not None:
    raise ValueError("Convolutional and recurrent stage-1 models are not supported.")
  for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
    kwargs.pop(key, None)
  if kwargs.get("distribution_cfg") is None:
    kwargs.pop("distribution_cfg", None)
  return kwargs


def _scalar(value: Any) -> torch.Tensor:
  return torch.as_tensor(value, dtype=torch.float32).detach().clone().reshape(())


def _cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  return {key: value.detach().cpu() for key, value in state.items()}


def _plain(value: Any) -> Any:
  """``value`` as plain types (YAML- and ``weights_only``-safe)."""
  if isinstance(value, Mapping):
    return {str(key): _plain(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_plain(item) for item in value]
  if isinstance(value, torch.Tensor):
    return _plain(value.tolist())
  if isinstance(value, Path):
    return str(value)
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  raise TypeError(f"Not a plain lineage value: {value!r}")


def _library_dir(run_dir: Path) -> str | None:
  """The stage-1 motion library, read from ``params/env.yaml`` as text."""
  try:
    text = (run_dir / "params" / "env.yaml").read_text()
  except OSError:
    return None
  match = re.search(r"^[ \t]*library_dir:[ \t]*(.*?)[ \t]*$", text, re.MULTILINE)
  if match is None or not match.group(1):
    return None
  try:
    value = yaml.safe_load(match.group(1))
  except yaml.YAMLError:
    return None
  return str(value) if value else None


def _as_lists(value: Any) -> Any:
  """``value`` with every tuple a list, as a lineage holds a config's tuples
  (hidden dims, std ranges)."""
  if isinstance(value, Mapping):
    return {key: _as_lists(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_as_lists(item) for item in value]
  return value


def _show(value: Any) -> str:
  return "not set" if value is _UNSET else repr(_as_lists(value))


def _read_text(path: Path) -> str:
  """``path``'s text, or "" when it cannot be read (a run's missing log)."""
  try:
    return path.read_text(errors="replace")
  except OSError:
    return ""


def _wandb_roots(run_dir: Path) -> list[Path]:
  roots: list[Path] = []
  for base in (*run_dir.parents, Path.cwd()):
    candidate = base / "wandb"
    if candidate not in roots and candidate.is_dir():
      roots.append(candidate)
  return roots


def _format_rows(rows: Sequence[int]) -> str:
  """Row spans in slice notation: ``[0, 1, 2, 5]`` reads ``0:3, 5``."""
  spans: list[str] = []
  i = 0
  while i < len(rows):
    j = i
    while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1:
      j += 1
    spans.append(str(rows[i]) if i == j else f"{rows[i]}:{rows[j] + 1}")
    i = j + 1
  return ", ".join(spans)
