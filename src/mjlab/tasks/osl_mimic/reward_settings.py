"""The reward settings of an OSL imitation run, for checks across launches.

What a run's ``mimic`` term trains on beyond musclemimic's kernels is its CLF
decrease condition (``ClfCfg``'s form, rates, ``pen_w`` and ``pen_scale``) and its
regularizers (``RegularizerCfg``), together with the env stages that ramp them.
This module reads those settings from a live env config, from a run's saved
``params/env.yaml`` and from a checkpoint (:data:`CHECKPOINT_KEY`), compares them
(:func:`reward_mismatches`), and turns them back into the command line that rebuilds
them (:func:`reward_flags`). The stage-2 warm start compares stage 1's with its own;
both runners refuse to train on from a checkpoint whose run's reward differs
(:func:`check_resume_reward`), and an evaluation scores a checkpoint under its run's
reward (:func:`checkpoint_reward_record`, :func:`apply_reward_settings`).

A run that records none of them (every run before 2026-11-18) trained with
musclemimic's decrease condition (:data:`LEGACY_DECAY_FORM`), ``pen_scale`` 1 and no
regularizers; one that records no leg-contact penalty (every run before 2026-11-19)
trained without it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

from .mdp.rewards import REGULARIZERS
from .touch_sites import load_checkpoint

CHECKPOINT_KEY = "reward_settings"
"""Every checkpoint's ``infos`` entry: ``{"config": <the env config's settings>,
"env_stages": <the run's env stages>}`` (:func:`reward_record`)."""
RESUME_FLAG = "--agent.allow-reward-mismatch"
CLF_SETTINGS = (
  "pen_w",
  "pen_scale",
  "decay_form",
  "decay_rate",
  "decay_norm_rate",
  "vdot_window",
)
"""The ``clf`` fields the settings record (``clf_<field>``)."""
REGULARIZER_SETTINGS = (
  "scale",
  *REGULARIZERS,
  "joint_limit_joints",
  "leg_contact_threshold",
  "leg_contact_owner",
)
"""The ``regularizers`` fields they record (``reg_<field>``). The leg-contact owner
is recorded for a stage-2 term only (one with reward channels): stage 1 ignores
it."""
REWARD_SETTING_KEYS: tuple[str, ...] = (
  *(f"clf_{name}" for name in CLF_SETTINGS),
  *(f"reg_{name}" for name in REGULARIZER_SETTINGS),
)
"""The keys of :func:`reward_settings`."""
LEGACY_DECAY_FORM = "musclemimic"
"""The decrease condition of a run that records none: every run before
2026-11-18, which also had no regularizers."""
_PARAMS = "--env.rewards.mimic.params"


def reward_settings(clf: Any, regularizers: Any, split: bool = False) -> dict[str, Any]:
  """The reward settings of a ``mimic`` term's ``clf`` and ``regularizers`` params
  (config objects, or their env.yaml mappings), under :data:`REWARD_SETTING_KEYS`;
  ``split`` for a stage-2 term (reward channels), whose leg-contact owner is
  recorded (None otherwise).

  Without ``clf`` (MimicReward) every ``clf_*`` is None. A run that records no
  decrease condition or no regularizers (every run before 2026-11-18) gets what it
  trained with: :data:`LEGACY_DECAY_FORM`, ``pen_scale`` 1 and no regularizers
  (every coefficient 0, ``scale`` 1, no joints); one that records no V-dot window
  (before 2026-11-19) differenced over 1 step; one that records no leg-contact
  penalty (before it existed) had none (coefficient 0, no threshold or owner).
  """

  def get(owner: Any, name: str, default: Any = None) -> Any:
    if isinstance(owner, Mapping):
      return owner.get(name, default)
    return getattr(owner, name, default)

  out: dict[str, Any] = dict.fromkeys(REWARD_SETTING_KEYS)
  if clf is not None:
    out["clf_pen_w"] = _number(get(clf, "pen_w"))
    out["clf_pen_scale"] = _number(get(clf, "pen_scale", 1.0))
    out["clf_decay_form"] = str(get(clf, "decay_form", LEGACY_DECAY_FORM))
    out["clf_decay_rate"] = _number(get(clf, "decay_rate"))
    out["clf_decay_norm_rate"] = _number(get(clf, "decay_norm_rate"))
    window = _number(get(clf, "vdot_window", 1))
    out["clf_vdot_window"] = None if window is None else int(window)
  for name in REGULARIZERS:
    value = get(regularizers, name, 0.0) if regularizers is not None else 0.0
    out[f"reg_{name}"] = _number(value)
  scale = get(regularizers, "scale", 1.0) if regularizers is not None else 1.0
  out["reg_scale"] = _number(scale)
  joints = get(regularizers, "joint_limit_joints", ()) if regularizers else ()
  out["reg_joint_limit_joints"] = [str(j) for j in joints or ()]
  if regularizers is not None:
    threshold = get(regularizers, "leg_contact_threshold", None)
    out["reg_leg_contact_threshold"] = _number(threshold)
    owner = get(regularizers, "leg_contact_owner", None) if split else None
    out["reg_leg_contact_owner"] = None if owner is None else str(owner)
  return out


def env_reward_settings(cfg: Any) -> dict[str, Any] | None:
  """The reward settings of a live env config's ``mimic`` term; None without one
  (another task, a fake env)."""
  rewards = getattr(cfg, "rewards", None)
  mimic = rewards.get("mimic") if isinstance(rewards, Mapping) else None
  if mimic is None:
    return None
  params = getattr(mimic, "params", None) or {}
  return reward_settings(
    params.get("clf"), params.get("regularizers"), split="channels" in params
  )


def saved_reward_settings(env: Any) -> dict[str, Any] | None:
  """The reward settings a saved ``params/env.yaml``, read as plain data, records;
  None without a ``mimic`` term."""
  params = _dig(env, "rewards", "mimic", "params")
  if not isinstance(params, Mapping):
    return None
  return reward_settings(
    params.get("clf"), params.get("regularizers"), split="channels" in params
  )


def reward_stage_setting(path: str) -> str | None:
  """The reward setting an env-stage path sets (``rewards.mimic.params.clf.pen_scale``
  -> ``clf_pen_scale``), or None for any other path."""
  for prefix, key in (
    ("rewards.mimic.params.clf.", "clf_"),
    ("rewards.mimic.params.regularizers.", "reg_"),
  ):
    if path.startswith(prefix):
      name = key + path[len(prefix) :]
      return name if name in REWARD_SETTING_KEYS else None
  return None


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


def staged_reward_settings(
  settings: Mapping[str, Any], stages: Iterable[Sequence[Any]], update: int
) -> dict[str, Any]:
  """``settings`` with the reward stages in force at ``update`` on top."""
  out = dict(settings)
  for path, value in env_stages_in_force(stages, update).items():
    key = reward_stage_setting(path)
    if key is not None:
      out[key] = value
  return out


def _with_legacy(settings: Mapping[str, Any]) -> dict[str, Any]:
  """``settings`` with every reward setting a record lacks (one written before
  2026-11-18) at what that run trained with (:func:`reward_settings`)."""
  out = dict(settings)
  clf = out.get("clf_pen_w") is not None
  legacy = reward_settings({"pen_w": out.get("clf_pen_w")} if clf else None, None)
  for key in REWARD_SETTING_KEYS:
    if key not in out:
      out[key] = legacy[key]
  return out


def reward_mismatches(
  a: Mapping[str, Any], b: Mapping[str, Any], clf_presence: bool = True
) -> list[str]:
  """How two reward settings (:func:`reward_settings`, or a lineage's
  ``stage1_env``) differ, one ``"<setting> <a> vs <b>"`` per difference, on what
  the reward computes: whether each uses the CLF at all (unless ``clf_presence`` is
  False: the warm start's reward switch handles that), the CLF penalty ``pen_w x
  pen_scale`` (when both use the CLF), the decrease condition's form (and the paper
  form's rates) where both penalize it, each regularizer times ``scale``, the
  joints where both penalize joint limits, and where both penalize leg contacts its
  threshold and (when both record one: stage 2) its owner. A record without these
  settings (from before 2026-11-18, or before the leg-contact penalty) counts as
  what it trained with. Numbers print exactly, as :func:`reward_flags` writes
  them."""
  a, b = _with_legacy(a), _with_legacy(b)
  lines: list[str] = []
  if clf_presence and (a["clf_pen_w"] is None) != (b["clf_pen_w"] is None):
    lines.append(f"reward {_reward_kind(a)} vs {_reward_kind(b)}")

  def differs(x: Any, y: Any) -> bool:
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
      return not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)
    return x != y

  def scale(value: Any) -> float:
    return 1.0 if value is None else float(value)

  if a["clf_pen_w"] is not None and b["clf_pen_w"] is not None:
    pen = [s["clf_pen_w"] * scale(s["clf_pen_scale"]) for s in (a, b)]
    if differs(*pen):
      lines.append(
        f"CLF penalty pen_w x pen_scale {_exact(pen[0])} vs {_exact(pen[1])}"
      )
    if pen[0] > 0.0 and pen[1] > 0.0:
      forms = (a["clf_decay_form"], b["clf_decay_form"])
      if forms[0] != forms[1]:
        lines.append(f"CLF decay form {forms[0]!r} vs {forms[1]!r}")
      elif forms[0] == "paper":
        for name in ("decay_rate", "decay_norm_rate", "vdot_window"):
          x, y = a[f"clf_{name}"], b[f"clf_{name}"]
          if differs(x, y):
            lines.append(f"CLF {name} {_exact(x, name)} vs {_exact(y, name)}")
  effective = {}
  for name in REGULARIZERS:
    pair = [(s[f"reg_{name}"] or 0.0) * scale(s["reg_scale"]) for s in (a, b)]
    effective[name] = pair
    if differs(*pair):
      lines.append(
        f"regularizer {name} (x scale) {_exact(pair[0])} vs {_exact(pair[1])}"
      )
  if min(effective["joint_limits"]) > 0.0:
    joints = [sorted(s["reg_joint_limit_joints"] or ()) for s in (a, b)]
    if joints[0] != joints[1]:
      lines.append(f"joint_limit_joints {joints[0]} vs {joints[1]}")
  if min(effective["leg_contact"]) > 0.0:
    threshold = [s["reg_leg_contact_threshold"] for s in (a, b)]
    if differs(*threshold):
      lines.append(
        f"leg_contact_threshold {_exact(threshold[0])} vs {_exact(threshold[1])}"
      )
    owner = [s["reg_leg_contact_owner"] for s in (a, b)]
    if None not in owner and owner[0] != owner[1]:
      lines.append(f"leg_contact_owner {owner[0]!r} vs {owner[1]!r}")
  return lines


def describe_reward_settings(settings: Mapping[str, Any]) -> str:
  """A one-line summary of reward settings, for provenance printouts."""
  s = _with_legacy(settings)
  if s["clf_pen_w"] is None:
    text = "MimicReward (no CLF)"
  else:
    form = s["clf_decay_form"]
    rates = (
      f" (rate {s['clf_decay_rate']:g}/s, normalizer {s['clf_decay_norm_rate']:g}/s,"
      f" V-dot over {s['clf_vdot_window'] or 1} steps)"
      if form == "paper" and s["clf_decay_rate"] is not None
      else ""
    )
    text = (
      f"CLF decay form {form}{rates}, pen_w {s['clf_pen_w']:g} x pen_scale "
      f"{_shown(s['clf_pen_scale'])}"
    )
  details = {"joint_limits": f"{len(s['reg_joint_limit_joints'] or ())} joints"}
  owner = s["reg_leg_contact_owner"]
  details["leg_contact"] = f"over {_shown(s['reg_leg_contact_threshold'])} N" + (
    f", {owner} channel{'s' if owner == 'both' else ''}" if owner else ""
  )
  active = [
    f"{name} {s[f'reg_{name}']:g}" + (f" ({details[name]})" if name in details else "")
    for name in REGULARIZERS
    if s[f"reg_{name}"]
  ]
  if not active:
    return f"{text}; no regularizers"
  return f"{text}; regularizers x {_shown(s['reg_scale'])}: {', '.join(active)}"


def reward_flags(
  settings: Mapping[str, Any], env_stages: Sequence[Sequence[Any]] | None = None
) -> list[str]:
  """The command line that gives a run these reward settings (and, given, these env
  stages): every setting, so a resume rebuilds exactly the run's reward. Numbers
  are written exactly (``repr``), so they parse back to the recorded values. A
  setting with no CLF (MimicReward) writes no CLF flag: the task decides that."""
  s = _with_legacy(settings)
  flags: list[str] = []
  if s["clf_pen_w"] is not None:
    flags += [
      f"{_PARAMS}.clf.decay-form {s['clf_decay_form']}",
      f"{_PARAMS}.clf.pen-w {_exact(s['clf_pen_w'])}",
      f"{_PARAMS}.clf.pen-scale {_exact(_or(s['clf_pen_scale'], 1.0))}",
    ]
    if s["clf_decay_form"] == "paper":
      for name in ("decay_rate", "decay_norm_rate", "vdot_window"):
        if s[f"clf_{name}"] is not None:
          value = _exact(s[f"clf_{name}"], name)
          flags.append(f"{_PARAMS}.clf.{_flag(name)} {value}")
  flags.append(f"{_PARAMS}.regularizers.scale {_exact(_or(s['reg_scale'], 1.0))}")
  for name in REGULARIZERS:
    value = _exact(s[f"reg_{name}"] or 0.0)
    flags.append(f"{_PARAMS}.regularizers.{_flag(name)} {value}")
  joints = s["reg_joint_limit_joints"]
  if s["reg_joint_limits"] and joints:
    flags.append(f'{_PARAMS}.regularizers.joint-limit-joints "{tuple(joints)!r}"')
  if s["reg_leg_contact_threshold"] is not None:
    threshold = _exact(s["reg_leg_contact_threshold"])
    flags.append(f"{_PARAMS}.regularizers.leg-contact-threshold {threshold}")
  if s["reg_leg_contact_owner"] is not None:
    owner = s["reg_leg_contact_owner"]
    flags.append(f"{_PARAMS}.regularizers.leg-contact-owner {owner}")
  if env_stages is not None:
    stages = tuple((int(u), str(p), float(v)) for u, p, v in env_stages)
    flags.append(f'--agent.env-stages "{stages!r}"')
  return flags


def reward_record(cfg: Any, env_stages: Iterable[Sequence[Any]]) -> dict | None:
  """A checkpoint's :data:`CHECKPOINT_KEY` entry: the env config's (unstaged) reward
  settings and the run's env stages; None for an env without a ``mimic`` term."""
  config = env_reward_settings(cfg)
  if config is None:
    return None
  stages = [[int(u), str(p), float(v)] for u, p, v in env_stages]
  return {"config": config, "env_stages": stages}


def run_reward_record(checkpoint: str | Path) -> dict | None:
  """A checkpoint's reward record from its run's ``params``, for checkpoints that
  carry none (written before 2026-11-18): ``env.yaml``'s settings, with the legacy
  defaults for what it lacks, and ``agent.yaml``'s env stages (none in stage 2,
  whose runs had none then). None without a readable ``env.yaml``."""
  params = Path(checkpoint).parent / "params"
  config = saved_reward_settings(read_params_yaml(params / "env.yaml"))
  if config is None:
    return None
  agent = read_params_yaml(params / "agent.yaml")
  stages = agent.get("env_stages") if isinstance(agent, Mapping) else None
  return {"config": config, "env_stages": [list(s) for s in stages or ()]}


def checkpoint_reward_record(
  checkpoint: str | Path, infos: Mapping[str, Any] | None = None
) -> tuple[dict | None, str]:
  """A checkpoint's reward record (:func:`reward_record`) and where it comes from:
  its own (``infos[CHECKPOINT_KEY]``; ``infos`` is read from the checkpoint when
  not given), else its run's ``params/`` (:func:`run_reward_record`) for one written
  before 2026-11-18; ``(None, "")`` when neither is there."""
  if infos is None:
    infos = load_checkpoint(checkpoint).get("infos")
  record = (infos or {}).get(CHECKPOINT_KEY)
  if record is not None:
    return dict(record), "its record"
  record = run_reward_record(checkpoint)
  if record is not None:
    return record, "its run's params/ (it predates the record)"
  return None, ""


def apply_reward_settings(cfg: Any, settings: Mapping[str, Any]) -> None:
  """Give a live env config's ``mimic`` term these reward settings (a record's
  ``config``), the inverse of :func:`env_reward_settings` on what the reward
  computes; one a record lacks counts as what that run trained with (before
  2026-11-18: musclemimic's decrease condition, no regularizers). The term gets new
  config objects, never edited shared ones; ``log_servo_terms``, a diagnostic, stays
  the config's.

  Raises:
    ValueError: For a config without a ``mimic`` term, or when one of the two uses
      the CLF and the other not.
  """
  s = _with_legacy(settings)
  rewards = getattr(cfg, "rewards", None)
  if not isinstance(rewards, Mapping) or rewards.get("mimic") is None:
    raise ValueError("The env config has no 'mimic' reward term.")
  mimic = rewards["mimic"]
  params = dict(mimic.params)
  clf = params.get("clf")
  if (clf is None) != (s["clf_pen_w"] is None):
    raise ValueError(
      f"Cannot give a {'MimicReward' if clf is None else 'CLF'} task the reward "
      f"settings of a {_reward_kind(s)} run."
    )
  if clf is not None:
    changes: dict[str, Any] = {
      "pen_w": float(s["clf_pen_w"]),
      "pen_scale": _or(s["clf_pen_scale"], 1.0),
      "decay_form": str(s["clf_decay_form"]),
      "vdot_window": int(s["clf_vdot_window"] or 1),
    }
    for name in ("decay_rate", "decay_norm_rate"):
      if s[f"clf_{name}"] is not None:
        changes[name] = float(s[f"clf_{name}"])
    params["clf"] = replace(clf, **changes)
  regs = params.get("regularizers")
  if regs is not None:
    changes = {name: float(s[f"reg_{name}"] or 0.0) for name in REGULARIZERS}
    changes["scale"] = _or(s["reg_scale"], 1.0)
    if s["reg_joint_limit_joints"]:
      changes["joint_limit_joints"] = tuple(s["reg_joint_limit_joints"])
    if s["reg_leg_contact_threshold"] is not None:
      changes["leg_contact_threshold"] = float(s["reg_leg_contact_threshold"])
    if s["reg_leg_contact_owner"] is not None:
      changes["leg_contact_owner"] = str(s["reg_leg_contact_owner"])
    params["regularizers"] = replace(regs, **changes)
  elif any(s[f"reg_{name}"] for name in REGULARIZERS):
    raise ValueError("The env config's 'mimic' term has no regularizers to set.")
  cfg.rewards = {**rewards, "mimic": replace(mimic, params=params)}


def resume_reward_mismatches(
  saved: Mapping[str, Any],
  live: Mapping[str, Any],
  live_stages: Iterable[Sequence[Any]],
  update: int,
  what: str = "update",
) -> list[str]:
  """How a resumed run's reward would differ from its run's (a
  :func:`reward_record`): once every env stage has ended (the env configs), and on
  every update the resumed run trains from ``update`` on, where both runs' env
  stages apply (stage 1 counts PPO updates, stage 2 its stage-local iteration).
  Staged values hold from one stage end to the next, so they are compared at
  ``update`` and at every later end of either run's reward stages. Each difference
  is reported once, where it first shows."""
  saved_config = saved["config"]
  saved_stages = list(saved.get("env_stages") or ())
  live_stages = list(live_stages)
  lines = reward_mismatches(saved_config, live)
  shown = set(lines)
  ends = {
    int(u)
    for u, path, _ in (*saved_stages, *live_stages)
    if int(u) > update and reward_stage_setting(str(path)) is not None
  }
  for point in sorted({int(update)} | ends):
    now = reward_mismatches(
      staged_reward_settings(saved_config, saved_stages, point),
      staged_reward_settings(live, live_stages, point),
      clf_presence=False,
    )
    for line in now:
      if line not in shown:
        shown.add(line)
        lines.append(f"at {what} {point}, with the env stages: {line}")
  return lines


def check_resume_reward(
  path: str | Path,
  infos: Mapping[str, Any] | None,
  cfg: Any,
  env_stages: Iterable[Sequence[Any]],
  update: int,
  allowed: bool,
  what: str = "update",
) -> None:
  """Refuse to train on from checkpoint ``path`` with another reward than its run
  trained, unless ``allowed`` (:data:`RESUME_FLAG`), which only warns. The runners
  call it when a resumed run starts to train, never on a load alone (evaluations
  load whole checkpoints too).

  The run's reward is :func:`checkpoint_reward_record`'s; the resumed run's is
  ``cfg``'s with its ``env_stages``, from ``update`` on
  (:func:`resume_reward_mismatches`). An env without a ``mimic`` term is not
  checked. A checkpoint from before the record whose run's ``params/`` is not
  beside it (a W&B download, say) gives nothing to compare: it is refused unless
  allowed, since its reward cannot be read.

  Raises:
    ValueError: On a difference, naming the flags that rebuild the run's reward (and
      the task to resume it under, when only one side uses the CLF) and
      :data:`RESUME_FLAG`; or when the run's reward cannot be read, naming the ways
      on.
  """
  live = env_reward_settings(cfg)
  if live is None:
    return
  record, source = checkpoint_reward_record(path, infos or {})
  if record is None:
    _refuse_unreadable_reward(path, allowed)
    return
  lines = resume_reward_mismatches(record, live, env_stages, update, what)
  if not lines:
    return
  text = (
    f"{path} was trained on another reward than this config's ({source}; run vs "
    f"this config): {'; '.join(lines)}."
  )
  if not allowed:
    raise ValueError(
      f"{text} A resume would train on another reward. {_rebuild(record, live)}; "
      f"or pass {RESUME_FLAG} True to train on with this config's reward, which the "
      "checkpoint's critic and return statistics do not value."
    )
  print(
    f"[WARN] {text} Allowed ({RESUME_FLAG}): training on with this config's reward, "
    "which the checkpoint's critic and return statistics do not value.",
    flush=True,
  )


def _rebuild(record: Mapping[str, Any], live: Mapping[str, Any]) -> str:
  """How to resume a run on its own reward: the flags, and the task when only one
  of the run and this config uses the CLF (a MimicReward run under a -CLF task, or
  the reverse), which no flag switches."""
  config = _with_legacy(record["config"])
  flags = " ".join(reward_flags(config, record.get("env_stages") or ()))
  run_clf = config["clf_pen_w"] is not None
  if run_clf == (live["clf_pen_w"] is not None):
    return f"Resume with the run's reward: {flags}"
  if run_clf:
    return (
      "The run trained with the CLF: resume it under its CLF task (the -CLF variant "
      f"of this task), with: {flags}"
    )
  return (
    "The run trained MimicReward (no CLF): resume it under its MimicReward task "
    f"(this task without -CLF), with: {flags}"
  )


def _refuse_unreadable_reward(path: str | Path, allowed: bool) -> None:
  params = Path(path).parent / "params"
  text = (
    f"{path} records no reward settings (it predates them) and its run's params/ "
    f"is not beside it ({params}), so the reward its run trained cannot be read."
  )
  if not allowed:
    raise ValueError(
      f"{text} Copy the run's params/ folder (env.yaml and agent.yaml) to "
      f"{params}, or pass {RESUME_FLAG} True to train on regardless, adding the "
      "run's old-reward flags to keep its reward (before 2026-11-18 a CLF run "
      f"trained {_PARAMS}.clf.decay-form musclemimic at its own pen-w, and no run "
      f"had regularizers: {_PARAMS}.regularizers.<name> 0 each; with the run's "
      "--agent.env-stages)."
    )
  print(
    f"[WARN] {text} Allowed ({RESUME_FLAG}): training on with this config's reward, "
    "unchecked.",
    flush=True,
  )


def read_params_yaml(path: Path) -> Any:
  """A saved ``params/*.yaml`` as plain data (Python tags read as their data, never
  constructed); None when missing or unreadable."""

  class Loader(yaml.SafeLoader):
    pass

  def python_data(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
      return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
      return loader.construct_sequence(node, deep=True)
    value = loader.construct_scalar(cast(yaml.ScalarNode, node))
    return suffix.removeprefix("name:") if suffix.startswith("name:") else value

  Loader.add_multi_constructor("tag:yaml.org,2002:python/", python_data)
  try:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
  except (OSError, yaml.YAMLError):
    return None


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


def _or(value: Any, default: float) -> float:
  return default if value is None else float(value)


def _exact(value: Any, name: str = "") -> str:
  """A setting as the command line takes it back unchanged: an int for
  ``vdot_window``, ``repr`` of the float otherwise (``:g`` keeps 6 digits)."""
  if value is None:
    return "not recorded"
  if name == "vdot_window":
    return str(int(value))
  return repr(float(value))


def _reward_kind(settings: Mapping[str, Any]) -> str:
  if settings["clf_pen_w"] is None:
    return "MimicReward (no CLF)"
  return f"CLF ({settings['clf_decay_form']} decrease condition)"


def _flag(name: str) -> str:
  return name.replace("_", "-")
