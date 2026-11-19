"""The reward settings of an OSL imitation run, for checks across launches.

What a run's ``mimic`` term trains on beyond musclemimic's kernels is its CLF
decrease condition (``ClfCfg``'s form, rates, ``pen_w`` and ``pen_scale``) and its
regularizers (``RegularizerCfg``), together with the env stages that ramp them.
This module reads those settings from a live env config, from a run's saved
``params/env.yaml`` and from a checkpoint (:data:`CHECKPOINT_KEY`), compares them
(:func:`reward_mismatches`), and turns them back into the command line that rebuilds
them (:func:`reward_flags`). The stage-2 warm start compares stage 1's with its own;
both runners refuse a training resume whose reward would differ from its run's
(:func:`check_resume_reward`).

A run that records none of them (every run before 2026-11-18) trained with
musclemimic's decrease condition (:data:`LEGACY_DECAY_FORM`), ``pen_scale`` 1 and no
regularizers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from .mdp.rewards import REGULARIZERS

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
REGULARIZER_SETTINGS = ("scale", *REGULARIZERS, "joint_limit_joints")
"""The ``regularizers`` fields they record (``reg_<field>``)."""
REWARD_SETTING_KEYS: tuple[str, ...] = (
  *(f"clf_{name}" for name in CLF_SETTINGS),
  *(f"reg_{name}" for name in REGULARIZER_SETTINGS),
)
"""The keys of :func:`reward_settings`."""
LEGACY_DECAY_FORM = "musclemimic"
"""The decrease condition of a run that records none: every run before
2026-11-18, which also had no regularizers."""
_PARAMS = "--env.rewards.mimic.params"


def reward_settings(clf: Any, regularizers: Any) -> dict[str, Any]:
  """The reward settings of a ``mimic`` term's ``clf`` and ``regularizers`` params
  (config objects, or their env.yaml mappings), under :data:`REWARD_SETTING_KEYS`.

  Without ``clf`` (MimicReward) every ``clf_*`` is None. A run that records no
  decrease condition or no regularizers (every run before 2026-11-18) gets what it
  trained with: :data:`LEGACY_DECAY_FORM`, ``pen_scale`` 1 and no regularizers
  (every coefficient 0, ``scale`` 1, no joints); one that records no V-dot window
  (before 2026-11-19) differenced over 1 step.
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
  return out


def env_reward_settings(cfg: Any) -> dict[str, Any] | None:
  """The reward settings of a live env config's ``mimic`` term; None without one
  (another task, a fake env)."""
  rewards = getattr(cfg, "rewards", None)
  mimic = rewards.get("mimic") if isinstance(rewards, Mapping) else None
  if mimic is None:
    return None
  params = getattr(mimic, "params", None) or {}
  return reward_settings(params.get("clf"), params.get("regularizers"))


def saved_reward_settings(env: Any) -> dict[str, Any] | None:
  """The reward settings a saved ``params/env.yaml``, read as plain data, records;
  None without a ``mimic`` term."""
  params = _dig(env, "rewards", "mimic", "params")
  if not isinstance(params, Mapping):
    return None
  return reward_settings(params.get("clf"), params.get("regularizers"))


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


def reward_mismatches(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
  """How two reward settings (:func:`reward_settings`, or a lineage's
  ``stage1_env``) differ, one ``"<setting> <a> vs <b>"`` per difference, on what
  the reward computes: the CLF penalty ``pen_w x pen_scale`` (when both use the CLF),
  the decrease condition's form (and the paper form's rates) where both penalize
  it, each regularizer times ``scale``, and the joints where both penalize joint
  limits. A record without these settings (from before 2026-11-18) counts as what
  it trained with."""
  a, b = _with_legacy(a), _with_legacy(b)
  lines: list[str] = []

  def differs(x: Any, y: Any) -> bool:
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
      return not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)
    return x != y

  def scale(value: Any) -> float:
    return 1.0 if value is None else float(value)

  if a["clf_pen_w"] is not None and b["clf_pen_w"] is not None:
    pen = [s["clf_pen_w"] * scale(s["clf_pen_scale"]) for s in (a, b)]
    if differs(*pen):
      lines.append(f"CLF penalty pen_w x pen_scale {pen[0]:g} vs {pen[1]:g}")
    if pen[0] > 0.0 and pen[1] > 0.0:
      forms = (a["clf_decay_form"], b["clf_decay_form"])
      if forms[0] != forms[1]:
        lines.append(f"CLF decay form {forms[0]!r} vs {forms[1]!r}")
      elif forms[0] == "paper":
        for name in ("decay_rate", "decay_norm_rate", "vdot_window"):
          x, y = a[f"clf_{name}"], b[f"clf_{name}"]
          if differs(x, y):
            lines.append(f"CLF {name} {x} vs {y}")
  effective = {}
  for name in REGULARIZERS:
    pair = [(s[f"reg_{name}"] or 0.0) * scale(s["reg_scale"]) for s in (a, b)]
    effective[name] = pair
    if differs(*pair):
      lines.append(f"regularizer {name} (x scale) {pair[0]:g} vs {pair[1]:g}")
  if min(effective["joint_limits"]) > 0.0:
    joints = [sorted(s["reg_joint_limit_joints"] or ()) for s in (a, b)]
    if joints[0] != joints[1]:
      lines.append(f"joint_limit_joints {joints[0]} vs {joints[1]}")
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
  active = [f"{name} {s[f'reg_{name}']:g}" for name in REGULARIZERS if s[f"reg_{name}"]]
  if not active:
    return f"{text}; no regularizers"
  joints = len(s["reg_joint_limit_joints"] or ())
  return f"{text}; regularizers x {_shown(s['reg_scale'])}: {', '.join(active)}" + (
    f" ({joints} joints)" if s["reg_joint_limits"] else ""
  )


def reward_flags(
  settings: Mapping[str, Any], env_stages: Sequence[Sequence[Any]] | None = None
) -> list[str]:
  """The command line that gives a run these reward settings (and, given, these env
  stages): every setting, so a resume rebuilds exactly the run's reward."""
  s = _with_legacy(settings)
  flags: list[str] = []
  if s["clf_pen_w"] is not None:
    flags += [
      f"{_PARAMS}.clf.decay-form {s['clf_decay_form']}",
      f"{_PARAMS}.clf.pen-w {s['clf_pen_w']:g}",
      f"{_PARAMS}.clf.pen-scale {_shown_g(s['clf_pen_scale'], 1.0)}",
    ]
    if s["clf_decay_form"] == "paper":
      for name in ("decay_rate", "decay_norm_rate", "vdot_window"):
        if s[f"clf_{name}"] is not None:
          flags.append(f"{_PARAMS}.clf.{_flag(name)} {s[f'clf_{name}']:g}")
  flags.append(f"{_PARAMS}.regularizers.scale {_shown_g(s['reg_scale'], 1.0)}")
  for name in REGULARIZERS:
    flags.append(f"{_PARAMS}.regularizers.{_flag(name)} {s[f'reg_{name}'] or 0.0:g}")
  joints = s["reg_joint_limit_joints"]
  if s["reg_joint_limits"] and joints:
    flags.append(f'{_PARAMS}.regularizers.joint-limit-joints "{tuple(joints)!r}"')
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
  config = saved_reward_settings(_read_yaml(params / "env.yaml"))
  if config is None:
    return None
  agent = _read_yaml(params / "agent.yaml")
  stages = agent.get("env_stages") if isinstance(agent, Mapping) else None
  return {"config": config, "env_stages": [list(s) for s in stages or ()]}


def resume_reward_mismatches(
  saved: Mapping[str, Any],
  live: Mapping[str, Any],
  live_stages: Iterable[Sequence[Any]],
  update: int,
  what: str = "update",
) -> list[str]:
  """How a resumed run's reward would differ from its run's (a
  :func:`reward_record`): once every env stage has ended (the env configs), and at
  the ``update`` the resumed run trains first, where both runs' env stages in force
  there apply (stage 1 counts PPO updates, stage 2 its stage-local iteration)."""
  saved_config = saved["config"]
  saved_stages = list(saved.get("env_stages") or ())
  live_stages = list(live_stages)
  lines = reward_mismatches(saved_config, live)
  now = reward_mismatches(
    staged_reward_settings(saved_config, saved_stages, update),
    staged_reward_settings(live, live_stages, update),
  )
  lines += [f"at {what} {update}, with the env stages: {line}" for line in now]
  return list(dict.fromkeys(lines))


def check_resume_reward(
  path: str | Path,
  infos: Mapping[str, Any] | None,
  cfg: Any,
  env_stages: Iterable[Sequence[Any]],
  update: int,
  allowed: bool,
  what: str = "update",
) -> None:
  """Refuse a training resume of checkpoint ``path`` that would train another reward
  than its run did, unless ``allowed`` (:data:`RESUME_FLAG`), which only warns.

  The run's reward is the checkpoint's :data:`CHECKPOINT_KEY` record, else its run's
  ``params`` (:func:`run_reward_record`); the resumed run's is ``cfg``'s with its
  ``env_stages``, from ``update`` on. An env without a ``mimic`` term is not checked.

  Raises:
    ValueError: On a difference, or when the run's reward cannot be read, naming
      the flags that rebuild the run's reward and :data:`RESUME_FLAG`.
  """
  live = env_reward_settings(cfg)
  if live is None:
    return
  record = (infos or {}).get(CHECKPOINT_KEY)
  source = "its record"
  if record is None:
    record = run_reward_record(path)
    source = "its run's params (it predates the record)"
  if record is None:
    lines = [
      "the checkpoint records no reward settings and its run's params/env.yaml is "
      "not beside it"
    ]
    rebuild = (
      "Resume with the run's own reward flags (before 2026-11-18: "
      f"{_PARAMS}.clf.decay-form musclemimic, the run's pen-w, every regularizer 0 "
      "and its env stages)"
    )
  else:
    lines = resume_reward_mismatches(record, live, env_stages, update, what)
    flags = reward_flags(record["config"], record.get("env_stages") or ())
    rebuild = "Resume with the run's reward: " + " ".join(flags)
  if not lines:
    return
  text = (
    f"{path} was trained on another reward than this config's ({source}; run vs "
    f"this config): {'; '.join(lines)}."
  )
  if not allowed:
    raise ValueError(
      f"{text} A resume would train on another reward. {rebuild}; or pass "
      f"{RESUME_FLAG} True to train on with this config's reward, which the "
      "checkpoint's critic and return statistics do not value."
    )
  print(
    f"[WARN] {text} Allowed ({RESUME_FLAG}): trained on, the checkpoint's critic "
    "and return statistics value another reward.",
    flush=True,
  )


def _read_yaml(path: Path) -> Any:
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


def _shown_g(value: Any, default: float) -> str:
  return f"{default if value is None else value:g}"


def _flag(name: str) -> str:
  return name.replace("_", "-")
