"""Canonical digests of the OSL tasks' configs, for bit-identity tests across commits.

``tests/test_osl_robot_recipe.py`` checks that every OSL task's training env, play
env and runner config equals the one of the fixture's source commit (its ``source``:
the CLF-RL reward commit's configs, then the OSL/left-leg contact sensor and penalty
fields, then the robot tasks' smaller networks), except the robot unicycle configs
the recipe sets: every musculoskeletal (MSK) task stays bit-identical.
:data:`FIXTURE` holds that commit's digests, one per top-level config field, so a
failure names the field. Regenerate it from the base commit's code, never from a
branch under test (a change that alters configs on purpose regenerates it from its
own tree once its field-level diff is checked)::

  PYTHONPATH=<base commit checkout>/src python tests/osl_cfg_snapshot.py <commit>

A config is canonicalized before hashing: dataclasses by class name and fields,
functions and classes by qualified name (closures with their captured values,
partials with their arguments), never an object address. ``library_dir`` and
``motion_file`` are blanked: their defaults come from the shell's ``MJLAB_OSL_*``
variables.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).parent / "fixtures" / "osl_cfg_digests.json"
BLANKED = frozenset({"library_dir", "motion_file"})
KINDS = ("env", "play_env", "rl")


def canon(x: Any, depth: int = 0) -> Any:
  """``x`` as plain JSON data that is the same in every process."""
  if depth > 60:
    raise ValueError("Config nested too deep to canonicalize.")
  if x is None or isinstance(x, (bool, int, str)):
    return x
  if isinstance(x, float):
    return x if math.isfinite(x) else repr(x)
  if isinstance(x, enum.Enum):
    return f"<enum {type(x).__qualname__}.{x.name}>"
  if isinstance(x, Path):
    return str(x)
  if isinstance(x, slice):
    return {"__slice__": [canon(x.start), canon(x.stop), canon(x.step)]}
  if isinstance(x, (list, tuple)):
    return [canon(v, depth + 1) for v in x]
  if isinstance(x, (set, frozenset)):
    return sorted((canon(v, depth + 1) for v in x), key=repr)
  if isinstance(x, dict):
    return {
      str(k): "<blanked>" if str(k) in BLANKED else canon(v, depth + 1)
      for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
    }
  if dataclasses.is_dataclass(x) and not isinstance(x, type):
    out: dict[str, Any] = {"__class__": _name(type(x))}
    for f in dataclasses.fields(x):
      value = getattr(x, f.name)
      out[f.name] = "<blanked>" if f.name in BLANKED else canon(value, depth + 1)
    return out
  if isinstance(x, functools.partial):
    return {
      "__partial__": canon(x.func, depth + 1),
      "args": canon(list(x.args), depth + 1),
      "keywords": canon(dict(x.keywords), depth + 1),
    }
  if isinstance(x, (types.FunctionType, types.MethodType)):
    func = x.__func__ if isinstance(x, types.MethodType) else x
    out = {"__function__": _name(x)}
    if func.__closure__:
      out["closure"] = {
        name: canon(_cell(cell), depth + 1)
        for name, cell in zip(func.__code__.co_freevars, func.__closure__, strict=True)
      }
    return out
  if isinstance(x, type):
    return {"__type__": _name(x)}
  if callable(x) and hasattr(x, "__qualname__"):
    return {"__callable__": _name(x)}
  if hasattr(x, "__dict__"):
    return {"__object__": _name(type(x)), "state": canon(dict(vars(x)), depth + 1)}
  raise TypeError(f"Cannot canonicalize {type(x).__qualname__}: {x!r}")


def field_digests(cfg: Any) -> dict[str, str]:
  """One digest per top-level field of a config dataclass."""
  data = canon(cfg)
  return {
    key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]
    for key, value in data.items()
  }


def task_digests(task: str) -> dict[str, dict[str, str]]:
  """:func:`field_digests` of a registered task's training env, play env and runner
  configs."""
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  return {
    "env": field_digests(load_env_cfg(task)),
    "play_env": field_digests(load_env_cfg(task, play=True)),
    "rl": field_digests(load_rl_cfg(task)),
  }


def osl_tasks() -> list[str]:
  """Every registered OSL task."""
  import mjlab.tasks  # noqa: F401  (registers the tasks)
  from mjlab.tasks.registry import list_tasks

  return sorted(t for t in list_tasks() if "OSL" in t)


def _cell(cell: Any) -> Any:
  try:
    return cell.cell_contents
  except ValueError:
    return "<empty cell>"


def _name(obj: Any) -> str:
  return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', '?')}"


def dumps(data: dict[str, Any]) -> str:
  """The fixture's text: one line per task and config, for readable diffs."""
  lines = ["{", f' "source": {json.dumps(data["source"])},', ' "tasks": {']
  tasks = sorted(data["tasks"])
  for i, task in enumerate(tasks):
    lines.append(f"  {json.dumps(task)}: {{")
    for j, kind in enumerate(KINDS):
      digests = json.dumps(data["tasks"][task][kind], sort_keys=True)
      lines.append(f"   {json.dumps(kind)}: {digests}" + ("," if j < 2 else ""))
    lines.append("  }" + ("," if i < len(tasks) - 1 else ""))
  return "\n".join([*lines, " }", "}", ""])


def main() -> None:
  import mjlab

  tasks = osl_tasks()
  data = {
    "source": sys.argv[1] if len(sys.argv) > 1 else "unrecorded",
    "tasks": {task: task_digests(task) for task in tasks},
  }
  FIXTURE.write_text(dumps(data))
  print(f"{len(tasks)} OSL tasks from {mjlab.__file__} -> {FIXTURE}", file=sys.stderr)


if __name__ == "__main__":
  main()
