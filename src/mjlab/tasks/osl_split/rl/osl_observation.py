"""Which OSL sensor set a stage-2 checkpoint was trained with, and the checks that
make an env feed it the inputs it learned on.

A run's ``osl`` group (its sensor switches, history, lags and noise) is recorded in
two places: the run's ``params/env.yaml`` (``observations.osl``), and, since
2026-11-14, every checkpoint's top-level :data:`~..mdp.OSL_OBSERVATION_KEY`
(:func:`~..mdp.osl_observation_entry`; W&B play downloads the ``.pt`` alone).
:func:`run_osl_observation` resolves them in that order: the checkpoint's key
(which must agree with the run's env.yaml, as the touch sites must), else the run's
env.yaml, else the legacy set (both load cells, no contacts) with a warning, since
every run before the contact sensors trained with it, unless a switch was flipped
on its command line (s2_D's ankle cell off, s2_B's and s2_E's 50 frames: their
width then fails loudly at load).

Play, export and resume rebuild or check against the resolved record:
:meth:`~.runner.SplitOnPolicyRunner.configure_env_for_checkpoint` builds it,
:func:`check_osl_observation` refuses a checkpoint whose input layout (terms,
channels, history) differs from the env's, naming the exact flags that build it.
Two sets of the same width can differ (socket cell vs ankle cell), so the width
alone, which the network's strict load checks, is not enough.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from mjlab.tasks.osl_gait_library.export import read_config
from mjlab.tasks.osl_mimic.touch_sites import load_checkpoint
from mjlab.tasks.osl_split.mdp.observations import (
  AXIAL_SWITCHES,
  OSL_OBS_GROUP,
  OSL_OBSERVATION_FORMAT,
  OSL_OBSERVATION_KEY,
  SENSOR_SWITCHES,
  TERM_SETTINGS,
  describe_osl_sensors,
  legacy_osl_record,
  osl_group_record,
  osl_obs_layout,
  osl_sensor_flags,
)

LEGACY_SOURCE = (
  "recorded nowhere (no checkpoint key, no params/env.yaml): read as the legacy "
  "set, both load cells and no contacts, which every run before 2026-11-14 trained "
  "with unless a switch was flipped on its command line"
)
_LAYOUT_PARAMS = frozenset(
  {
    "entity_name",
    "randomize",
    "projected_gravity",
    "applied_torque",
    *(switch for switches in SENSOR_SWITCHES.values() for switch in switches),
    *AXIAL_SWITCHES.values(),
  }
)
"""Params that set the layout or the mode, not a lag, error bound or noise (a
record without the axial switches reads six-axis cells, which the layout check
compares)."""


class ResolvedOslObservation(NamedTuple):
  """A checkpoint's ``osl`` group record and where it was found."""

  record: dict[str, Any]
  source: str
  recorded: bool
  """False for the legacy fallback."""


def checkpoint_osl_observation(payload: Mapping[str, Any]) -> dict[str, Any] | None:
  """The ``osl`` group record a checkpoint holds; None without the key (saved
  before 2026-11-14).

  Raises:
    ValueError: For a key of another format.
  """
  entry = payload.get(OSL_OBSERVATION_KEY)
  if entry is None:
    return None
  if not isinstance(entry, Mapping) or entry.get("format") != OSL_OBSERVATION_FORMAT:
    found = entry.get("format") if isinstance(entry, Mapping) else type(entry)
    raise ValueError(
      f"The checkpoint's {OSL_OBSERVATION_KEY!r} has format {found!r}, not "
      f"{OSL_OBSERVATION_FORMAT!r}."
    )
  return dict(entry["group"])


def saved_env_osl_observation(saved: Mapping[str, Any]) -> dict[str, Any] | None:
  """The ``osl`` group a run's ``params/env.yaml``, read as plain data, records;
  None without one."""
  group = (saved.get("observations") or {}).get(OSL_OBS_GROUP)
  return None if group is None else dict(group)


def run_osl_observation(
  checkpoint: str | Path,
  payload: Mapping[str, Any] | None = None,
  saved_env: Mapping[str, Any] | None = None,
) -> ResolvedOslObservation:
  """The ``osl`` group a checkpoint was trained with (module docstring): its key,
  else its run's ``params/env.yaml``, else the legacy set, with a warning.

  Args:
    checkpoint: The checkpoint file; its run directory holds ``params/``.
    payload: The loaded checkpoint, if at hand (it is loaded otherwise).
    saved_env: The run's ``params/env.yaml`` as plain data, if at hand (the file
      is read otherwise).

  Raises:
    ValueError: If the checkpoint's key and its run's env.yaml record different
      input layouts (a checkpoint moved into another run's directory).
  """
  checkpoint = Path(checkpoint)
  if payload is None:
    payload = load_checkpoint(checkpoint)
  own = checkpoint_osl_observation(payload)
  env_yaml = checkpoint.parent / "params" / "env.yaml"
  if saved_env is None and env_yaml.is_file():
    saved_env = read_config(env_yaml)
  recorded = saved_env_osl_observation(saved_env) if saved_env is not None else None
  if own is not None and recorded is not None:
    if osl_obs_layout(own) != osl_obs_layout(recorded):
      raise ValueError(
        f"{checkpoint} records the OSL sensors {describe_osl_sensors(own)}, but its "
        f"run's params/env.yaml records {describe_osl_sensors(recorded)}. Was the "
        "checkpoint moved into another run's directory?"
      )
  if own is not None:
    return ResolvedOslObservation(
      own, f"the checkpoint's {OSL_OBSERVATION_KEY!r}", True
    )
  if recorded is not None:
    return ResolvedOslObservation(
      recorded, f"{env_yaml}; the checkpoint predates the key", True
    )
  print(f"[WARN] {checkpoint}: the OSL sensor set is {LEGACY_SOURCE}.")
  return ResolvedOslObservation(legacy_osl_record(), LEGACY_SOURCE, False)


def setting_differences(
  trained: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
  """The lag, error-bound and noise settings two records of the same layout set
  differently, as ``"<term>.<setting> <trained> -> <current>"``."""
  differences = []
  for name, ours in current["terms"].items():
    theirs = trained["terms"].get(name)
    if ours is None or theirs is None:
      continue
    for key in TERM_SETTINGS:
      if key.startswith("delay") and key in theirs and theirs[key] != ours.get(key):
        differences.append(f"{name}.{key} {theirs[key]} -> {ours.get(key)}")
    for key in sorted(set(theirs["params"]) | set(ours["params"])):
      if key in _LAYOUT_PARAMS:
        continue
      a, b = theirs["params"].get(key), ours["params"].get(key)
      if a != b:
        differences.append(f"{name}.params.{key} {a} -> {b}")
  return differences


def check_osl_observation(
  payload: Mapping[str, Any], path: str | Path, env: Any
) -> ResolvedOslObservation:
  """Refuse a checkpoint trained on another OSL input layout (terms, channels,
  history) than ``env`` builds; only warn when lags, error bounds or noise alone
  differ.

  Returns the checkpoint's resolved record.

  Raises:
    ValueError: On a layout mismatch, naming the flags that build the checkpoint's
      set, or for a checkpoint that contradicts its run's env.yaml.
  """
  resolved = run_osl_observation(path, payload)
  current = osl_group_record(env_osl_group(env))
  if osl_obs_layout(current) != osl_obs_layout(resolved.record):
    flags = osl_sensor_flags(resolved.record)
    raise ValueError(
      f"{path} was trained with the OSL sensors "
      f"{describe_osl_sensors(resolved.record)} (from {resolved.source}), but this "
      f"env builds {describe_osl_sensors(current)}. The OSL actor reads its inputs "
      "by position, so another set, even of the same width, feeds it other "
      "signals. Launch with exactly these OSL sensor flags: "
      + (flags or "none (the default set)")
      + "."
    )
  differences = setting_differences(resolved.record, current)
  if differences:
    print(
      f"[WARN] {path}: the OSL observation's lags, error bounds or noise differ "
      f"from those it trained with ({resolved.source}): {'; '.join(differences)}."
    )
  return resolved


def env_osl_group(env: Any) -> Any:
  """The ``osl`` group config of an env (wrapped or not).

  Raises:
    ValueError: If its config has none.
  """
  unwrapped = getattr(env, "unwrapped", env)
  observations = getattr(getattr(unwrapped, "cfg", None), "observations", None) or {}
  group = observations.get(OSL_OBS_GROUP)
  if group is None:
    raise ValueError(f"The env's config has no {OSL_OBS_GROUP!r} observation group.")
  return group
