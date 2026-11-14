"""The foot touch-site version, chosen per run and recorded where tools find it.

``osl_touch_sites`` is ``"feet_extended"`` (the default for every newly configured
run: the OSL heel zone plus the intact foot's zones lowered to below its colliders),
``"heel_extended"`` (the OSL heel zone only, the default for part of 2026-11-13) or
``"original"`` (the model XML's zones, which every run before 2026-11-13 trained
with); :data:`~mjlab.asset_zoo.robots.osl_fullbody.osl_constants.TOUCH_SITE_BOXES`
has the boxes. Only the human actor (stage 1's unified policy, stage 2's and stage
1.5's human) and the critics observe touch, but the version is a property of the
model, so every OSL task carries it:

* **the env config**: :class:`OslEnvCfg`'s ``osl_touch_sites`` field
  (``--env.osl-touch-sites``), a plain entry of the run's ``params/env.yaml``. It is
  copied into the robot entity (``OslEntityCfg.touch_sites``), which builds it;
* **every checkpoint** the OSL runners save: the top-level key
  :data:`CHECKPOINT_KEY`;
* **the stage-2 warm start's lineage** (``osl_split.rl.warmstart``);
* **every deployment bundle**: the manifest's ``touch_sites``
  (:func:`touch_sites_manifest`), read from the bundle's own ``model.mjb``.

Anything that records no version means ``"original"``. A policy never meets other
zones: resuming refuses a mismatch (:func:`check_checkpoint_touch_sites`), stage 2
refuses a warm start trained on other zones, and play, evaluation and export build
the version the checkpoint was trained with (:func:`run_touch_sites`). The stage-1
normalizer keeps ``r_foot`` at a mean of about 0.0015 N and a std of about 1.4 N, so
a heel load of 200-700 N would reach an ``original`` policy at hundreds of
normalized units; the lowered left zones read loads in quiet stance that the older
ones missed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import mujoco
import torch

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  DEFAULT_TOUCH_SITES,
  LEGACY_TOUCH_SITES,
  R_FOOT_TOUCH_SITE,
  TOUCH_SITE_EDITS,
  TOUCH_SITE_SENSORS,
  OslEntityCfg,
  TouchSites,
  check_touch_sites,
  touch_sites_of_model,
)
from mjlab.envs import ManagerBasedRlEnvCfg

ENV_FIELD = "osl_touch_sites"
"""The env config field, and its key in ``params/env.yaml``."""
CHECKPOINT_KEY = "osl_touch_sites"
"""The top-level key of every checkpoint the OSL runners save."""
CLI_FLAG = "--env.osl-touch-sites"
ROBOT = "robot"
"""The OSL entity's name in the scene, whose sites are prefixed ``robot/``."""
TOUCH_SITES_FRAME = (
  "site pos/quat/size in the parent body's frame (MuJoCo site_pos, site_quat, "
  "site_size; box half-sizes)"
)
"""The manifest's ``touch_sites.frame``."""
_ENV_YAML_LINE = re.compile(rf"^{ENV_FIELD}:[ \t]*['\"]?(\w+)['\"]?[ \t]*$", re.M)


@dataclass(kw_only=True)
class OslEnvCfg(ManagerBasedRlEnvCfg):
  """The env config of every OSL full-body task: its robot entity builds the OSL
  foot's touch zones that ``osl_touch_sites`` names."""

  osl_touch_sites: TouchSites = DEFAULT_TOUCH_SITES
  """The foot touch-sensor zones. ``feet_extended`` (the default) extends the OSL
  foot's ``r_foot`` zone back over the heel, where it touches down, and lowers the
  intact foot's ``l_foot`` and ``l_toes`` zones 8 mm, so that they catch a foot
  resting in the contact margin; ``heel_extended`` has only the ``r_foot`` change;
  ``original`` is the model XML's zones, whose ``r_foot`` never fires, and what
  every run before 2026-11-13 trained with. Resuming, warm-starting or playing a
  checkpoint needs the zones it was trained with."""

  def __post_init__(self) -> None:
    # tyro builds the config after applying the command line, so this sees the
    # final value.
    sync_touch_sites(self)


def sync_touch_sites(cfg: OslEnvCfg) -> None:
  """Give the robot entity the config's ``osl_touch_sites``.

  The config gets a new scene config rather than an edited one: ``dataclasses.
  replace`` and the field-copying conversions share the scene between configs.

  Raises:
    ValueError: For an unknown version.
    TypeError: If the scene's ``robot`` entity is not an ``OslEntityCfg``.
  """
  value = check_touch_sites(cfg.osl_touch_sites, ENV_FIELD)
  robot = cfg.scene.entities.get(ROBOT)
  if not isinstance(robot, OslEntityCfg):
    raise TypeError(
      f"An OSL env config needs an OslEntityCfg {ROBOT!r} entity (from "
      f"get_osl_robot_cfg or get_osl_msk_cfg); got {type(robot).__name__}."
    )
  if robot.touch_sites != value:
    entities = {**cfg.scene.entities, ROBOT: replace(robot, touch_sites=value)}
    cfg.scene = replace(cfg.scene, entities=entities)


def set_touch_sites(cfg: ManagerBasedRlEnvCfg, touch_sites: TouchSites) -> None:
  """Set an OSL env config's touch-site version, robot entity included.

  Raises:
    TypeError: If ``cfg`` is not an :class:`OslEnvCfg`.
    ValueError: For an unknown version.
  """
  if not isinstance(cfg, OslEnvCfg):
    raise TypeError(f"Not an OSL env config: {type(cfg).__name__}.")
  cfg.osl_touch_sites = check_touch_sites(touch_sites, ENV_FIELD)
  sync_touch_sites(cfg)


def env_touch_sites(env: Any) -> TouchSites:
  """The touch-site version an env simulates: its config's ``osl_touch_sites``,
  checked against its compiled model when that has the foot touch sites.

  Raises:
    TypeError: If the env's config has no ``osl_touch_sites`` (not an OSL task).
    RuntimeError: If the config and the model disagree.
  """
  value = getattr(getattr(env, "cfg", None), ENV_FIELD, None)
  if value is None:
    raise TypeError(
      f"The env's config has no {ENV_FIELD!r}: not an OSL full-body task (OslEnvCfg)."
    )
  value = check_touch_sites(value, ENV_FIELD)
  model = getattr(getattr(env, "sim", None), "mj_model", None)
  if isinstance(model, mujoco.MjModel):
    built = touch_sites_of_model(model, f"{ROBOT}/")
    if built is not None and built != value:
      raise RuntimeError(
        f"The env config's {ENV_FIELD} is {value!r}, but its model was built with "
        f"the {built!r} zones. Change it with set_touch_sites(), which also sets "
        "the robot entity's."
      )
  return value


##
# Checkpoints and runs.
##


class ResolvedTouchSites(NamedTuple):
  """A checkpoint's touch-site version and where it was found."""

  value: TouchSites
  source: str


def load_checkpoint(path: str | Path) -> dict[str, Any]:
  """A checkpoint on CPU, memory-mapped: cheap to read a key from."""
  return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def checkpoint_touch_sites(payload: Mapping[str, Any]) -> TouchSites | None:
  """The version a checkpoint records; None when it has no :data:`CHECKPOINT_KEY`
  (saved before 2026-11-13, so ``"original"``)."""
  value = payload.get(CHECKPOINT_KEY)
  return None if value is None else check_touch_sites(value, CHECKPOINT_KEY)


def saved_env_touch_sites(saved: Mapping[str, Any]) -> TouchSites | None:
  """The version a run's ``params/env.yaml``, read as plain data, records; None
  when it has no ``osl_touch_sites`` field (a run before 2026-11-13)."""
  value = saved.get(ENV_FIELD)
  return None if value is None else check_touch_sites(value, f"env.yaml {ENV_FIELD}")


def env_yaml_touch_sites(path: str | Path) -> TouchSites | None:
  """:func:`saved_env_touch_sites` of a ``params/env.yaml`` file (None when the
  file or the field is missing), from its top-level ``osl_touch_sites:`` line."""
  try:
    text = Path(path).read_text(encoding="utf-8")
  except FileNotFoundError:
    return None
  match = _ENV_YAML_LINE.search(text)
  return None if match is None else check_touch_sites(match[1], f"{path} {ENV_FIELD}")


def run_touch_sites(
  checkpoint: str | Path,
  payload: Mapping[str, Any] | None = None,
  saved_env: Mapping[str, Any] | None = None,
) -> ResolvedTouchSites:
  """The touch-site version a checkpoint was trained with: its own
  :data:`CHECKPOINT_KEY`, else ``"original"`` (every OSL runner records the key
  since 2026-11-13, and every checkpoint saved before trained with the model XML's
  zones).

  The run's ``params/env.yaml`` field, where there is one, must agree. A
  checkpoint whose run records other zones (one without the key inside a run that
  records ``heel_extended``, say) was moved there from another run, and is refused
  rather than built either way.

  Args:
    checkpoint: The checkpoint file; its run directory holds ``params/``.
    payload: The loaded checkpoint, if at hand (it is loaded otherwise).
    saved_env: The run's ``params/env.yaml`` as plain data, if at hand (the file
      is read otherwise).

  Raises:
    ValueError: If the checkpoint and its run's env.yaml disagree.
  """
  checkpoint = Path(checkpoint)
  if payload is None:
    payload = load_checkpoint(checkpoint)
  own = checkpoint_touch_sites(payload)
  trained = own or LEGACY_TOUCH_SITES
  env_yaml = checkpoint.parent / "params" / "env.yaml"
  recorded = (
    saved_env_touch_sites(saved_env)
    if saved_env is not None
    else env_yaml_touch_sites(env_yaml)
  )
  if recorded is not None and recorded != trained:
    what = (
      f"records OSL touch sites {own!r}"
      if own is not None
      else "records no OSL touch sites (so it predates the setting and trained "
      f"with {LEGACY_TOUCH_SITES!r})"
    )
    raise ValueError(
      f"{checkpoint} {what}, but its run's params/env.yaml records {recorded!r}. "
      "Was the checkpoint moved into another run's directory?"
    )
  if own is not None:
    return ResolvedTouchSites(own, f"the checkpoint's {CHECKPOINT_KEY!r} key")
  if recorded is not None:
    return ResolvedTouchSites(trained, f"{env_yaml}; the checkpoint predates the key")
  return ResolvedTouchSites(
    trained,
    "recorded nowhere: the checkpoint predates the setting, and every run before "
    "2026-11-13 trained with the model XML's zones",
  )


def check_checkpoint_touch_sites(
  payload: Mapping[str, Any], path: str | Path, env: Any
) -> TouchSites:
  """Refuse a checkpoint trained with other touch sites than ``env`` simulates
  (:func:`run_touch_sites` decides what it trained with).

  Returns the checkpoint's version.

  Raises:
    ValueError: On a mismatch, naming the flag that fixes it, or for a checkpoint
      that contradicts its run's env.yaml.
  """
  trained = run_touch_sites(path, payload).value
  current = env_touch_sites(env)
  if trained != current:
    why = (
      ""
      if checkpoint_touch_sites(payload) is not None
      else (
        " (it records none: every checkpoint saved before 2026-11-13 trained with "
        "the 'original' zones)"
      )
    )
    raise ValueError(
      f"{path} was trained with OSL touch sites {trained!r}{why}, but this env "
      f"builds {current!r}. A checkpoint only meets the zones it trained with: "
      "a policy that observes the foot touch sensors would read inputs it never "
      "saw (the original r_foot zone never fires, so under the extended one a heel "
      "load reaches the network at hundreds of normalized units; the "
      "feet_extended left zones read quiet-stance loads the others miss). "
      f"Relaunch with {CLI_FLAG} {trained}."
    )
  return trained


def check_checkpoint_file(path: str | Path, env: Any) -> TouchSites:
  """:func:`check_checkpoint_touch_sites` of a checkpoint file."""
  return check_checkpoint_touch_sites(load_checkpoint(path), path, env)


def configure_for_checkpoint(
  cfg: ManagerBasedRlEnvCfg, checkpoint: str | Path
) -> ResolvedTouchSites:
  """Make ``cfg`` build the touch sites ``checkpoint`` was trained with
  (:func:`run_touch_sites`), in place, and say so: for play, which takes no
  ``--env`` overrides."""
  resolved = run_touch_sites(checkpoint)
  set_touch_sites(cfg, resolved.value)
  print(f"[INFO] OSL touch sites: {resolved.value} (from {resolved.source}).")
  return resolved


##
# Deployment bundles.
##


def touch_sites_manifest(model: mujoco.MjModel, prefix: str = f"{ROBOT}/") -> dict:
  """A bundle manifest's ``touch_sites``: the version and the four touch sites as
  ``model`` compiled them (the contract shared with osl_deploy).

  Raises:
    ValueError: If the model lacks a touch site, a site is not a box, or a site
      does not carry exactly its touch sensor.
  """
  version = touch_sites_of_model(model, prefix)
  if version is None:
    raise ValueError(f"The model has no site {prefix + R_FOOT_TOUCH_SITE!r}.")
  sites: dict[str, dict[str, Any]] = {}
  for site_name, sensor_name in TOUCH_SITE_SENSORS.items():
    name = prefix + site_name
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
      raise ValueError(f"The model has no touch site {name!r}.")
    if model.site_type[site] != mujoco.mjtGeom.mjGEOM_BOX:
      raise ValueError(f"Touch site {name!r} is not a box.")
    sensors = [
      model.sensor(i).name
      for i in range(model.nsensor)
      if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH
      and model.sensor_objtype[i] == mujoco.mjtObj.mjOBJ_SITE
      and model.sensor_objid[i] == site
    ]
    if sensors != [prefix + sensor_name]:
      raise ValueError(
        f"Touch site {name!r} carries the touch sensors {sensors}, not "
        f"[{prefix + sensor_name!r}]."
      )
    sites[name] = {
      "sensor": sensors[0],
      "body": model.body(int(model.site_bodyid[site])).name,
      "type": "box",
      "pos": [float(v) for v in model.site_pos[site]],
      "quat": [float(v) for v in model.site_quat[site]],
      "size": [float(v) for v in model.site_size[site]],
    }
  return {
    "version": version,
    "edited_from_xml": [prefix + site for site in TOUCH_SITE_EDITS[version]],
    "frame": TOUCH_SITES_FRAME,
    "sites": sites,
  }


_README_WHAT: dict[TouchSites, str] = {
  "original": "the model XML's: the OSL heel lands behind the `r_foot` zone, which "
  "reads 0",
  "heel_extended": "the OSL foot's `r_foot` zone extended back over the heel",
  "feet_extended": "the OSL foot's `r_foot` zone extended back over the heel, and "
  "the intact foot's `l_foot` and `l_toes` zones extended 8 mm down, below its "
  "colliders",
}


def touch_sites_readme_line(touch_sites: Mapping[str, Any]) -> str:
  """The README's line on a manifest's ``touch_sites``."""

  def site(sensor: str) -> Mapping[str, Any]:
    return next(
      s for s in touch_sites["sites"].values() if s["sensor"].endswith(f"/{sensor}")
    )

  r_foot, l_foot = site("r_foot"), site("l_foot")
  back = r_foot["pos"][0] - r_foot["size"][0]
  front = r_foot["pos"][0] + r_foot["size"][0]
  sole = l_foot["pos"][1] - l_foot["size"][1]
  version = check_touch_sites(touch_sites["version"], "touch_sites.version")
  return (
    f"- Touch zones: **{version}**, {_README_WHAT[version]} (OSL foot x {back:.3f} "
    f"to {front:.3f} m, intact foot zones down to y {sole:.3f} m; manifest "
    "`touch_sites`)"
  )
