"""The contact between the OSL and the biological left leg.

Every OSL imitation scene (stage 1, stage 2, stairs, stage 1.5) carries one contact
sensor between the prosthesis's subtree and the left leg's
(:func:`leg_contact_sensor_cfg`, added by :func:`add_leg_contact_sensor`). It reads
the net contact force between the two subtrees in every physics substep of a control
step. A sensor changes no physics.

The imitation reward reads it twice (``mdp.rewards.mimic_reward``): as diagnostics
(:data:`LEG_CONTACT_DIAGNOSTICS`: the contact rate and the hardest strike) and as the
optional ``leg_contact`` regularizer, which charges the share of the step's substeps
whose force exceeds a threshold (:func:`contact_fraction`).

What it cannot see: the OSL foot's capsules have ``conaffinity`` 0, like every
biological leg capsule, so a prosthetic foot crossing the left leg makes no contact
at all. The six OSL hulls (socket, pylons, adapter, knee and ankle assemblies)
collide with the leg's capsules (thigh, shank, foot and toes).

The sensor is a training measurement, not part of the deployed plant: the stage-1
exporters leave it out of a bundle's model (:func:`drop_leg_contact_sensor`), whose
bytes it would otherwise change. Stage-2 bundles have always carried it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mjlab.sensor import ContactMatch, ContactSensorCfg

LEG_CONTACT_SENSOR = "osl_leg_contact"
"""The scene contact sensor between the OSL and the biological left leg (a scene
sensor: not prefixed)."""
OSL_ROOT_BODY = "OSL_KA_L1_osl_generic_socket"
"""The root of every OSL body in both sockets (its parent is ``femur_r``)."""
LEFT_LEG_BODY = "femur_l"
"""The root of the biological left leg: femur, patella, tibia, talus, calcaneus and
toes."""
LEG_CONTACT_DIAGNOSTICS: tuple[str, ...] = ("leg_contact", "leg_contact_force")
"""The imitation reward's leg-contact diagnostics, present when the scene has the
sensor: ``leg_contact`` is 1 where an OSL hull touched the left leg in any physics
substep of the step (its mean is a contact rate), ``leg_contact_force`` the largest
net contact force over those substeps (N)."""


def leg_contact_sensor_cfg(substeps: int) -> ContactSensorCfg:
  """The contact sensor between the OSL's subtree and the biological left leg's.

  It reads the net force of every contact between the two subtrees (MuJoCo's
  ``mjSENS_CONTACT``, which mujoco_warp computes too), kept for the last
  ``substeps`` physics substeps, one control step, so a strike between two control
  steps still shows. The six OSL hulls collide with the leg's capsules; the OSL
  foot capsules do not (``conaffinity`` 0 on both sides), so a foot crossing the
  biological leg makes no contact at all.
  """
  return ContactSensorCfg(
    name=LEG_CONTACT_SENSOR,
    primary=ContactMatch(mode="subtree", pattern=f"^{OSL_ROOT_BODY}$", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=LEFT_LEG_BODY, entity="robot"),
    fields=("force",),
    reduce="netforce",
    history_length=substeps,
  )


def add_leg_contact_sensor(cfg: Any) -> None:
  """Give an env config's scene the leg-contact sensor, over one control step's
  substeps (``cfg.decimation``), unless it has one: in place, with a new sensors
  tuple."""
  sensors = tuple(cfg.scene.sensors)
  if any(sensor.name == LEG_CONTACT_SENSOR for sensor in sensors):
    return
  cfg.scene.sensors = (*sensors, leg_contact_sensor_cfg(cfg.decimation))


def drop_leg_contact_sensor(cfg: Any) -> None:
  """Take the leg-contact sensor out of an env config's scene, in place, with a new
  sensors tuple: a stage-1 bundle's model is the deployed plant, which never had
  it."""
  sensors = tuple(cfg.scene.sensors)
  kept = tuple(sensor for sensor in sensors if sensor.name != LEG_CONTACT_SENSOR)
  if len(kept) != len(sensors):
    cfg.scene.sensors = kept


def scene_sensor(env: Any, name: str) -> Any | None:
  """The scene's sensor ``name``, or None (a scene without it, or a fake env)."""
  sensors = getattr(getattr(env, "scene", None), "sensors", None)
  return sensors.get(name) if isinstance(sensors, Mapping) else None


def substep_forces(data: Any) -> torch.Tensor:
  """(B, H) the net contact force's magnitude in each of the last H physics
  substeps (newest first), the largest over the sensor's primaries, from its
  ``ContactData``; without a substep history, (B, 1) the current force."""
  history = getattr(data, "force_history", None)  # (B, P, H, 3)
  if history is not None:
    return history.norm(dim=-1).amax(dim=1)
  return data.force.norm(dim=-1).amax(dim=-1, keepdim=True)  # (B, P, 3) -> (B, 1)


def contact_fraction(forces: torch.Tensor, threshold: float) -> torch.Tensor:
  """(B,) the share of substeps whose force exceeds ``threshold`` (N), in [0, 1]:
  mjlab's ``self_collision_cost`` (the substeps with a contact over the threshold),
  divided by the substeps."""
  return (forces > threshold).float().mean(dim=-1)
