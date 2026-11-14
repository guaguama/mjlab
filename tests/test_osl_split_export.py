"""Two-controller deployment export of stage-2 split runs (CPU only).

A four-clip library is generated from the real model by forward kinematics, and a
fake stage-2 run directory holds the task's own saved configs and two small
actors, so ``export_bundle`` runs end to end on both models. The OSL observation
conformance sequence is checked against a NumPy rebuild from the bundle's own
files, the way a deployment consumer builds it. A robot run from before the OSL
touch-site setting exports too.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort
import pytest
import torch
import yaml
from tensordict import TensorDict

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  OSL_XMLS,
  TouchSites,
  get_spec,
)
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_mimic.touch_sites import set_touch_sites
from mjlab.tasks.osl_split import export as split_export
from mjlab.tasks.osl_split.config.env_cfgs import osl_split_env_cfg
from mjlab.tasks.osl_split.config.rl_cfg import osl_split_runner_cfg
from mjlab.utils.os import dump_yaml

DT = 0.01
HIDDEN = (16, 16)
OSL_OBS = 600
# (model, socket, reward) -> (human obs width, env action terms).
CASES = {
  ("robot", "rigid", "mimic"): (635, [["servos", 27], ["osl_motors", 2]]),
  ("msk", "compliant", "clf"): (2338, [["osl_motors", 2], ["muscles", 339]]),
}
ROBOT = ("robot", "rigid", "mimic")
R_FOOT = "robot/OSL_KA_L1_r_foot_touch"
L_FOOT = "robot/foot_touch_l"
L_TOES = "robot/toes_touch_l"
# The manifest's ``touch_sites`` of a feet_extended model (the default): the
# contract shared with osl_deploy
# (logs/handover/round6/briefs/brief_mjlab_feet_touch_zones.md), verbatim.
TOUCH_SITES_CONTRACT: dict[str, Any] = {
  "version": "feet_extended",
  "edited_from_xml": [R_FOOT, L_FOOT, L_TOES],
  "frame": (
    "site pos/quat/size in the parent body's frame (MuJoCo site_pos, site_quat, "
    "site_size; box half-sizes)"
  ),
  "sites": {
    R_FOOT: {
      "sensor": "robot/r_foot",
      "body": "robot/OSL_KA_L1_osl_foot_assembly",
      "type": "box",
      "pos": [0.01, -0.04, 0.0],
      "quat": [1.0, 0.0, 0.0, 0.0],
      "size": [0.065, 0.03, 0.05],
    },
    "robot/OSL_KA_L1_r_toes_touch": {
      "sensor": "robot/r_toes",
      "body": "robot/OSL_KA_L1_osl_foot_assembly",
      "type": "box",
      "pos": [0.215, -0.04, -0.005],
      "quat": [1.0, 0.0, 0.0, 0.0],
      "size": [0.075, 0.03, 0.06],
    },
    "robot/foot_touch_l": {
      "sensor": "robot/l_foot",
      "body": "robot/calcn_l",
      "type": "box",
      "pos": [0.09, -0.014, 0.0],
      "quat": [1.0, 0.0, 0.0, 0.0],
      "size": [0.1, 0.014, 0.055],
    },
    "robot/toes_touch_l": {
      "sensor": "robot/l_toes",
      "body": "robot/toes_l",
      "type": "box",
      "pos": [0.0275, -0.014, 0.0],
      "quat": [0.9393726845901743, 0.0, 0.3428978848663966, 0.0],
      "size": [0.04, 0.014, 0.0675],
    },
  },
}


def _expected_touch_sites(version: str) -> dict[str, Any]:
  """The manifest ``touch_sites`` of ``version``: the default's contract with the
  sites ``version`` does not edit back at the model XML's boxes (feet_extended
  edits a superset of heel_extended's sites, with the same r_foot box)."""
  edited = {
    "original": [],
    "heel_extended": [R_FOOT],
    "feet_extended": [R_FOOT, L_FOOT, L_TOES],
  }[version]
  xml_model = mujoco.MjModel.from_xml_path(str(OSL_XMLS[("robot", "rigid")]))
  expected = copy.deepcopy(TOUCH_SITES_CONTRACT)
  expected.update(version=version, edited_from_xml=edited)
  for name, site in expected["sites"].items():
    if name not in edited:
      xml = xml_model.site(name.removeprefix("robot/"))
      site.update(pos=xml.pos.tolist(), quat=xml.quat.tolist(), size=xml.size.tolist())
  return expected


def _readme_zones(osl_x: str, sole: str) -> str:
  return (
    f"(OSL foot x {osl_x} m, intact foot zones down to y {sole} m; manifest "
    "`touch_sites`)"
  )


def _clip(model, data, frames, vel, wz):
  """``frames`` of the start pose moving at world velocity ``vel`` and yaw rate
  ``wz``, as clipper's mjlab format stores them."""
  base = model.qpos0.copy()
  pos, quat, joints = [], [], []
  for t in range(frames):
    q = base.copy()
    q[:2] = np.asarray(vel[:2]) * t * DT
    yaw = wz * t * DT
    q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    pos.append(data.xpos[1:].copy())
    quat.append(data.xquat[1:].copy())
    joints.append(q[7:].copy())
  body_pos = np.stack(pos)
  ang = np.zeros_like(body_pos)
  ang[..., 2] = wz
  return {
    "joint_pos": np.stack(joints),
    "joint_vel": np.zeros((frames, model.nq - 7)),
    "body_pos_w": body_pos,
    "body_quat_w": np.stack(quat),
    "body_lin_vel_w": np.gradient(body_pos, DT, axis=0),
    "body_ang_vel_w": ang,
    "fps": np.array([1.0 / DT]),
  }


def _se2(x, y, yaw):
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def library(tmp_path_factory) -> Path:
  folder = tmp_path_factory.mktemp("library")
  model = get_spec("robot", "rigid").compile()
  data = mujoco.MjData(model)
  frames, v, wz = 50, 0.8, 0.6
  # Forward is the root's body -y, which is world -y at zero yaw.
  clips = {
    "idle": ((0, 0, 0), (0.0, 0.0, 0.0), 0.0, _se2(0, 0, 0)),
    "walk_fwd": ((0, -v, 0), (v, 0.0, 0.0), 0.0, _se2(0, -v * frames * DT, 0)),
    "walk_bck": ((0, v, 0), (-v, 0.0, 0.0), 0.0, _se2(0, v * frames * DT, 0)),
    "walk_turn": ((0, 0, 0), (0.0, 0.0, wz), wz, _se2(0, 0, wz * frames * DT)),
  }
  for name, (vel, twist, rate, stride) in clips.items():
    clip = _clip(model, data, frames, vel, rate)
    np.savez(folder / f"{name}.npz", twist=np.array(twist), stride_se2=stride, **clip)
  return folder


def _actor(group: str, obs_dim: int, action_dim: int, distribution: dict):
  obs = TensorDict({group: torch.zeros(1, obs_dim)}, batch_size=[1])
  actor = MuscleMimicMLPModel(
    obs,
    {"actor": [group]},
    "actor",
    action_dim,
    hidden_dims=HIDDEN,
    distribution_cfg=distribution,
  )
  actor.obs_normalizer.mean.uniform_(-1.0, 1.0)  # type: ignore[union-attr]
  actor.obs_normalizer.var.uniform_(0.5, 2.0)  # type: ignore[union-attr]
  return actor


def _make_run(
  root: Path,
  case,
  library: Path,
  legacy: bool = False,
  touch_sites: TouchSites | None = None,
) -> Path:
  """A run on the default touch sites, recorded in its env.yaml, its checkpoint
  and its warm start's lineage; a ``legacy`` run predates the setting, so none of
  them records any (``original``)."""
  model, socket, reward = case
  obs_dim, terms = CASES[case]
  run = root / f"{model}_{socket}" / "run"
  env_cfg = osl_split_env_cfg(model, socket, reward, "train")
  env_cfg.commands["motion"].library_dir = str(library)  # type: ignore[attr-defined]
  agent = osl_split_runner_cfg(model, socket, reward, "exp")
  agent.actor.hidden_dims = HIDDEN
  agent.osl_actor.hidden_dims = HIDDEN
  if touch_sites is not None:
    set_touch_sites(env_cfg, touch_sites)
  env = asdict(env_cfg)
  if legacy:
    del env["osl_touch_sites"], env["scene"]["entities"]["robot"]["touch_sites"]
  dump_yaml(run / "params" / "env.yaml", env)
  dump_yaml(run / "params" / "agent.yaml", asdict(agent))
  torch.manual_seed(0)
  human_dim = sum(int(dim) for name, dim in terms if name != "osl_motors")
  human = _actor("actor", obs_dim, human_dim, dict(agent.actor.distribution_cfg or {}))
  osl = _actor("osl", OSL_OBS, 2, dict(agent.osl_actor.distribution_cfg or {}))
  lineage = {
    "model": model,
    "socket": socket,
    "action_terms": terms,
    "stage1_checkpoint": "/stage1/run/model_4.pt",
  }
  payload = {
    "human_actor_state_dict": human.state_dict(),
    "osl_actor_state_dict": osl.state_dict(),
    "iter": 7,
    "stage_iteration": 2,
    "infos": {"split": {"stage_start": 5, "lineage": lineage}},
  }
  if not legacy:  # Stage 1 and stage 2 trained on the same touch sites.
    lineage["osl_touch_sites"] = payload["osl_touch_sites"] = env_cfg.osl_touch_sites
  torch.save(payload, run / "model_7.pt")
  return run / "model_7.pt"


@pytest.fixture(scope="module", params=list(CASES), ids=lambda c: "-".join(c))
def exported(request, tmp_path_factory, library):
  checkpoint = _make_run(tmp_path_factory.mktemp("runs"), request.param, library)
  bundle = split_export.export_bundle(checkpoint)
  manifest = json.loads((bundle / "manifest.json").read_text())
  return request.param, checkpoint, bundle, manifest


@pytest.fixture(scope="module")
def legacy(tmp_path_factory, library):
  """The bundle and manifest of a robot run from before the touch-site setting."""
  checkpoint = _make_run(tmp_path_factory.mktemp("legacy"), ROBOT, library, True)
  bundle = split_export.export_bundle(checkpoint)
  return bundle, json.loads((bundle / "manifest.json").read_text())


def _check_model_touch_sites(bundle: Path, touch_sites: dict) -> mujoco.MjModel:
  """The bundle's model.mjb carries each touch site exactly as the manifest says."""
  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  for name, entry in touch_sites["sites"].items():
    site = model.site(name)
    assert model.body(int(site.bodyid[0])).name == entry["body"]
    assert site.type[0] == mujoco.mjtGeom.mjGEOM_BOX and entry["type"] == "box"
    for key in ("pos", "quat", "size"):
      assert getattr(site, key).tolist() == entry[key], (name, key)
    sensor = model.sensor(entry["sensor"])
    assert sensor.type[0] == mujoco.mjtSensor.mjSENS_TOUCH
    assert (sensor.objtype[0], sensor.objid[0]) == (mujoco.mjtObj.mjOBJ_SITE, site.id)
  return model


def test_bundle_contract(exported) -> None:
  case, checkpoint, bundle, m = exported
  obs_dim, terms = CASES[case]
  assert bundle == checkpoint.parent / "deploy" / "model_7"
  assert (m["schema_version"], m["kind"]) == (2, "osl_split_dual")
  assert m["run"]["task"].startswith("Mjlab-Split-Unicycle-OSL-Fullbody")
  assert m["composition"]["env_action_terms"] == terms
  human, osl = m["controllers"]["human"], m["controllers"]["osl"]
  assert human["observation"]["schema"] == "osl_mimic_gait_v1"
  assert human["io"]["input_shape"] == [1, obs_dim]
  assert osl["observation"]["schema"] == "osl_split_sensors_v1"
  assert osl["io"]["input_shape"] == [1, OSL_OBS]
  assert osl["observation"]["terms"] == ["actpack", "loadcell", "last_action"]
  # Each controller's blocks start at 0 in its own output, and together they
  # drive every actuator exactly once.
  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  ids = []
  for name, section in m["controllers"].items():
    blocks = section["action"]["blocks"]
    assert blocks[0]["start"] == 0
    assert [b["name"] for b in blocks] == (
      ["osl_motors"] if name == "osl" else [t for t, _ in terms if t != "osl_motors"]
    )
    ids += [model.actuator(n).id for n in section["action"]["actuator_names"]]
  assert sorted(ids) == list(range(model.nu))
  assert osl["action"]["default_inference"] == "mean"
  # Fixture std: MSK 3.0 (expected), robot 0.2 (mean).
  assert human["action"]["default_inference"] == (
    "expected" if case[0] == "msk" else "mean"
  )
  assert osl["training"]["actuator_delay"]["osl_motors"]["seconds"] == [0.0, 0.02]
  buses = osl["observation"]["latency"]["buses"]
  assert set(buses) == {"actpack", "loadcell"}
  assert buses["actpack"] == {
    "training_steps": [0, 2],
    "held_per_episode": True,
    "validation_steps": 1,
  }
  # The OSL sensors are compiled into the bundle's model.
  for sensor in osl["observation"]["sensors"]["sensors"]:
    assert model.sensor(sensor["scene_name"]).dim[0] == sensor["dim"]
  for name, digest in m["files"].items():
    assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest


def test_onnx_parity_and_metadata(exported) -> None:
  _, _, bundle, m = exported
  parity = np.load(bundle / "parity.npz")
  for name, section in m["controllers"].items():
    session = ort.InferenceSession(
      str(bundle / section["onnx"]), providers=["CPUExecutionProvider"]
    )
    io = section["io"]
    assert session.get_inputs()[0].name == io["input"]
    assert session.get_outputs()[0].shape == io["output_shape"]
    out = np.stack(
      [
        np.asarray(session.run(None, {io["input"]: o[None]})[0])[0]
        for o in parity[f"{name}_observations"]
      ]
    )
    np.testing.assert_allclose(out, parity[f"{name}_means"], atol=1e-6)
    meta = json.loads(session.get_modelmeta().custom_metadata_map["osl_policy"])
    assert meta["controller"] == name and meta["kind"] == "osl_split_dual"


def test_osl_conformance_matches_a_numpy_rebuild(exported) -> None:
  """A consumer's OSL observation, from model.mjb and the manifest alone,
  reproduces mjlab's own terms and buffers at every training latency."""
  _, _, bundle, m = exported
  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  obs = m["controllers"]["osl"]["observation"]
  sensors = {s["name"]: s for s in obs["sensors"]["sensors"]}
  tare = obs["sensors"]["load_cell_tare"]["values"]
  parity = np.load(bundle / "parity.npz")
  c = {k: parity[f"osl_conformance_{k}"] for k in ("qpos", "qvel", "act", "ctrl")}
  last = parity["osl_conformance_last_action"]
  assert np.all(last[0] == 0.0) and np.abs(last).max() > 1.0

  def read(data, name):
    s = model.sensor(sensors[name]["scene_name"])
    value = data.sensordata[s.adr[0] : s.adr[0] + s.dim[0]].copy()
    if sensors[name]["saturation"] is not None:
      value = np.clip(value, -sensors[name]["saturation"], sensors[name]["saturation"])
    return value - np.asarray(tare.get(name, 0.0))

  data = mujoco.MjData(model)
  frames = []
  for t in range(len(last)):
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = c["qpos"][t], c["qvel"][t]
    data.act[:], data.ctrl[:] = c["act"][t], c["ctrl"][t]
    mujoco.mj_forward(model, data)
    frames.append(
      {
        term["name"]: np.clip(last[t], -1.0, 1.0)
        if term["name"] == "last_action"
        else np.concatenate([read(data, s) for s in term["sensors"]])
        for term in obs["term_layout"]
      }
    )
  flat = np.stack(
    [np.concatenate([f[t["name"]] for t in obs["term_layout"]]) for f in frames]
  )
  np.testing.assert_allclose(
    flat, parity["osl_conformance_frames"], rtol=1e-5, atol=1e-4
  )
  for lag in parity["osl_conformance_lags"]:
    rows = []
    for k in range(len(frames)):
      parts = []
      for term in obs["term_layout"]:
        delay = int(lag) if term["name"] in obs["latency"]["buses"] else 0
        h = term["history_length"]
        # History slot j (0 oldest) holds the delayed frame of step k - (h-1-j),
        # backfilled before the reset; step s's delayed frame is max(s-L, 0)'s.
        parts += [
          frames[max(max(k - (h - 1 - j), 0) - delay, 0)][term["name"]]
          for j in range(h)
        ]
      rows.append(np.concatenate(parts))
    np.testing.assert_allclose(
      np.stack(rows),
      parity[f"osl_conformance_obs_lag{int(lag)}"],
      rtol=1e-5,
      atol=1e-4,
    )


def test_refuses_stage1_runs_and_osl_drift(exported, tmp_path) -> None:
  _, checkpoint, _, _ = exported
  run = checkpoint.parent
  payload = torch.load(checkpoint, weights_only=False)
  stage1 = tmp_path / "stage1.pt"
  torch.save({"actor_state_dict": payload["human_actor_state_dict"]}, stage1)
  with pytest.raises(ValueError, match="stage-1 checkpoint"):
    split_export.load_payload(stage1)

  agent = yaml.safe_load(
    (run / "params/agent.yaml").read_text().replace("!!python/tuple", "")
  )
  agent["algorithm"]["class_name"] = "mjlab.tasks.osl_mimic.rl.ppo:MuscleMimicPPO"
  with pytest.raises(ValueError, match="not a stage-2 split run"):
    split_export.split_run_spec(checkpoint, agent)

  cfg = osl_split_env_cfg("msk", "compliant", "clf", "train")
  saved = split_export.gl.read_config(run / "params/env.yaml")["observations"]["osl"]
  saved["terms"]["actpack"]["params"]["projected_gravity"] = True
  saved["terms"]["actpack"]["delay_max_lag"] = 3
  group = cfg.observations["osl"]
  split_export.apply_osl_group(group, saved)
  assert group.terms["actpack"].params["projected_gravity"] is True
  assert group.terms["actpack"].delay_max_lag == 3
  saved["terms"]["loadcell"]["func"] = "somewhere.else"
  with pytest.raises(ValueError, match="loadcell.func"):
    split_export.apply_osl_group(cfg.observations["osl"], saved)


def test_bundle_declares_its_touch_sites(exported) -> None:
  """A run on the default zones: the schema-2 manifest's ``touch_sites`` is the
  contract shared with osl_deploy, model.mjb carries exactly those sites, and the
  README says which zones the bundle has."""
  _, _, bundle, m = exported
  assert m["schema_version"] == 2
  assert m["touch_sites"] == TOUCH_SITES_CONTRACT
  _check_model_touch_sites(bundle, m["touch_sites"])
  readme = (bundle / "README.md").read_text()
  assert "\n- Touch zones: **feet_extended**, " in readme
  assert _readme_zones("-0.055 to 0.075", "-0.028") in readme


def test_legacy_run_exports_the_xml_touch_sites(legacy) -> None:
  """A run from before the setting records no touch sites (nor does its
  lineage), so it trained with the model XML's: its bundle says ``original``,
  edits nothing, and both the manifest and model.mjb carry the XML's ``r_foot``
  box; the other zones never change."""
  bundle, m = legacy
  assert m["schema_version"] == 2
  xml_model = mujoco.MjModel.from_xml_path(str(OSL_XMLS[("robot", "rigid")]))
  assert m["touch_sites"] == _expected_touch_sites("original")
  model = _check_model_touch_sites(bundle, m["touch_sites"])
  for name in m["touch_sites"]["sites"]:
    xml = xml_model.site(name.removeprefix("robot/"))
    for key in ("pos", "quat", "size"):
      np.testing.assert_array_equal(getattr(model.site(name), key), getattr(xml, key))
  readme = (bundle / "README.md").read_text()
  assert "\n- Touch zones: **original**, " in readme
  assert _readme_zones("-0.025 to 0.075", "-0.020") in readme


def test_heel_extended_run_exports_its_touch_sites(library, tmp_path) -> None:
  """A run from the first fix (the extended heel, the XML's left zones) still
  exports its own zones under the new default."""
  checkpoint = _make_run(tmp_path, ROBOT, library, touch_sites="heel_extended")
  bundle = split_export.export_bundle(checkpoint)
  m = json.loads((bundle / "manifest.json").read_text())
  assert m["touch_sites"] == _expected_touch_sites("heel_extended")
  _check_model_touch_sites(bundle, m["touch_sites"])
  readme = (bundle / "README.md").read_text()
  assert "\n- Touch zones: **heel_extended**, " in readme
  assert _readme_zones("-0.055 to 0.075", "-0.020") in readme


def test_refuses_a_lineage_on_other_touch_sites(library, tmp_path) -> None:
  """Stage 2 trains the human on stage 1's touch sites: a run whose warm-start
  lineage records others, or none (a stage 1 from before the setting, so
  ``original``), is refused."""
  checkpoint = _make_run(tmp_path, ROBOT, library)
  payload = torch.load(checkpoint, weights_only=False)
  lineage = payload["infos"]["split"]["lineage"]
  mismatch = (
    "trained with the OSL touch sites 'feet_extended', but its warm start's stage "
    "1 with 'original'"
  )
  recorded, unrecorded = (checkpoint.with_name(f"model_{i}.pt") for i in (8, 9))
  lineage["osl_touch_sites"] = "original"
  torch.save(payload, recorded)
  del lineage["osl_touch_sites"]
  torch.save(payload, unrecorded)
  for path in (recorded, unrecorded):
    with pytest.raises(ValueError, match=mismatch):
      split_export.export_bundle(path)
