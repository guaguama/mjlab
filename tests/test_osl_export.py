"""Deployment export of the OSL gait-library policies, robot and MSK (CPU only).

A four-clip library (idle, forward, backward, pivot) is generated from the real
model by forward kinematics, and a fake run directory holds the task's own saved
configs and a small actor, so ``export_bundle`` runs end to end on both models. A
robot run from before the OSL touch-site setting exports too.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest
import torch
import yaml
from tensordict import TensorDict

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import OSL_XMLS, get_spec
from mjlab.tasks.osl_gait_library import export as gl_export
from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg
from mjlab.tasks.osl_gait_library.config.rl_cfg import (
  UNICYCLE_MSK_INIT_STD,
  osl_gait_library_runner_cfg,
)
from mjlab.tasks.osl_mimic.rl import export as policy_export
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_mimic.touch_sites import set_touch_sites
from mjlab.utils.os import dump_yaml

DT = 0.01
HIDDEN = (16, 16)
# (model, socket, reward) -> (observation width, action width, blocks): the schema.
CASES = {
  ("robot", "rigid", "mimic"): (635, 29, [("servos", 27), ("osl_motors", 2)]),
  ("msk", "compliant", "clf"): (2338, 341, [("osl_motors", 2), ("muscles", 339)]),
}
ROBOT = ("robot", "rigid", "mimic")
R_FOOT = "robot/OSL_KA_L1_r_foot_touch"
# The manifest's ``touch_sites`` of a heel_extended model: the contract shared with
# osl_deploy (logs/handover/touch_fix/brief_mjlab.md), verbatim.
TOUCH_SITES_CONTRACT: dict[str, Any] = {
  "version": "heel_extended",
  "edited_from_xml": [R_FOOT],
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
      "pos": [0.09, -0.01, 0.0],
      "quat": [1.0, 0.0, 0.0, 0.0],
      "size": [0.1, 0.01, 0.055],
    },
    "robot/toes_touch_l": {
      "sensor": "robot/l_toes",
      "body": "robot/toes_l",
      "type": "box",
      "pos": [0.0275, -0.01, 0.0],
      "quat": [0.9393726845901743, 0.0, 0.3428978848663966, 0.0],
      "size": [0.04, 0.01, 0.0675],
    },
  },
}


def _clip(model, data, frames, vel, wz, root_xy):
  """``frames`` of the start pose moving at world velocity ``vel`` and yaw rate
  ``wz``, as clipper's mjlab format stores them."""
  base = model.qpos0.copy()
  pos, quat, joints = [], [], []
  for t in range(frames):
    q = base.copy()
    q[:2] = np.asarray(root_xy) + np.asarray(vel[:2]) * t * DT
    yaw = wz * t * DT
    q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    pos.append(data.xpos[1:].copy())
    quat.append(data.xquat[1:].copy())
    joints.append(q[7:].copy())
  body_pos = np.stack(pos)
  lin = np.gradient(body_pos, DT, axis=0)
  ang = np.zeros_like(lin)
  ang[..., 2] = wz
  return {
    "joint_pos": np.stack(joints),
    "joint_vel": np.zeros((frames, model.nq - 7)),
    "body_pos_w": body_pos,
    "body_quat_w": np.stack(quat),
    "body_lin_vel_w": lin,
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
  frames, v, wz = 40, 0.6, 1.0
  # Forward is the root's body -y, which is world -y at zero yaw.
  clips = {
    "idle": ((0, 0, 0), (0.0, 0.0, 0.0), 0.0, _se2(0, 0, 0)),
    "walk_fwd": ((0, -v, 0), (v, 0.0, 0.0), 0.0, _se2(0, -v * frames * DT, 0)),
    "walk_bck": ((0, v, 0), (-v, 0.0, 0.0), 0.0, _se2(0, v * frames * DT, 0)),
    "walk_turn": ((0, 0, 0), (0.0, 0.0, wz), wz, _se2(0, 0, wz * frames * DT)),
  }
  for name, (vel, twist, rate, stride) in clips.items():
    clip = _clip(model, data, frames, vel, rate, (0.0, 0.0))
    np.savez(folder / f"{name}.npz", twist=np.array(twist), stride_se2=stride, **clip)
  return folder


def _make_run(root: Path, case, library: Path, legacy: bool = False) -> Path:
  """A run on the default touch sites; a ``legacy`` run predates the setting, so
  neither its env.yaml nor its checkpoint records any (``original``)."""
  model, socket, reward = case
  obs_dim, action_dim, _ = CASES[case]
  run = root / f"{model}_{socket}" / "run"
  env_cfg = osl_gait_library_env_cfg(model, socket, reward, "train", "unicycle")
  env_cfg.commands["motion"].library_dir = str(library)  # type: ignore[attr-defined]
  agent = osl_gait_library_runner_cfg(model, socket, reward, "exp", "unicycle")
  agent.actor.hidden_dims = HIDDEN
  env = asdict(env_cfg)
  if legacy:
    del env["osl_touch_sites"], env["scene"]["entities"]["robot"]["touch_sites"]
  dump_yaml(run / "params" / "env.yaml", env)
  dump_yaml(run / "params" / "agent.yaml", asdict(agent))
  torch.manual_seed(0)
  obs = TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1])
  actor = MuscleMimicMLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    action_dim,
    hidden_dims=HIDDEN,
    distribution_cfg=dict(agent.actor.distribution_cfg or {}),
  )
  actor.obs_normalizer.mean.uniform_(-1.0, 1.0)  # type: ignore[union-attr]
  actor.obs_normalizer.var.uniform_(0.5, 2.0)  # type: ignore[union-attr]
  payload = {"actor_state_dict": actor.state_dict(), "iter": 7}
  if not legacy:  # The OSL runners record the touch sites in every checkpoint.
    payload["osl_touch_sites"] = env_cfg.osl_touch_sites
  torch.save(payload, run / "model_7.pt")
  return run / "model_7.pt"


@pytest.fixture(scope="module", params=list(CASES), ids=lambda c: "-".join(c))
def exported(request, tmp_path_factory, library):
  checkpoint = _make_run(tmp_path_factory.mktemp("runs"), request.param, library)
  bundle = gl_export.export_bundle(checkpoint)
  return request.param, checkpoint, bundle


@pytest.fixture(scope="module")
def legacy_bundle(tmp_path_factory, library) -> Path:
  """The bundle of a robot run from before the touch-site setting."""
  checkpoint = _make_run(tmp_path_factory.mktemp("legacy"), ROBOT, library, True)
  return gl_export.export_bundle(checkpoint)


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
  case, checkpoint, bundle = exported
  model, socket, reward = case
  obs_dim, action_dim, blocks = CASES[case]
  assert bundle == checkpoint.parent / "deploy" / "model_7"
  m = json.loads((bundle / "manifest.json").read_text())
  assert m["run"]["task"] == (
    "Mjlab-Gait-Library-Unicycle-OSL-Fullbody"
    + ("-MSK" if model == "msk" else "-ROBOT")
    + ("-CLF" if reward == "clf" else "")
    + ("-Compliant" if socket == "compliant" else "")
  )
  assert m["socket"] == socket and m["provenance"]["iteration"] == 7
  # osl_deploy's schema: version, names, the term list with muscles iff MSK.
  assert m["schema_version"] == 1 and m["observation"]["schema"] == "osl_mimic_gait_v1"
  assert m["observation"]["size"] == obs_dim
  assert ("muscles" in m["observation"]["terms"]) == (model == "msk")
  assert len(m["observation"]["muscle_names"]) == (339 if model == "msk" else 0)
  assert len(m["joint_names"]) == (77 if socket == "compliant" else 73)
  action = m["action"]
  assert [(b["name"], b["size"]) for b in action["blocks"]] == blocks
  assert action["size"] == action_dim == len(action["std"])
  # Robot policies run on the raw mean; MSK ones, at the fixture's muscle std of
  # 1.0 (above 0.3), on the noise-averaged action.
  assert action["default_inference"] == ("expected" if model == "msk" else "mean")
  for b in action["blocks"]:
    span = slice(b["start"], b["start"] + b["size"])
    low = 0.0 if b["name"] == "muscles" else -1.0
    assert set(action["expected_low"][span]) == {low}
    assert set(action["expected_high"][span]) == {1.0}
  # The fake checkpoint keeps the runner config's initial std (the unicycle MSK
  # recipe's; its per-term OSL value is set by the PPO, not by the model).
  init_std = UNICYCLE_MSK_INIT_STD if model == "msk" else 0.2
  np.testing.assert_allclose(action["std"], init_std, rtol=1e-6)
  # Unicycle DR: OSL motors 0-20 ms, the human 20-40 ms, at 2 ms physics steps.
  delays = m["training"]["actuator_delay"]
  human = "muscles" if model == "msk" else "servos"
  assert delays["osl_motors"]["steps"] == [0, 10]
  assert delays[human]["steps"] == [10, 20]
  np.testing.assert_allclose(delays[human]["seconds"], [0.02, 0.04])
  with np.load(bundle / "reference.npz") as ref:
    assert ref["twists"].shape == (4, 3)
    assert ref["joint_pos"].shape[-1] == len(m["joint_names"])
  assert m["reference"]["clip_names"] == ["idle", "walk_bck", "walk_fwd", "walk_turn"]
  assert (bundle / "README.md").read_text().startswith(f"# {m['run']['task']}")


def test_bundle_files_and_onnx(exported) -> None:
  import onnx
  import onnxruntime as ort

  _, checkpoint, bundle = exported
  m = json.loads((bundle / "manifest.json").read_text())
  assert set(m["files"]) == set(gl_export.CHECKED_FILES)
  for name, digest in m["files"].items():
    assert gl_export.sha256(bundle / name) == digest
  assert m["validation"]["onnx_max_abs_error"] < 1e-4
  embedded = {
    p.key: p.value for p in onnx.load(str(bundle / "actor.onnx")).metadata_props
  }
  assert json.loads(embedded[policy_export.METADATA_KEY]) == {
    k: v for k, v in m.items() if k != "files"
  }
  session = ort.InferenceSession(str(bundle / "actor.onnx"))
  (inp,), (out,) = session.get_inputs(), session.get_outputs()
  assert (inp.name, inp.shape) == ("observation", [1, m["observation"]["size"]])
  assert (out.name, out.shape) == ("mean", [1, m["action"]["size"]])
  with np.load(bundle / "parity.npz") as parity:
    out = session.run(None, {"observation": parity["observation"][None]})
    mean = np.asarray(out[0])[0]
    np.testing.assert_allclose(mean, parity["mean"], atol=1e-6)
  with pytest.raises(FileExistsError):
    gl_export.export_bundle(checkpoint)


def test_config_drift_and_missing_library_refuse(exported, tmp_path) -> None:
  _, checkpoint, _ = exported
  run = tmp_path / "run"
  (run / "params").mkdir(parents=True)
  shutil.copy(checkpoint, run / "model_7.pt")  # Its touch-site key is read first.
  env = gl_export.read_config(checkpoint.parent / "params/env.yaml")
  agent = (checkpoint.parent / "params/agent.yaml").read_text()
  (run / "params/agent.yaml").write_text(agent)
  env["decimation"] = 4
  env["sim"]["mujoco"]["iterations"] = 7
  (run / "params/env.yaml").write_text(yaml.safe_dump(env, sort_keys=False))
  with pytest.raises(ValueError, match=r"sim\.mujoco\.iterations, decimation"):
    gl_export.build_components(run / "model_7.pt")
  env = gl_export.read_config(checkpoint.parent / "params/env.yaml")
  env["commands"]["motion"]["library_dir"] = str(tmp_path / "elsewhere")
  (run / "params/env.yaml").write_text(yaml.safe_dump(env, sort_keys=False))
  with pytest.raises(ValueError, match="--library"):
    gl_export.build_components(run / "model_7.pt")


def test_bundle_declares_its_touch_sites(exported) -> None:
  """A run on the default zones: the manifest's ``touch_sites`` is the contract
  shared with osl_deploy, model.mjb carries exactly those sites, and the README
  says which zones the bundle has."""
  _, _, bundle = exported
  m = json.loads((bundle / "manifest.json").read_text())
  assert m["touch_sites"] == TOUCH_SITES_CONTRACT
  _check_model_touch_sites(bundle, m["touch_sites"])
  readme = (bundle / "README.md").read_text()
  assert "\n- Touch zones: **heel_extended**, " in readme
  assert "(foot x -0.055 to 0.075 m; manifest `touch_sites`)" in readme


def test_legacy_run_exports_the_xml_touch_sites(legacy_bundle) -> None:
  """A run from before the setting records no touch sites, so it trained with the
  model XML's: its bundle says ``original``, edits nothing, and both the manifest
  and model.mjb carry the XML's ``r_foot`` box; the other zones never change."""
  m = json.loads((legacy_bundle / "manifest.json").read_text())
  xml_model = mujoco.MjModel.from_xml_path(str(OSL_XMLS[("robot", "rigid")]))
  xml = xml_model.site(R_FOOT.removeprefix("robot/"))
  expected = copy.deepcopy(TOUCH_SITES_CONTRACT)
  expected.update(version="original", edited_from_xml=[])
  expected["sites"][R_FOOT].update(
    pos=xml.pos.tolist(), quat=xml.quat.tolist(), size=xml.size.tolist()
  )
  assert m["touch_sites"] == expected
  model = _check_model_touch_sites(legacy_bundle, m["touch_sites"])
  for key in ("pos", "quat", "size"):
    np.testing.assert_array_equal(getattr(model.site(R_FOOT), key), getattr(xml, key))
  readme = (legacy_bundle / "README.md").read_text()
  assert "\n- Touch zones: **original**, " in readme
  assert "(foot x -0.025 to 0.075 m; manifest `touch_sites`)" in readme


def test_touch_site_drift_and_disagreement_refuse(library, tmp_path) -> None:
  """The rebuilt config's touch sites are checked against the run's env.yaml (one
  without the field trained with ``original``), and a checkpoint whose key
  disagrees with its run's env.yaml is refused."""
  checkpoint = _make_run(tmp_path, ROBOT, library)
  saved = gl_export.read_config(checkpoint.parent / "params/env.yaml")
  cfg = osl_gait_library_env_cfg("robot", "rigid", "mimic", "train", "unicycle")
  assert saved["osl_touch_sites"] == cfg.osl_touch_sites == "heel_extended"
  assert gl_export.config_mismatches(cfg, saved) == []
  drift = ["osl_touch_sites (original -> heel_extended)"]
  saved["osl_touch_sites"] = "original"
  assert gl_export.config_mismatches(cfg, saved) == drift
  del saved["osl_touch_sites"]
  assert gl_export.config_mismatches(cfg, saved) == drift
  set_touch_sites(cfg, "original")
  assert gl_export.config_mismatches(cfg, saved) == []

  payload = torch.load(checkpoint, weights_only=False)
  payload["osl_touch_sites"] = "original"
  moved = checkpoint.with_name("model_8.pt")
  torch.save(payload, moved)
  disagree = r"touch sites 'original', but its run's params/env\.yaml records 'heel"
  with pytest.raises(ValueError, match=disagree):
    gl_export.build_components(moved)


def test_run_spec_reads_the_validation_cfg(tmp_path) -> None:
  checkpoint = tmp_path / "exp" / "run" / "model_1.pt"
  v = {"model": "msk", "socket": "rigid", "reward": "clf"}
  spec = gl_export.run_spec(checkpoint, {"validation": v, "experiment_name": "e"})
  # Forward-library runs predate the variant field.
  assert (spec.variant, spec.task_id) == (
    "walk_fwd",
    "Mjlab-Gait-Library-OSL-Fullbody-MSK-CLF",
  )
  assert spec.run == "run"
  with pytest.raises(ValueError, match="not a gait-library run"):
    gl_export.run_spec(checkpoint, {"validation": {"interval": 30}})


def test_unsupported_processing_allows_training_noise_only() -> None:
  raw = {"noise": None, "clip": None, "scale": None, "delay_min_lag": 0}
  terms = {name: dict(raw) for name in ("root_pos", "joint_pos", "goal")}
  assert policy_export.unsupported_processing(terms) == []
  terms["joint_pos"]["noise"] = {"n_min": -0.01, "n_max": 0.01}
  assert policy_export.unsupported_processing(terms) == []
  terms["root_pos"]["history_length"] = 3
  terms["goal"]["delay_max_lag"] = 2
  terms["goal"]["clip"] = [-1, 1]
  assert policy_export.unsupported_processing(terms, group_history=4) == [
    "root_pos.history_length",
    "goal.clip",
    "goal.delay_max_lag",
    "history_length",
  ]


def _fake_bundle(tmp_path: Path) -> Path:
  bundle = tmp_path / "deploy" / "model_9"
  bundle.mkdir(parents=True)
  manifest = {"run": {"run": "my_run", "task": "Mjlab-Task"}}
  (bundle / "manifest.json").write_text(json.dumps(manifest))
  return bundle


def test_upload_falls_back_to_the_hf_tool(tmp_path, monkeypatch) -> None:
  bundle = _fake_bundle(tmp_path)
  monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # Import fails.
  calls = []
  monkeypatch.setattr(gl_export.shutil, "which", lambda name: f"/bin/{name}")
  monkeypatch.setattr(
    gl_export.subprocess,
    "run",
    lambda cmd, check: calls.append(cmd) or SimpleNamespace(returncode=0),
  )
  url = gl_export.upload_bundle(bundle, "me/policies")
  assert url == "https://huggingface.co/me/policies/tree/main/my_run/model_9"
  (cmd,) = calls
  assert cmd[:5] == ["/bin/hf", "upload", "me/policies", str(bundle), "my_run/model_9"]
  assert "--private" in cmd
  # Neither huggingface_hub nor the tool: the command to run by hand.
  monkeypatch.setattr(gl_export.shutil, "which", lambda name: None)
  with pytest.raises(RuntimeError, match="hf upload me/policies"):
    gl_export.upload_bundle(bundle, "me/policies", "custom/path", private=False)


def test_bundle_solver_caps_converge_on_cpu(exported, tmp_path) -> None:
  """Bundles get at least 10/20 solver iterations (the OSL tasks train at 4/8, where
  CPU MuJoCo truncates its contact solve), recorded in the manifest; flags override."""
  _, checkpoint, bundle = exported
  m = json.loads((bundle / "manifest.json").read_text())
  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  assert (model.opt.iterations, model.opt.ls_iterations) == (10, 20)
  assert m["deployment"]["solver"] == {"iterations": 10, "ls_iterations": 20}
  assert m["deployment"]["training_solver"] == {"iterations": 4, "ls_iterations": 8}
  assert "10 Newton, 20" in (bundle / "README.md").read_text()
  custom = gl_export.export_bundle(
    checkpoint, tmp_path / "b", solver_iterations=50, ls_solver_iterations=50
  )
  model = mujoco.MjModel.from_binary_path(str(custom / "model.mjb"))
  assert (model.opt.iterations, model.opt.ls_iterations) == (50, 50)


def test_default_inference_follows_the_muscle_std(exported) -> None:
  """The manifest's default is the noise-averaged action only for a muscle policy
  whose median muscle std exceeds 0.3; a low-std MSK policy (the stage-1 recipe
  ends at 0.07) and every robot policy run on the raw mean. The OSL motors' std
  does not count."""
  case, checkpoint, _ = exported
  model = case[0]
  obs_dim, action_dim, _ = CASES[case]
  parts = gl_export.build_components(checkpoint).parts
  low, high = np.full(action_dim, 0.07), np.full(action_dim, 1.0)
  assert policy_export.default_inference(parts, low) == "mean"
  assert policy_export.default_inference(parts, high) == (
    "expected" if model == "msk" else "mean"
  )
  manifest = policy_export.policy_metadata(parts, low, obs_dim, None)
  assert manifest["action"]["default_inference"] == "mean"
  if model == "msk":
    osl_noisy = low.copy()
    osl_noisy[:2] = 3.0  # The OSL motors come first on the MSK model.
    assert policy_export.default_inference(parts, osl_noisy) == "mean"
