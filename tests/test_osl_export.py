"""Deployment export of the OSL gait-library policies, robot and MSK (CPU only).

A four-clip library (idle, forward, backward, pivot) is generated from the real
model by forward kinematics, and a fake run directory holds the task's own saved
configs and a small actor, so ``export_bundle`` runs end to end on both models.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import yaml
from tensordict import TensorDict

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import get_spec
from mjlab.tasks.osl_gait_library import export as gl_export
from mjlab.tasks.osl_gait_library.config.env_cfgs import osl_gait_library_env_cfg
from mjlab.tasks.osl_gait_library.config.rl_cfg import osl_gait_library_runner_cfg
from mjlab.tasks.osl_mimic.rl import export as policy_export
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.utils.os import dump_yaml

DT = 0.01
HIDDEN = (16, 16)
# (model, socket, reward) -> (observation width, action width, blocks): the schema.
CASES = {
  ("robot", "rigid", "mimic"): (635, 29, [("servos", 27), ("osl_motors", 2)]),
  ("msk", "compliant", "clf"): (2338, 341, [("osl_motors", 2), ("muscles", 339)]),
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


def _make_run(root: Path, case, library: Path) -> Path:
  model, socket, reward = case
  obs_dim, action_dim, _ = CASES[case]
  run = root / f"{model}_{socket}" / "run"
  env_cfg = osl_gait_library_env_cfg(model, socket, reward, "train", "unicycle")
  env_cfg.commands["motion"].library_dir = str(library)  # type: ignore[attr-defined]
  agent = osl_gait_library_runner_cfg(model, socket, reward, "exp", "unicycle")
  agent.actor.hidden_dims = HIDDEN
  dump_yaml(run / "params" / "env.yaml", asdict(env_cfg))
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
  torch.save({"actor_state_dict": actor.state_dict(), "iter": 7}, run / "model_7.pt")
  return run / "model_7.pt"


@pytest.fixture(scope="module", params=list(CASES), ids=lambda c: "-".join(c))
def exported(request, tmp_path_factory, library):
  checkpoint = _make_run(tmp_path_factory.mktemp("runs"), request.param, library)
  bundle = gl_export.export_bundle(checkpoint)
  return request.param, checkpoint, bundle


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
  # Robot policies run on the raw mean, MSK ones on the noise-averaged action.
  assert action["default_inference"] == ("expected" if model == "msk" else "mean")
  for b in action["blocks"]:
    span = slice(b["start"], b["start"] + b["size"])
    low = 0.0 if b["name"] == "muscles" else -1.0
    assert set(action["expected_low"][span]) == {low}
    assert set(action["expected_high"][span]) == {1.0}
  np.testing.assert_allclose(action["std"], 3.0 if model == "msk" else 0.2, rtol=1e-6)
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
