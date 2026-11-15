"""Deployment export of stairs runs, stage 1 and stage 2 (CPU only).

A short climbing clip is generated from the real model by forward kinematics,
with a two-box terrain sidecar, and fake run directories hold the tasks' own
saved configs, the terrain copy and small actors, so both exporters run end to
end. The full-body observation conformance set is checked against a NumPy
rebuild from the bundle's own files, as a deployment consumer builds it; the
stage-2 OSL sequence against unicycle stage 2's rebuild.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
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
from test_osl_split_export import (
  test_osl_conformance_matches_a_numpy_rebuild as check_osl_conformance,
)

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import get_spec
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_split import mdp as split_mdp
from mjlab.tasks.osl_split_stairs import export as split_stairs_export
from mjlab.tasks.osl_split_stairs.config.env_cfgs import osl_split_stairs_env_cfg
from mjlab.tasks.osl_split_stairs.config.rl_cfg import osl_split_stairs_runner_cfg
from mjlab.tasks.osl_stairs import export as stairs_export
from mjlab.tasks.osl_stairs.config.env_cfgs import osl_stairs_env_cfg, sync_terrain
from mjlab.tasks.osl_stairs.config.rl_cfg import osl_stairs_runner_cfg
from mjlab.tasks.osl_stairs.terrain import check_scene_terrain, load_terrain
from mjlab.utils.os import dump_yaml

DT = 0.01
FRAMES = 120
HIDDEN = (16, 16)
OSL_OBS = 500
# The two-box flight of the stairs task tests (TERRA's upstairs04 yaw).
BOXES = [
  {
    "pos": [-0.78, 0.05, 0.1],
    "size": [0.2, 0.5, 0.1],
    "yaw": 0.825,
    "pitch": 0.0,
    "name": "terrain_box_0",
  },
  {
    "pos": [-0.56, 0.29, 0.2],
    "size": [0.12, 0.5, 0.2],
    "yaw": 0.825,
    "pitch": 0.0,
    "name": "terrain_box_1",
  },
]
# (model, socket, reward) -> (full-body obs width, env action terms).
CASES = {
  ("robot", "rigid", "mimic"): (630, [["servos", 27], ["osl_motors", 2]]),
  ("msk", "compliant", "clf"): (2333, [["osl_motors", 2], ["muscles", 339]]),
}
TERMS = ["root_pos", "joint_pos", "root_vel", "joint_vel", "muscles", "touch", "goal"]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
  """A clip that walks, turns and rises, with moving joints, and its terrain."""
  folder = tmp_path_factory.mktemp("stairs")
  model = get_spec("robot", "rigid").compile()
  data = mujoco.MjData(model)
  rng = np.random.default_rng(0)
  phase = rng.uniform(0, 2 * math.pi, model.nq - 7)
  pos, quat, joints = [], [], []
  for t in range(FRAMES):
    q = model.qpos0.copy()
    q[:3] += [0.3 * t * DT, -0.5 * t * DT, 0.4 * t * DT]
    yaw = 0.4 * t * DT
    q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    q[7:] += 0.05 * np.sin(2 * math.pi * t * DT + phase)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    pos.append(data.xpos[1:].copy())
    quat.append(data.xquat[1:].copy())
    joints.append(q[7:].copy())
  body_pos, joint_pos = np.stack(pos), np.stack(joints)
  ang = np.zeros_like(body_pos)
  ang[..., 2] = 0.4
  np.savez(
    folder / "climb.npz",
    joint_pos=joint_pos,
    joint_vel=np.gradient(joint_pos, DT, axis=0),
    body_pos_w=body_pos,
    body_quat_w=np.stack(quat),
    body_lin_vel_w=np.gradient(body_pos, DT, axis=0),
    body_ang_vel_w=ang,
    fps=np.array([1.0 / DT]),
  )
  (folder / "climb_terrain.json").write_text(json.dumps({"boxes": BOXES}))
  return folder / "climb.npz"


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


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(run: Path, env_cfg: Any, agent: Any, clip: Path) -> None:
  motion = env_cfg.commands["motion"]
  motion.motion_file = str(clip)
  sync_terrain(env_cfg)
  dump_yaml(run / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(run / "params" / "agent.yaml", asdict(agent))
  shutil.copyfile(
    clip.with_name(clip.stem + "_terrain.json"), run / "params/terrain.json"
  )


def _stage1_run(root: Path, case, clip: Path) -> Path:
  model, socket, reward = case
  obs_dim, terms = CASES[case]
  run = root / "stage1" / f"{model}_{socket}"
  env_cfg = osl_stairs_env_cfg(model, socket, reward, "train")
  agent = osl_stairs_runner_cfg(model, socket, reward, "exp")
  agent.actor.hidden_dims = HIDDEN
  _write_run(run, env_cfg, agent, clip)
  torch.manual_seed(0)
  action_dim = sum(int(dim) for _, dim in terms)
  actor = _actor("actor", obs_dim, action_dim, dict(agent.actor.distribution_cfg or {}))
  payload = {
    "actor_state_dict": actor.state_dict(),
    "iter": 4,
    "osl_touch_sites": env_cfg.osl_touch_sites,
  }
  torch.save(payload, run / "model_4.pt")
  return run / "model_4.pt"


def _stage2_run(root: Path, case, clip: Path, stage1: Path) -> Path:
  model, socket, reward = case
  obs_dim, terms = CASES[case]
  run = root / "stage2" / f"{model}_{socket}"
  env_cfg = osl_split_stairs_env_cfg(model, socket, reward, "train")
  agent = osl_split_stairs_runner_cfg(model, socket, reward, "exp")
  agent.actor.hidden_dims = HIDDEN
  agent.osl_actor.hidden_dims = HIDDEN
  _write_run(run, env_cfg, agent, clip)
  torch.manual_seed(0)
  human_dim = sum(int(dim) for name, dim in terms if name != "osl_motors")
  human = _actor("actor", obs_dim, human_dim, dict(agent.actor.distribution_cfg or {}))
  osl = _actor("osl", OSL_OBS, 2, dict(agent.osl_actor.distribution_cfg or {}))
  lineage = {
    "model": model,
    "socket": socket,
    "action_terms": terms,
    "stage1_checkpoint": str(stage1),
    "stage1_run_dir": str(stage1.parent),
    "osl_touch_sites": env_cfg.osl_touch_sites,
  }
  payload = {
    "human_actor_state_dict": human.state_dict(),
    "osl_actor_state_dict": osl.state_dict(),
    "iter": 7,
    "stage_iteration": 2,
    "infos": {
      "split": {"stage_start": 5, "lineage": lineage},
      "stairs": {
        "clip": clip.name,
        "terrain_sha256": _sha256(run / "params/terrain.json"),
      },
    },
    "osl_touch_sites": env_cfg.osl_touch_sites,
    "osl_observation": split_mdp.osl_observation_entry(env_cfg.observations["osl"]),
  }
  torch.save(payload, run / "model_7.pt")
  return run / "model_7.pt"


@pytest.fixture(scope="module", params=list(CASES), ids=lambda c: "-".join(c))
def exported(request, tmp_path_factory, clip):
  """Both stages of one case: (case, clip, stage-1 and stage-2 (checkpoint, bundle,
  manifest))."""
  root = tmp_path_factory.mktemp("runs")
  stage1 = _stage1_run(root, request.param, clip)
  stage2 = _stage2_run(root, request.param, clip, stage1)
  out = {}
  for name, checkpoint, module in (
    ("stage1", stage1, stairs_export),
    ("stage2", stage2, split_stairs_export),
  ):
    bundle = module.export_bundle(checkpoint)
    out[name] = (checkpoint, bundle, json.loads((bundle / "manifest.json").read_text()))
  return request.param, clip, out


def _check_files_and_terrain(bundle: Path, m: dict[str, Any], clip: Path) -> None:
  for name, digest in m["files"].items():
    assert _sha256(bundle / name) == digest
  assert "terrain.json" in m["files"]
  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  terrain = bundle / "terrain.json"
  check_scene_terrain(model, load_terrain(terrain), terrain)
  t = m["terrain"]
  assert (t["box_count"], t["body"], t["file"]) == (2, "terrain", "terrain.json")
  assert t["sha256"] == _sha256(terrain)
  assert [b["geom"] for b in t["boxes"]] == ["terrain_box_0", "terrain_box_1"]
  # The reference: the clip's root and sites, unshifted, (T, ...) float32.
  r = np.load(bundle / "reference.npz")
  assert int(r["num_frames"]) == FRAMES and r["num_frames"].shape == ()
  with np.load(clip) as source:
    np.testing.assert_allclose(r["root_pos"], source["body_pos_w"][:, 0], atol=1e-6)
  assert r["site_pos"].shape == (FRAMES, 17, 3) and r["rpos"].shape == (FRAMES, 16, 3)
  assert all(r[k].dtype == np.float32 for k in r if k != "num_frames")
  np.testing.assert_allclose(
    r["rpos"], r["site_pos"][:, 1:] - r["site_pos"][:, :1], atol=1e-6
  )
  ref = m["reference"]
  assert (ref["kind"], ref["num_frames"], ref["start_frame"]) == (
    "single_clip",
    FRAMES,
    0,
  )
  assert (ref["clip"], ref["clip_file"]) == ("climb", "climb.npz")
  assert m["run"]["variant"] == "stairs"
  assert m["touch_sites"]["version"] == "feet_extended"
  assert m["failure"]["site_deviation"]["threshold_m"] == 0.5


def test_stage1_bundle_contract(exported) -> None:
  case, clip, out = exported
  checkpoint, bundle, m = out["stage1"]
  obs_dim, terms = CASES[case]
  assert bundle == checkpoint.parent / "deploy" / "model_4"
  assert m["schema_version"] == 1 and "kind" not in m
  assert m["run"]["task"].startswith("Mjlab-Stairs-OSL-Fullbody")
  assert m["run"]["stage"] == 1
  obs = m["observation"]
  assert obs["schema"] == "osl_mimic_v1" and obs["size"] == obs_dim
  assert obs["terms"] == [t for t in TERMS if case[0] == "msk" or t != "muscles"]
  assert [b["name"] for b in m["action"]["blocks"]] == [t for t, _ in terms]
  _check_files_and_terrain(bundle, m, clip)


def test_stage2_bundle_contract(exported) -> None:
  case, clip, out = exported
  _, bundle, m = out["stage2"]
  obs_dim, terms = CASES[case]
  assert (m["schema_version"], m["kind"]) == (2, "osl_split_dual")
  assert m["run"]["task"].startswith("Mjlab-Split-Stairs-OSL-Fullbody")
  assert m["run"]["recorded_clip"]["clip"] == "climb.npz"
  assert m["composition"]["env_action_terms"] == terms
  human, osl = m["controllers"]["human"], m["controllers"]["osl"]
  assert human["observation"]["schema"] == "osl_mimic_v1"
  assert human["io"]["input_shape"] == [1, obs_dim]
  assert osl["observation"]["schema"] == "osl_split_sensors_v1"
  assert osl["io"]["input_shape"] == [1, OSL_OBS]
  assert m["reference"]["used_by"] == ["human"]
  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  ids = [
    model.actuator(n).id
    for section in m["controllers"].values()
    for n in section["action"]["actuator_names"]
  ]
  assert sorted(ids) == list(range(model.nu))
  _check_files_and_terrain(bundle, m, clip)


def _rotvec(mat: np.ndarray) -> np.ndarray:
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, mat.reshape(-1))
  quat = -quat if quat[0] < 0 else quat
  norm = np.linalg.norm(quat[1:])
  if norm < 1e-12:
    return 2.0 * quat[1:]
  return quat[1:] / norm * 2.0 * math.atan2(norm, quat[0])


def _full_body_rebuild(bundle: Path, m: dict, section: dict, prefix: str) -> None:
  """A consumer's full-body observation, from model.mjb, reference.npz and the
  manifest alone, reproduces mjlab's at every conformance row."""
  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  obs, ref = section["observation"], m["reference"]
  r = np.load(bundle / "reference.npz")
  parity = np.load(bundle / "parity.npz")
  frames = parity[f"{prefix}_frames"]
  num_frames = int(r["num_frames"])
  steps, stride = ref["lookahead_steps"], ref["lookahead_stride"]
  # Every lookahead entry is seen just before and after its clamp, up to the end.
  last = num_frames - 1
  assert {0, last, last - stride, last - stride + 1} <= set(frames.tolist())
  root = model.joint(m["root_joint"])
  qa, va = int(root.qposadr[0]), int(root.dofadr[0])
  joints = [model.joint(n) for n in m["joint_names"]]
  qids = [int(j.qposadr[0]) for j in joints]
  vids = [int(j.dofadr[0]) for j in joints]
  muscles = np.array([model.actuator(n).id for n in obs["muscle_names"]], dtype=int)
  touch = [model.sensor(n) for n in obs["touch_names"]]
  bodies = [model.body(n).id for n in obs["body_names"]]
  ang_b = np.empty_like(r["root_ang_vel_w"], dtype=np.float64)
  for f in range(num_frames):
    inverse = np.empty(4)
    mujoco.mju_negQuat(inverse, r["root_quat"][f].astype(np.float64))
    mujoco.mju_rotVecQuat(ang_b[f], r["root_ang_vel_w"][f].astype(np.float64), inverse)
  data = mujoco.MjData(model)
  rows = []
  for t, f in enumerate(frames):
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = (
      parity[f"{prefix}_qpos"][t],
      parity[f"{prefix}_qvel"][t],
    )
    data.act[:], data.ctrl[:] = parity[f"{prefix}_act"][t], parity[f"{prefix}_ctrl"][t]
    mujoco.mj_forward(model, data)
    pos = data.xpos[bodies]
    mat = data.xmat[bodies].reshape(-1, 3, 3)
    vel = np.zeros((len(bodies), 6))
    for i, b in enumerate(bodies):
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, vel[i], 0)
    ang, lin = vel[:, :3], vel[:, 3:]
    rel = [mat[0].T @ mat[i] for i in range(1, len(bodies))]
    rpos = pos[1:] - pos[:1]
    rangles = np.stack([_rotvec(x) for x in rel])
    rvel = np.stack(
      [
        np.concatenate([rel[i - 1].T @ ang[i] - ang[0], mat[0] @ (lin[0] - lin[i])])
        for i in range(1, len(bodies))
      ]
    )
    goal = [rpos.ravel(), rangles.ravel(), rvel.ravel(), r["rpos"][f].ravel()]
    for i in range(1, steps):
      g = min(f + i * stride, last)
      goal += [
        r["root_pos"][g] - r["root_pos"][f],
        r["root_lin_vel"][g] - r["root_lin_vel"][f],
        ang_b[g] - ang_b[f],
        r["rpos"][g].ravel(),
      ]
    goal.append([f / num_frames])
    terms = {
      "root_pos": data.qpos[qa + 2 : qa + 7],
      "joint_pos": data.qpos[qids],
      "root_vel": data.qvel[va : va + 6],
      "joint_vel": data.qvel[vids],
      "muscles": np.concatenate(
        [
          data.actuator_length[muscles],
          data.actuator_velocity[muscles],
          data.actuator_force[muscles],
          data.ctrl[muscles],
          data.act[model.actuator_actadr[muscles]],
        ]
      ),
      "touch": np.concatenate(
        [data.sensordata[s.adr[0] : s.adr[0] + s.dim[0]] for s in touch]
      ),
      "goal": np.concatenate([np.asarray(x, dtype=np.float64) for x in goal]),
    }
    assert [terms[n].size for n in obs["terms"]] == [
      obs["term_sizes"][n] for n in obs["terms"]
    ]
    rows.append(np.concatenate([terms[n] for n in obs["terms"]]))
  np.testing.assert_allclose(
    np.stack(rows), parity[f"{prefix}_obs"], rtol=1e-5, atol=1e-4
  )


def test_actor_conformance_matches_a_numpy_rebuild(exported) -> None:
  _, _, out = exported
  _, bundle, m = out["stage1"]
  _full_body_rebuild(bundle, m, m, "actor_conformance")
  _, bundle, m = out["stage2"]
  _full_body_rebuild(bundle, m, m["controllers"]["human"], "human_conformance")


def test_osl_conformance_matches_a_numpy_rebuild(exported) -> None:
  _, _, out = exported
  _, bundle, m = out["stage2"]
  start = m["validation"]["osl_conformance"]["start_frame"]
  assert 0 <= start <= FRAMES - split_stairs_export.CONFORMANCE_STEPS
  check_osl_conformance((bundle, m))


def test_onnx_outputs_on_the_conformance_set(exported) -> None:
  _, _, out = exported
  for stage, onnx, prefix in (
    ("stage1", "actor.onnx", "actor_conformance"),
    ("stage2", "human_actor.onnx", "human_conformance"),
  ):
    _, bundle, _ = out[stage]
    parity = np.load(bundle / "parity.npz")
    session = ort.InferenceSession(
      str(bundle / onnx), providers=["CPUExecutionProvider"]
    )
    means = np.stack(
      [
        np.asarray(session.run(None, {"observation": o[None]})[0])[0]
        for o in parity[f"{prefix}_obs"].astype(np.float32)
      ]
    )
    np.testing.assert_allclose(means, parity[f"{prefix}_means"], atol=1e-6)
    assert len({row.tobytes() for row in parity[f"{prefix}_obs"]}) == len(means)


def _agent(run: Path) -> dict[str, Any]:
  return yaml.safe_load(
    (run / "params/agent.yaml").read_text().replace("!!python/tuple", "")
  )


def test_refusals(exported, tmp_path) -> None:
  _, clip, out = exported
  stage1, _, _ = out["stage1"]
  stage2, _, _ = out["stage2"]
  with pytest.raises(ValueError, match="osl_split_stairs.export"):
    stairs_export.export_bundle(stage2, tmp_path / "a")
  with pytest.raises(ValueError, match="osl_stairs.export"):
    split_stairs_export.export_bundle(stage1, tmp_path / "b")
  # A gait-library run (no terrain), and a stairs run without its terrain copy.
  saved = stairs_export.gl.read_config(stage1.parent / "params/env.yaml")
  library = {**saved, "commands": {"motion": {"library_dir": "/lib"}}}
  with pytest.raises(ValueError, match="osl_gait_library.export"):
    stairs_export.check_stairs_run(stage1, library)
  bare = tmp_path / "bare" / "run"
  shutil.copytree(stage1.parent / "params", bare / "params")
  (bare / "params/terrain.json").unlink()
  with pytest.raises(ValueError, match="no params/terrain.json"):
    stairs_export.check_stairs_run(bare / stage1.name, saved)
  # Stage 2: the checkpoint's own record of the clip and terrain.
  payload = torch.load(stage2, weights_only=False)
  terrain = stage2.parent / "params/terrain.json"
  with pytest.raises(ValueError, match="another terrain"):
    bad = {"infos": {"stairs": {**payload["infos"]["stairs"], "terrain_sha256": "0"}}}
    split_stairs_export.check_recorded_clip(bad, clip, terrain)
  with pytest.raises(ValueError, match="trained on the clip"):
    split_stairs_export.check_recorded_clip(payload, clip.with_name("x.npz"), terrain)
  # A unicycle stage-2 run.
  agent = _agent(stage2.parent)
  agent["validation"]["variant"] = "unicycle"
  with pytest.raises(ValueError, match="mjlab.tasks.osl_split.export"):
    split_stairs_export.stairs_split_run_spec(stage2, agent, saved)
