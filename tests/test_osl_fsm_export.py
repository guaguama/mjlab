"""Deployment export of stage-1.5 runs: the human actor beside the walking controller
(CPU only).

A four-clip library is generated from the real model by forward kinematics, and a
fake stage-1.5 run directory holds the task's own saved configs, a small human actor
and the controller settings its env's action term records, so ``export_bundle`` runs
end to end on both models: the robot run on the sdk rules without the swing exit,
the MSK run on MyoSuite's. The bundle is checked as osl_deploy reads it: no OSL
network, stage 2's OSL section without one, and a controller conformance that a
consumer reproduces from the manifest and ``model.mjb`` alone.
"""

from __future__ import annotations

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

from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import Model, Socket, get_spec
from mjlab.tasks.osl_fsm import export as fsm_export
from mjlab.tasks.osl_fsm.config import fsm_task_id
from mjlab.tasks.osl_fsm.config.env_cfgs import OslFsmCfg, osl_fsm_env_cfg
from mjlab.tasks.osl_fsm.config.rl_cfg import osl_fsm_runner_cfg
from mjlab.tasks.osl_fsm.fsm import BatchedFsm, rule_set
from mjlab.tasks.osl_fsm.mdp import OslFsmAction
from mjlab.tasks.osl_gait_library import export as gl
from mjlab.tasks.osl_mimic.config.env_cfgs import Reward
from mjlab.tasks.osl_mimic.rl.models import MuscleMimicMLPModel
from mjlab.tasks.osl_split import export as split_export
from mjlab.utils.os import dump_yaml

DT = 0.01
HIDDEN = (16, 16)
OSL_OBS = 600
FRAMES = 50
# (model, socket, reward) -> (human obs width, env action terms, controller).
Case = tuple[Model, Socket, Reward]
CASES: dict[Case, tuple[int, list[list[Any]], OslFsmCfg]] = {
  ("robot", "rigid", "mimic"): (
    635,
    [["servos", 27], ["osl_motors", 2]],
    OslFsmCfg(rules="sdk", swing_load_exit=False),
  ),
  ("msk", "compliant", "clf"): (
    2338,
    [["osl_motors", 2], ["muscles", 339]],
    OslFsmCfg(),
  ),
}
DEPLOY = "python -m osl_deploy.osl_fsm BUNDLE"
FILES = {"model.mjb", "reference.npz", "human_actor.onnx", "parity.npz"}
OSL_SWING = {
  "OSL_KA_L1_osl_knee_angle_r": (0.6, 0.5),
  "OSL_KA_L1_osl_ankle_angle_r": (0.05, 0.15),
}
"""(mean, amplitude) in rad of the OSL joints in a moving clip, once per loop: the
knee from 6 to 63 deg, the ankle from -6 to 11 deg, across the walking controller's
angle thresholds (6, 30 and 50 deg), so the conformance passes through its
states."""


def _clip(model, data, frames, vel, wz, swing=False):
  """``frames`` of the start pose moving at world velocity ``vel`` and yaw rate
  ``wz``, as clipper's mjlab format stores them; with ``swing``, the OSL joints
  swing once per loop (:data:`OSL_SWING`)."""
  base = model.qpos0.copy()
  phase = 2 * math.pi * np.arange(frames) / frames
  rate = 2 * math.pi / (frames * DT)
  joint_vel = np.zeros((frames, model.nq - 7))
  swung = {}
  for name, (mean, amplitude) in OSL_SWING.items() if swing else ():
    column = int(model.joint(name).qposadr[0]) - 7
    swung[column] = mean + amplitude * np.sin(phase)
    joint_vel[:, column] = amplitude * rate * np.cos(phase)
  pos, quat, joints = [], [], []
  for t in range(frames):
    q = base.copy()
    q[:2] = np.asarray(vel[:2]) * t * DT
    yaw = wz * t * DT
    q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    for column, angles in swung.items():
      q[7 + column] = angles[t]
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
    "joint_vel": joint_vel,
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
  v, wz, span = 0.8, 0.6, FRAMES * DT
  # Forward is the root's body -y, which is world -y at zero yaw.
  clips = {
    "idle": ((0, 0, 0), (0.0, 0.0, 0.0), 0.0, _se2(0, 0, 0)),
    "walk_fwd": ((0, -v, 0), (v, 0.0, 0.0), 0.0, _se2(0, -v * span, 0)),
    "walk_bck": ((0, v, 0), (-v, 0.0, 0.0), 0.0, _se2(0, v * span, 0)),
    "walk_turn": ((0, 0, 0), (0.0, 0.0, wz), wz, _se2(0, 0, wz * span)),
  }
  for name, (vel, twist, rate, stride) in clips.items():
    clip = _clip(model, data, FRAMES, vel, rate, swing=name != "idle")
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


def _make_run(root: Path, case: Case, library: Path) -> Path:
  """A stage-1.5 run on the default touch sites with ``CASES``' controller: its
  env.yaml, agent.yaml, and a checkpoint whose controller settings are the ones the
  env's action term resolves (``OslFsmAction.settings``, as the runner records
  them), with a warm-start lineage on the same touch sites."""
  model, socket, reward = case
  obs_dim, terms, controller = CASES[case]
  run = root / f"{model}_{socket}" / "run"
  env_cfg = osl_fsm_env_cfg(model, socket, reward, "train", fsm=controller)
  env_cfg.commands["motion"].library_dir = str(library)  # type: ignore[attr-defined]
  agent = osl_fsm_runner_cfg(model, socket, reward, "exp")
  agent.actor.hidden_dims = HIDDEN
  dump_yaml(run / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(run / "params" / "agent.yaml", asdict(agent))
  saved = gl.read_config(run / "params" / "env.yaml")
  _, _, actions, _ = gl.compile_task(env_cfg, saved)
  term = actions["osl_motors"]
  assert isinstance(term, OslFsmAction)
  torch.manual_seed(0)
  human_dim = sum(int(dim) for name, dim in terms if name != "osl_motors")
  human = _actor("actor", obs_dim, human_dim, dict(agent.actor.distribution_cfg or {}))
  lineage = {
    "model": model,
    "socket": socket,
    "action_terms": terms,
    "stage1_checkpoint": "/stage1/run/model_4.pt",
    "osl_touch_sites": env_cfg.osl_touch_sites,
  }
  payload = {
    "algorithm": "human_only",
    "human_actor_state_dict": human.state_dict(),
    "iter": 7,
    "stage_iteration": 2,
    "osl_touch_sites": env_cfg.osl_touch_sites,
    "infos": {
      "split": {"stage_start": 5, "lineage": lineage},
      "osl_fsm": term.settings(),
    },
  }
  torch.save(payload, run / "model_7.pt")
  return run / "model_7.pt"


@pytest.fixture(scope="module", params=list(CASES), ids=lambda c: "-".join(c))
def exported(request, tmp_path_factory, library):
  checkpoint = _make_run(tmp_path_factory.mktemp("runs"), request.param, library)
  bundle = fsm_export.export_bundle(checkpoint)
  manifest = json.loads((bundle / "manifest.json").read_text())
  return request.param, checkpoint, bundle, manifest


@pytest.fixture(scope="module")
def rebuilt(exported) -> fsm_export.FsmRun:
  """The exported run rebuilt on CPU, as the exporter rebuilds it."""
  return fsm_export.build_run(exported[1])


def _plain(value: Any) -> Any:
  """A section as the manifest stores it: JSON types."""
  return json.loads(json.dumps(split_export._plain(value)))


def test_bundle_contract(exported) -> None:
  """Schema 2 and kind osl_split_dual, as osl_deploy composes controllers; the OSL
  section without onnx and io; no osl_actor.onnx, every checksum verified; and a
  model.mjb with the OSL sensors and the run's touch sites."""
  case, checkpoint, bundle, m = exported
  obs_dim, terms, _ = CASES[case]
  assert bundle == checkpoint.parent / "deploy" / "model_7"
  assert (m["schema_version"], m["kind"]) == (2, "osl_split_dual")
  assert m["run"]["task"] == fsm_task_id(*case)
  assert (m["run"]["stage"], m["run"]["osl_controller"]) == (1.5, "osl_fsm")
  assert m["composition"]["controllers"] == ["human", "osl"]
  assert m["composition"]["env_action_terms"] == terms
  human, osl = m["controllers"]["human"], m["controllers"]["osl"]
  assert human["onnx"] == "human_actor.onnx"
  assert human["io"]["input_shape"] == [1, obs_dim]
  assert human["observation"]["schema"] == "osl_mimic_gait_v1"
  assert "onnx" not in osl and "io" not in osl
  assert set(osl) == {"observation", "action", "training", "device", "memory", "fsm"}
  obs = osl["observation"]
  assert (obs["schema"], obs["size"]) == ("osl_split_sensors_v1", OSL_OBS)
  assert obs["terms"] == ["actpack", "loadcell", "last_action"]
  assert obs["normalization"].startswith("none")
  assert obs["latency"]["buses"] == {}  # The controller read its sensors at once.
  assert osl["action"]["std"] == [0.0, 0.0]
  assert osl["action"]["default_inference"] == "mean"
  assert [b["name"] for b in osl["action"]["blocks"]] == ["osl_motors"]
  assert osl["training"]["actuator_delay"]["osl_motors"]["seconds"] == [0.0, 0.0]
  assert osl["memory"]["kind"] == "state_machine"
  assert osl["fsm"]["format"] == "mjlab.osl_fsm/1"
  recorded = torch.load(checkpoint, weights_only=False)["infos"]["osl_fsm"]
  assert set(recorded) == {*fsm_export.FSM_SETTINGS, "controller"}
  assert {k: osl["fsm"][k] for k in fsm_export.FSM_SETTINGS} == _plain(
    {k: recorded[k] for k in fsm_export.FSM_SETTINGS}
  )
  assert osl["fsm"]["matches_osl_deploy_constants"] is True

  assert set(m["files"]) == FILES
  assert not (bundle / "osl_actor.onnx").exists()
  for name, digest in m["files"].items():
    assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest

  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  ids = [
    model.actuator(n).id for s in (human, osl) for n in s["action"]["actuator_names"]
  ]
  assert sorted(ids) == list(range(model.nu))
  for sensor in obs["sensors"]["sensors"]:
    assert model.sensor(sensor["scene_name"]).dim[0] == sensor["dim"]
  assert m["touch_sites"]["version"] == m["lineage"]["osl_touch_sites"]
  for name, entry in m["touch_sites"]["sites"].items():
    for key in ("pos", "quat", "size"):
      assert getattr(model.site(name), key).tolist() == entry[key], (name, key)


def test_osl_section_is_stage2s_without_its_network(exported, rebuilt) -> None:
  """controllers.osl is what stage 2's exporter writes for an OSL actor on the same
  task, without onnx and io: the observation contract, the training section and
  the torque map unchanged; no normalizer and a std of 0; the state machine in
  place of the observation history, and the controller block."""
  m = exported[3]
  actor = _actor("osl", OSL_OBS, 2, {"class_name": "GaussianDistribution"})
  section = split_export.osl_controller(rebuilt.task, actor, OSL_OBS, rebuilt.lineage)
  stage2 = _plain(section)
  ours = m["controllers"]["osl"]
  assert set(stage2) - set(ours) == {"onnx", "io"}
  assert set(ours) - set(stage2) == {"fsm"}
  assert ours["training"] == stage2["training"]
  assert ours["device"] == stage2["device"]

  def without(section: dict, *keys: str) -> dict:
    return {k: v for k, v in section.items() if k not in keys}

  assert without(ours["observation"], "normalization") == without(
    stage2["observation"], "normalization"
  )
  assert without(ours["action"], "std") == without(stage2["action"], "std")
  assert stage2["memory"]["kind"] == "observation_history"


def test_human_onnx_parity_and_metadata(exported) -> None:
  _, _, bundle, m = exported
  section = m["controllers"]["human"]
  parity = np.load(bundle / "parity.npz")
  session = ort.InferenceSession(
    str(bundle / section["onnx"]), providers=["CPUExecutionProvider"]
  )
  io = section["io"]
  assert session.get_inputs()[0].name == io["input"]
  assert session.get_outputs()[0].shape == io["output_shape"]
  out = np.stack(
    [
      np.asarray(session.run(None, {io["input"]: o[None]})[0])[0]
      for o in parity["human_observations"]
    ]
  )
  np.testing.assert_allclose(out, parity["human_means"], atol=1e-6)
  meta = json.loads(session.get_modelmeta().custom_metadata_map["osl_policy"])
  assert meta["controller"] == "human" and meta["kind"] == "osl_split_dual"
  assert meta["run"]["stage"] == 1.5


def test_fsm_conformance_reproduces_from_the_bundle(exported) -> None:
  """A consumer reproduces the controller conformance from the manifest and
  model.mjb alone, the way osl_deploy's state machine reads them: the signals'
  newest frames of osl_conformance_obs_lag0 (whose newest frames are the undelayed
  ones), the load as the negated tared force, the body weight and motor gains from
  the model, the clip, offset and scale from the action block."""
  _, _, bundle, m = exported
  osl = m["controllers"]["osl"]
  fsm, action = osl["fsm"], osl["action"]
  parity = np.load(bundle / "parity.npz")
  rows = parity["osl_conformance_obs_lag0"]
  steps = m["validation"]["osl_fsm_conformance"]["steps"]
  assert rows.shape[0] == steps == fsm_export.CONFORMANCE_STEPS > FRAMES
  assert sorted(m["validation"]["osl_fsm_conformance"]["keys"]) == [
    "osl_fsm_conformance_action",
    "osl_fsm_conformance_state",
    "osl_fsm_conformance_torque",
  ]
  newest = []
  for term in osl["observation"]["term_layout"]:
    start = term["offset"] + (term["history_length"] - 1) * term["frame_dim"]
    newest += list(range(start, start + term["frame_dim"]))
  np.testing.assert_array_equal(rows[:, newest], parity["osl_conformance_frames"])

  # The signals' newest columns from the manifest's term layout alone, as a consumer
  # finds them (osl_deploy's newest_channels), not through the exporter's helper.
  index = {}
  for term in osl["observation"]["term_layout"]:
    start = term["offset"] + (term["history_length"] - 1) * term["frame_dim"]
    index.update({name: start + c for c, name in enumerate(term["channels"])})
  columns = [index[channel] for channel in fsm["signals"].values()]
  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  options = fsm["deploy_options"]
  mass = options.get("body_mass", float(model.body_mass.sum()))
  weight = mass * float(np.linalg.norm(model.opt.gravity))
  assert weight == fsm["body_weight_n"]
  machine = BatchedFsm(
    rule_set(options["rules"], options.get("swing_load_exit")), weight, 1
  )
  motors = [model.actuator(name).id for name in action["actuator_names"]]
  nm_per_ctrl = model.actuator_gainprm[motors, 0] * model.actuator_gear[motors, 0]
  place = [0 if "knee" in target else 1 for target in action["blocks"][0]["targets"]]
  states, actions = [], []
  for row in torch.as_tensor(rows, dtype=torch.float32):
    knee, knee_velocity, ankle, ankle_velocity, force = row[columns].reshape(-1, 1)
    states.append(int(machine.update(-force, knee, knee_velocity, ankle)[0]))
    torque = machine.torque(
      torch.stack((knee, ankle), 1), torch.stack((knee_velocity, ankle_velocity), 1)
    )[0].double()
    ctrl = np.clip(
      torque.numpy()[place] / nm_per_ctrl, action["clip_low"], action["clip_high"]
    )
    actions.append((ctrl - np.asarray(action["offset"])) / np.asarray(action["scale"]))
  assert states == parity["osl_fsm_conformance_state"].tolist()
  assert set(states) == {0, 1, 2, 3}  # The swinging clip cycles every state.
  np.testing.assert_allclose(
    actions, parity["osl_fsm_conformance_action"], rtol=1e-5, atol=1e-6
  )


def test_looped_clip_continues_through_the_stride(rebuilt) -> None:
  """Past its last frame the conformance clip starts over where the stride
  transform puts it: the first loop is the clip itself, the next one continues the
  root's motion and repeats the joints."""
  command = rebuilt.task.command
  clip = command.clip_names.index("walk_turn")
  view = fsm_export.looped_clip(command, clip, 2 * FRAMES + 3)
  data = command.data
  for key in fsm_export.CLIP_FIELDS:
    torch.testing.assert_close(
      view.data[key][0, :FRAMES], data[key][clip, :FRAMES].double(), rtol=0, atol=0
    )
  joints = view.data["joint_pos"][0]
  torch.testing.assert_close(joints[FRAMES : 2 * FRAMES], joints[:FRAMES])
  pos, vel = view.data["root_pos"][0], view.data["root_lin_vel"][0]
  expected = pos[FRAMES - 1] + vel[FRAMES - 1] * DT
  assert torch.linalg.norm(pos[FRAMES, :2] - expected[:2]) < 0.01
  w = view.data["root_quat"][0, FRAMES]
  yaw = math.atan2(2 * (w[0] * w[3] + w[1] * w[2]), 1 - 2 * (w[2] ** 2 + w[3] ** 2))
  assert abs(yaw - 0.6 * FRAMES * DT) < 0.02  # walk_turn's yaw after one loop.


def test_deploy_command_names_the_controller(exported) -> None:
  """The deploy command always names the rules (osl_deploy defaults to sdk), the
  sdk rules' swing exit explicitly either way, and the body mass when the run set
  one; the README gives it first."""
  case, _, bundle, m = exported
  fsm = m["controllers"]["osl"]["fsm"]
  expected = {
    "myosuite": f"{DEPLOY} --rules myosuite",
    "sdk": f"{DEPLOY} --rules sdk --no-swing-load-exit",
  }
  assert fsm["deploy_command"] == expected[CASES[case][2].rules]
  readme = (bundle / "README.md").read_text()
  assert fsm["deploy_command"].replace("BUNDLE", "<this folder>") in readme
  text = " ".join(readme.split())
  assert "defaults to `--rules sdk`, so the `--rules` flag is required" in text

  model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
  sdk = {**fsm, "rules": "sdk", "swing_load_exit": True}
  assert fsm_export.deploy_options(sdk, model) == {
    "rules": "sdk",
    "swing_load_exit": True,
  }
  assert fsm_export.deploy_command(fsm_export.deploy_options(sdk, model)) == (
    f"{DEPLOY} --rules sdk --swing-load-exit"
  )
  heavy = {**fsm, "rules": "myosuite", "swing_load_exit": False, "body_mass": 90.5}
  heavy["body_mass_kg"] = 90.5
  assert fsm_export.deploy_command(fsm_export.deploy_options(heavy, model)) == (
    f"{DEPLOY} --rules myosuite --body-mass 90.5"
  )


def test_refuses_other_runs_and_checkpoints(exported, tmp_path) -> None:
  """Stage-2 and stage-1 runs and checkpoints are refused naming their exporters,
  as are a human-only checkpoint without controller settings and one whose
  controller is not its run's."""
  _, checkpoint, _, _ = exported
  run = checkpoint.parent
  agent = yaml.safe_load(
    (run / "params/agent.yaml").read_text().replace("!!python/tuple", "")
  )
  agent["algorithm"]["class_name"] = split_export.ALGORITHM_CLASS
  with pytest.raises(ValueError, match="osl_split.export"):
    fsm_export.fsm_run_spec(checkpoint, agent)
  agent["algorithm"]["class_name"] = "mjlab.tasks.osl_mimic.rl.ppo:MuscleMimicPPO"
  with pytest.raises(ValueError, match="osl_gait_library.export"):
    fsm_export.fsm_run_spec(checkpoint, agent)

  payload = torch.load(checkpoint, weights_only=False)
  human = payload["human_actor_state_dict"]
  stage2, stage1 = tmp_path / "stage2.pt", tmp_path / "stage1.pt"
  torch.save({"human_actor_state_dict": human, "osl_actor_state_dict": {}}, stage2)
  torch.save({"actor_state_dict": human}, stage1)
  with pytest.raises(ValueError, match="stage-2 checkpoint.*osl_split.export"):
    fsm_export.load_payload(stage2)
  with pytest.raises(ValueError, match="stage-1 checkpoint.*osl_gait_library"):
    fsm_export.load_payload(stage1)

  infos = payload["infos"]
  unrecorded, moved = (checkpoint.with_name(f"model_{i}.pt") for i in (8, 9))
  torch.save({**payload, "infos": {"split": infos["split"]}}, unrecorded)
  with pytest.raises(ValueError, match="records no OSL controller settings"):
    fsm_export.export_bundle(unrecorded)
  rules = "myosuite" if infos["osl_fsm"]["rules"] == "sdk" else "sdk"
  other = {**infos["osl_fsm"], "rules": rules, "swing_load_exit": rules == "sdk"}
  torch.save({**payload, "infos": {**infos, "osl_fsm": other}}, moved)
  with pytest.raises(ValueError, match="another OSL controller than its run's"):
    fsm_export.export_bundle(moved)


def test_refuses_a_controller_this_checkout_does_not_build(exported) -> None:
  """A checkpoint whose recorded controller is not the one this checkout rebuilds
  (here its gains changed) cannot be described by the bundle, and is refused."""
  _, checkpoint, _, _ = exported
  payload = torch.load(checkpoint, weights_only=False)
  settings = payload["infos"]["osl_fsm"]
  impedance = json.loads(json.dumps(settings["impedance"]))
  impedance["e_stance"]["knee"]["stiffness_nm_per_rad"] *= 2.0
  payload["infos"]["osl_fsm"] = {**settings, "impedance": impedance}
  tuned = checkpoint.with_name("model_10.pt")
  torch.save(payload, tuned)
  with pytest.raises(
    ValueError, match=r"differs from the one the run recorded in \['impedance'\]"
  ):
    fsm_export.export_bundle(tuned)


def test_refuses_a_lineage_on_other_touch_sites(exported) -> None:
  _, checkpoint, _, _ = exported
  payload = torch.load(checkpoint, weights_only=False)
  payload["infos"]["split"]["lineage"]["osl_touch_sites"] = "original"
  other = checkpoint.with_name("model_11.pt")
  torch.save(payload, other)
  with pytest.raises(ValueError, match="but its warm start's stage 1 with 'original'"):
    fsm_export.export_bundle(other)
