"""The OSL touch-site setting across configs, checkpoints and play
(``tasks/osl_mimic/touch_sites.py``).

Every OSL task builds ``feet_extended`` unless told otherwise
(``--env.osl-touch-sites``), records its value in ``params/env.yaml`` and in every
checkpoint, and never lets a policy meet other zones: resuming refuses a mismatch,
play builds what the checkpoint trained with and evaluation envs build the
training env's. Anything that records no value trained with ``original``. The
model-level tests are in ``test_osl_constants.py``.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import mujoco
import pytest
import torch
import tyro

import mjlab
import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.asset_zoo.robots.osl_fullbody.osl_constants import (
  DEFAULT_TOUCH_SITES,
  TOUCH_SITES,
  OslEntityCfg,
  touch_sites_of_model,
)
from mjlab.entity import EntityCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.scene import Scene
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.osl_mimic.rl.runner import OslMimicOnPolicyRunner
from mjlab.tasks.osl_mimic.touch_sites import (
  CHECKPOINT_KEY,
  OslEnvCfg,
  env_touch_sites,
  env_yaml_touch_sites,
  set_touch_sites,
  touch_sites_manifest,
  touch_sites_readme_line,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.config.osl import OslTrackingOnPolicyRunner
from mjlab.utils.os import dump_yaml

FLAG = "--env.osl-touch-sites"
OSL_TASKS = [task for task in list_tasks() if "OSL" in task]
MIMIC = "Mjlab-Mimic-OSL-Fullbody-MSK-CLF-Compliant"
GAIT_LIBRARY = "Mjlab-Gait-Library-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
SPLIT = "Mjlab-Split-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
FSM = "Mjlab-FSM-Unicycle-OSL-Fullbody-MSK-CLF-Compliant"
STAIRS = "Mjlab-Stairs-OSL-Fullbody-ROBOT"
TRACKING = "Mjlab-Tracking-Flat-OSL-Fullbody-ROBOT"
# One task per OSL runner class.
RUNNER_TASKS = [MIMIC, GAIT_LIBRARY, SPLIT, FSM, STAIRS, TRACKING]

R_FOOT = "robot/OSL_KA_L1_r_foot_touch"
R_TOES = "robot/OSL_KA_L1_r_toes_touch"
L_FOOT = "robot/foot_touch_l"
L_TOES = "robot/toes_touch_l"
OSL_FOOT = "robot/OSL_KA_L1_osl_foot_assembly"
IDENTITY = [1.0, 0.0, 0.0, 0.0]
L_TOES_QUAT = [0.9393726845901743, 0.0, 0.3428978848663966, 0.0]
FRAME = (
  "site pos/quat/size in the parent body's frame (MuJoCo site_pos, site_quat, "
  "site_size; box half-sizes)"
)


def _contract(
  version: str, edited: list[str], r_foot: dict, l_foot: dict, l_toes: dict
) -> dict:
  """A manifest ``touch_sites`` block: the version's three edited-site boxes, and
  r_toes, which no version edits."""
  return {
    "version": version,
    "edited_from_xml": edited,
    "frame": FRAME,
    "sites": {
      R_FOOT: {"sensor": "robot/r_foot", "body": OSL_FOOT, "type": "box", **r_foot},
      R_TOES: {
        "sensor": "robot/r_toes",
        "body": OSL_FOOT,
        "type": "box",
        "pos": [0.215, -0.04, -0.005],
        "quat": IDENTITY,
        "size": [0.075, 0.03, 0.06],
      },
      L_FOOT: {
        "sensor": "robot/l_foot",
        "body": "robot/calcn_l",
        "type": "box",
        **l_foot,
      },
      L_TOES: {
        "sensor": "robot/l_toes",
        "body": "robot/toes_l",
        "type": "box",
        **l_toes,
      },
    },
  }


# The boxes of the contract: the XML's, and the edited ones.
XML_R_FOOT = {"pos": [0.025, -0.04, 0.0], "quat": IDENTITY, "size": [0.05, 0.03, 0.05]}
HEEL_R_FOOT = {"pos": [0.01, -0.04, 0.0], "quat": IDENTITY, "size": [0.065, 0.03, 0.05]}
XML_L_FOOT = {"pos": [0.09, -0.01, 0.0], "quat": IDENTITY, "size": [0.1, 0.01, 0.055]}
LOW_L_FOOT = {"pos": [0.09, -0.014, 0.0], "quat": IDENTITY, "size": [0.1, 0.014, 0.055]}
XML_L_TOES = {
  "pos": [0.0275, -0.01, 0.0],
  "quat": L_TOES_QUAT,
  "size": [0.04, 0.01, 0.0675],
}
LOW_L_TOES = {
  "pos": [0.0275, -0.014, 0.0],
  "quat": L_TOES_QUAT,
  "size": [0.04, 0.014, 0.0675],
}
# The bundle manifest's ``touch_sites`` per version: the contract shared with
# osl_deploy (logs/handover/touch_fix/brief_mjlab.md for original and heel_extended,
# logs/handover/round6/briefs/brief_mjlab_feet_touch_zones.md for feet_extended),
# value for value.
TOUCH_SITES_CONTRACT = {
  "original": _contract("original", [], XML_R_FOOT, XML_L_FOOT, XML_L_TOES),
  "heel_extended": _contract(
    "heel_extended", [R_FOOT], HEEL_R_FOOT, XML_L_FOOT, XML_L_TOES
  ),
  "feet_extended": _contract(
    "feet_extended", [R_FOOT, L_FOOT, L_TOES], HEEL_R_FOOT, LOW_L_FOOT, LOW_L_TOES
  ),
}
# The feet_extended block exactly as the brief writes it.
FEET_EXTENDED_BRIEF = {
  "version": "feet_extended",
  "edited_from_xml": [
    "robot/OSL_KA_L1_r_foot_touch",
    "robot/foot_touch_l",
    "robot/toes_touch_l",
  ],
  "frame": (
    "site pos/quat/size in the parent body's frame (MuJoCo site_pos, site_quat, "
    "site_size; box half-sizes)"
  ),
  "sites": {
    "robot/OSL_KA_L1_r_foot_touch": {
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


def _cli_env(task: str, *args: str) -> OslEnvCfg:
  """The env config train.py's command line builds for ``task``."""
  cfg = tyro.cli(
    TrainConfig,
    args=list(args),
    default=TrainConfig.from_task(task),
    config=mjlab.TYRO_FLAGS,
  )
  assert isinstance(cfg.env, OslEnvCfg)
  return cfg.env


def _robot(cfg: Any) -> OslEntityCfg:
  robot = cfg.scene.entities["robot"]
  assert isinstance(robot, OslEntityCfg)
  return robot


##
# Configs.
##


def test_every_osl_task_builds_the_extended_feet_by_default() -> None:
  assert DEFAULT_TOUCH_SITES == "feet_extended"
  assert len(OSL_TASKS) > 30
  for task in OSL_TASKS:
    for play in (False, True):
      cfg = load_env_cfg(task, play=play)
      assert isinstance(cfg, OslEnvCfg), task
      assert cfg.osl_touch_sites == _robot(cfg).touch_sites == "feet_extended", task


def test_env_yaml_records_the_touch_sites(tmp_path: Path) -> None:
  """train.py dumps the env config as plain data (it cannot be reloaded as a
  config), and tools read the value back from its top-level line."""
  for task in OSL_TASKS:
    path = tmp_path / task / "params" / "env.yaml"
    dump_yaml(path, asdict(load_env_cfg(task)))
    lines = re.findall(r"^osl_touch_sites:.*$", path.read_text(), re.M)
    assert lines == ["osl_touch_sites: feet_extended"], task
    assert env_yaml_touch_sites(path) == "feet_extended", task
  for version in ("original", "heel_extended"):
    overridden = tmp_path / f"{version}.yaml"
    dump_yaml(overridden, asdict(_cli_env(GAIT_LIBRARY, FLAG, version)))
    assert env_yaml_touch_sites(overridden) == version
  # A run from before the setting records none. The robot entity's nested
  # touch_sites is no record of the run's.
  legacy = asdict(load_env_cfg(GAIT_LIBRARY))
  del legacy["osl_touch_sites"]
  dump_yaml(tmp_path / "legacy.yaml", legacy)
  assert "touch_sites: feet_extended" in (tmp_path / "legacy.yaml").read_text()
  assert env_yaml_touch_sites(tmp_path / "legacy.yaml") is None
  assert env_yaml_touch_sites(tmp_path / "missing.yaml") is None


@pytest.mark.parametrize("task", [GAIT_LIBRARY, STAIRS, TRACKING])
def test_cli_sets_the_touch_sites(
  task: str, capsys: pytest.CaptureFixture[str]
) -> None:
  """tyro rebuilds the env config after the overrides, so the flag reaches the robot
  entity. The entity's own field is no flag: the env's is the one runs record."""
  for version in TOUCH_SITES:
    cfg = _cli_env(task, FLAG, version)
    assert type(cfg) is type(load_env_cfg(task))  # The stairs subclass survives.
    assert cfg.osl_touch_sites == _robot(cfg).touch_sites == version
  assert _cli_env(task).osl_touch_sites == "feet_extended"
  with pytest.raises(SystemExit) as error:
    _cli_env(task, "--env.scene.entities.robot.touch-sites", "original")
  assert error.value.code == 2
  assert "Unrecognized" in capsys.readouterr().err
  with pytest.raises(SystemExit):
    _cli_env(task, FLAG, "heel")


@pytest.fixture(scope="module")
def scene_models() -> dict[tuple[str, str], mujoco.MjModel]:
  """The scenes of a stage-2 task (a spec_fn chain adds the OSL sensors) and of a
  tracking task (its own spec_fn), compiled from train.py's command line: the
  default (feet_extended), and ``--env.osl-touch-sites`` with the other two."""
  models = {}
  for task in (SPLIT, TRACKING):
    for version in TOUCH_SITES:
      args = () if version == DEFAULT_TOUCH_SITES else (FLAG, version)
      models[task, version] = Scene(_cli_env(task, *args).scene, "cpu").compile()
  return models


def test_scene_builds_the_configured_touch_sites(
  scene_models: dict[tuple[str, str], mujoco.MjModel],
) -> None:
  for (task, version), model in scene_models.items():
    assert touch_sites_of_model(model, "robot/") == version, task


def test_manifest_follows_the_contract(
  scene_models: dict[tuple[str, str], mujoco.MjModel],
) -> None:
  """A bundle's ``touch_sites``, read from the compiled model, is the contract
  osl_deploy reads: all four sites, full scene names, plain JSON."""
  assert TOUCH_SITES_CONTRACT["feet_extended"] == FEET_EXTENDED_BRIEF
  assert set(TOUCH_SITES_CONTRACT) == set(TOUCH_SITES)
  edges = {
    "original": ("-0.025 to 0.075", "-0.020"),
    "heel_extended": ("-0.055 to 0.075", "-0.020"),
    "feet_extended": ("-0.055 to 0.075", "-0.028"),
  }
  for (task, version), model in scene_models.items():
    manifest = touch_sites_manifest(model)
    assert json.loads(json.dumps(manifest)) == manifest, task
    expected = json.loads(json.dumps(TOUCH_SITES_CONTRACT[version]))
    for site in expected["sites"].values():
      for key in ("pos", "quat", "size"):
        site[key] = pytest.approx(site[key], rel=0, abs=1e-12)
    assert manifest == expected, (task, version)
    # edited_from_xml lists exactly the sites whose box is not the XML's.
    xml = TOUCH_SITES_CONTRACT["original"]["sites"]
    assert manifest["edited_from_xml"] == [
      name for name, site in manifest["sites"].items() if site != xml[name]
    ]
    line = touch_sites_readme_line(manifest)
    osl_x, sole = edges[version]
    assert f"**{version}**" in line and f"OSL foot x {osl_x} m" in line
    assert f"intact foot zones down to y {sole} m" in line


##
# env_touch_sites and set_touch_sites.
##


def test_touch_sites_need_an_osl_config() -> None:
  """A config without the field is no OSL task's, and an OSL config whose robot is a
  plain EntityCfg would never build the setting."""
  g1 = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  with pytest.raises(TypeError, match="osl_touch_sites"):
    env_touch_sites(SimpleNamespace(cfg=g1))
  with pytest.raises(TypeError, match="Not an OSL env config"):
    set_touch_sites(g1, "original")
  cfg = load_env_cfg(TRACKING)
  cfg.scene.entities["robot"] = EntityCfg(spec_fn=_robot(cfg).spec_fn)
  with pytest.raises(TypeError, match="OslEntityCfg"):
    set_touch_sites(cfg, "original")
  unknown = cast(Any, "heel")
  with pytest.raises(ValueError, match="expected one of"):
    set_touch_sites(load_env_cfg(TRACKING), unknown)
  with pytest.raises(ValueError, match="expected one of"):
    env_touch_sites(SimpleNamespace(cfg=SimpleNamespace(osl_touch_sites=unknown)))


def test_env_touch_sites_checks_the_compiled_model(
  scene_models: dict[tuple[str, str], mujoco.MjModel],
) -> None:
  """The config's value, checked against the env's model when that has the OSL
  foot's r_foot site."""
  cfg = load_env_cfg(SPLIT)
  assert isinstance(cfg, OslEnvCfg)
  sim = SimpleNamespace(mj_model=scene_models[SPLIT, "original"])
  env = SimpleNamespace(cfg=cfg, sim=sim)
  for configured in TOUCH_SITES:
    set_touch_sites(cfg, configured)
    assert cfg.osl_touch_sites == _robot(cfg).touch_sites == configured
    for built in TOUCH_SITES:
      sim.mj_model = scene_models[SPLIT, built]
      if built == configured:
        assert env_touch_sites(env) == configured
      else:
        with pytest.raises(RuntimeError, match=f"built with the '{built}' zones"):
          env_touch_sites(env)
  sim.mj_model = mujoco.MjModel.from_xml_string("<mujoco/>")
  assert env_touch_sites(env) == "feet_extended"


##
# Checkpoints: save, resume.
##

RUNNERS = [OslMimicOnPolicyRunner, OslTrackingOnPolicyRunner]


def _runner(cls: type, touch_sites: str) -> Any:
  """A runner over a fake env that simulates ``touch_sites``, with what save and
  load read."""
  runner = object.__new__(cls)
  env = SimpleNamespace(
    cfg=SimpleNamespace(osl_touch_sites=touch_sites), common_step_counter=24
  )
  runner.env = SimpleNamespace(unwrapped=env)
  runner.alg = SimpleNamespace(save=lambda: {"actor_state_dict": {"w": 1.0}})
  runner.current_learning_iteration = 7
  runner.cfg = {"upload_model": True}
  runner.logger = SimpleNamespace(save_model=mock.Mock())
  return runner


@pytest.fixture
def parent_load(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
  """mjlab's own load, stubbed: the calls that got past the touch-site check."""
  calls: list[tuple] = []

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    calls.append((self, path, load_cfg, strict, map_location))
    return {"loaded": path}

  monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
  return calls


@pytest.mark.parametrize("cls", RUNNERS, ids=lambda cls: cls.__name__)
def test_resume_refuses_a_legacy_checkpoint_on_the_new_default(
  cls: type, tmp_path: Path, parent_load: list[tuple]
) -> None:
  """A checkpoint saved before the setting (by mjlab's own save, as every OSL
  runner's was) trained with the XML's zone: an env on the new default refuses it
  and names the flag; relaunched with the flag, it loads."""
  legacy = str(tmp_path / "model_3050.pt")
  MjlabOnPolicyRunner.save(_runner(cls, "original"), legacy)
  assert CHECKPOINT_KEY not in torch.load(legacy, weights_only=False)
  for current in ("feet_extended", "heel_extended"):
    with pytest.raises(ValueError, match=re.escape(f"{FLAG} original")) as error:
      _runner(cls, current).load(legacy)
    assert "records none" in str(error.value)
  assert parent_load == []
  runner = _runner(cls, "original")
  assert runner.load(legacy, map_location="cpu") == {"loaded": legacy}
  assert parent_load == [(runner, legacy, None, True, "cpu")]


@pytest.mark.parametrize("cls", RUNNERS, ids=lambda cls: cls.__name__)
def test_resume_refuses_a_heel_extended_checkpoint_on_the_new_default(
  cls: type, tmp_path: Path, parent_load: list[tuple]
) -> None:
  """A checkpoint from the first fix (the extended heel, the XML's left zones) is
  refused under the new default, naming the flag that rebuilds its zones, and
  loads with it."""
  path = str(tmp_path / "model_500.pt")
  _runner(cls, "heel_extended").save(path)
  with pytest.raises(ValueError, match=re.escape(f"{FLAG} heel_extended")) as error:
    _runner(cls, DEFAULT_TOUCH_SITES).load(path)
  assert "builds 'feet_extended'" in str(error.value)
  assert parent_load == []
  runner = _runner(cls, "heel_extended")
  assert runner.load(path) == {"loaded": path}
  assert parent_load == [(runner, path, None, True, None)]


@pytest.mark.parametrize("touch_sites", TOUCH_SITES)
@pytest.mark.parametrize("cls", RUNNERS, ids=lambda cls: cls.__name__)
def test_checkpoints_record_their_touch_sites(
  cls: type, touch_sites: str, tmp_path: Path, parent_load: list[tuple]
) -> None:
  """Every checkpoint is mjlab's own plus the version the env simulates. It loads
  into an env with the same zones, and one with the other refuses it, naming the
  flag that builds its zones."""
  assert CHECKPOINT_KEY == "osl_touch_sites"
  path, native = str(tmp_path / "model_7.pt"), str(tmp_path / "native.pt")
  runner = _runner(cls, touch_sites)
  runner.save(path, {"note": 1})
  runner.logger.save_model.assert_called_once_with(path, 7)
  MjlabOnPolicyRunner.save(_runner(cls, touch_sites), native, {"note": 1})
  saved = torch.load(path, weights_only=False)
  assert saved == {
    **torch.load(native, weights_only=False),
    CHECKPOINT_KEY: touch_sites,
  }
  assert runner.load(path) == {"loaded": path}
  for other in TOUCH_SITES:
    if other != touch_sites:
      with pytest.raises(ValueError, match=re.escape(f"{FLAG} {touch_sites}")):
        _runner(cls, other).load(path)
  assert parent_load == [(runner, path, None, True, None)]


##
# Play and evaluation envs.
##


def test_every_osl_runner_lets_play_build_the_checkpoints_touch_sites() -> None:
  """play.py asks the task's runner class to configure the env for the checkpoint
  before building it (play takes no --env overrides)."""
  classes = {load_runner_cls(task) for task in OSL_TASKS}
  assert classes == {load_runner_cls(task) for task in RUNNER_TASKS}
  for cls in classes:
    assert callable(getattr(cls, "configure_env_for_checkpoint", None)), cls


@pytest.mark.parametrize("task", RUNNER_TASKS)
def test_play_builds_what_the_checkpoint_trained_with(
  task: str, tmp_path: Path
) -> None:
  """The checkpoint's key, else ``original`` (which its run's env.yaml may say); a
  checkpoint that its run's env.yaml contradicts is refused."""
  configure = cast(Any, load_runner_cls(task)).configure_env_for_checkpoint

  def checkpoint(name: str, key: str | None, env_yaml: str | None) -> Path:
    """A checkpoint recording ``key`` in a run whose env.yaml records
    ``env_yaml``; None records nothing there, as before the setting."""
    run = tmp_path / name
    payload: dict[str, Any] = {"actor_state_dict": {}, "iter": 10, "infos": None}
    if key is not None:
      payload[CHECKPOINT_KEY] = key
    saved = asdict(load_env_cfg(task))
    if env_yaml is None:
      del saved["osl_touch_sites"]
    else:
      saved["osl_touch_sites"] = env_yaml
    dump_yaml(run / "params" / "env.yaml", saved)
    torch.save(payload, run / "model_10.pt")
    return run / "model_10.pt"

  def played(path: Path, start: str) -> str:
    cfg = load_env_cfg(task, play=True)
    assert isinstance(cfg, OslEnvCfg)
    set_touch_sites(cfg, cast(Any, start))
    configure(cfg, path)
    assert cfg.osl_touch_sites == _robot(cfg).touch_sites
    return cfg.osl_touch_sites

  # Saved before the setting, from a run before it.
  assert played(checkpoint("legacy", None, None), "feet_extended") == "original"
  # A bare checkpoint, with no run directory around it.
  bare = tmp_path / "bare.pt"
  torch.save({"actor_state_dict": {}, "iter": 10, "infos": None}, bare)
  assert played(bare, "feet_extended") == "original"
  assert played(checkpoint("new", "heel_extended", None), "original") == "heel_extended"
  heel = checkpoint("heel", "heel_extended", "heel_extended")
  assert played(heel, "feet_extended") == "heel_extended"
  feet = checkpoint("feet", "feet_extended", "feet_extended")
  assert played(feet, "heel_extended") == "feet_extended"
  assert played(checkpoint("yaml", None, "original"), "feet_extended") == "original"
  agreeing = checkpoint("agreeing", "original", "original")
  assert played(agreeing, "feet_extended") == "original"
  moved = checkpoint("moved", "heel_extended", "original")
  with pytest.raises(ValueError, match="params/env.yaml records 'original'"):
    configure(load_env_cfg(task, play=True), moved)
  # Every runner records the key since the setting, so a key-less checkpoint in a
  # run that records the extended zones came from elsewhere.
  stray = checkpoint("stray", None, "heel_extended")
  with pytest.raises(ValueError, match="records no OSL touch sites"):
    configure(load_env_cfg(task, play=True), stray)


@pytest.mark.parametrize("task", [MIMIC, GAIT_LIBRARY, SPLIT, FSM, STAIRS])
def test_evaluation_envs_build_the_training_touch_sites(task: str) -> None:
  """Validation and video configs are rebuilt from the task and stress copies the
  training one: all three build the training env's zones, as a resumed run on the
  old zones (``--env.osl-touch-sites original``) needs."""
  train = load_env_cfg(task)
  set_touch_sites(train, "original")
  runner_cls = load_runner_cls(task)
  assert runner_cls is not None
  runner: Any = object.__new__(runner_cls)
  runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=train))
  runner.validation_cfg = dict(asdict(load_rl_cfg(task))["validation"])
  for purpose in ("validation", "video", "stress"):
    cfg = runner._eval_env_cfg(purpose)
    assert isinstance(cfg, OslEnvCfg), purpose
    assert cfg.osl_touch_sites == _robot(cfg).touch_sites == "original", purpose
