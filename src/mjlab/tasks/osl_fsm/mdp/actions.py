"""The OSL motors under the classical walking controller: stage 1.5's plant.

Stage 1.5 is stage 2 with the learned OSL actor replaced by the controller of
:mod:`~mjlab.tasks.osl_fsm.fsm`, running on the device from its own sensors.
:class:`OslFsmAction` takes the stage-1 ``osl_motors`` slot unchanged (the same
JointEffortAction fields: scale 2.88, offset 0, clip +-2.88 ctrl, two rows), so
everything that slices the action by term (``ActionLayout``, the split reward's
owner penalties, the warm start's layout check, the exporter's action metadata)
works as in stage 2. The policy's two OSL rows are ignored; the learner writes zeros
there.

The plant is osl_deploy's ``osl_fsm``, step for step:

* **one reading per control step, the boundary's.** ``process_actions`` runs once
  per env step, right after the previous step's single ``sim.forward()`` (or
  reset's), and nothing writes the simulation state in between. So the sensors it
  reads are the instant the observations saw: the newest frame osl_deploy's
  ``FsmPolicy`` reads. Clean and immediate: no lag, bias or noise.
* **at most one transition, then the new state's torque, held.** ``apply_actions``
  writes the same command on every physics substep, a 100 Hz zero-order hold as in
  osl_deploy. The OSL motors' actuator delay must be zero for that hold to be the
  command's (``osl_fsm_env_cfg`` sets it).
* **the nominal motor.** ``ctrl = clip(torque / (gain * gear), clip)`` with the gain
  and gear of the host model ``env.sim.mj_model``, which the domain randomization
  never writes (it writes the per-env warp model): a motor-strength draw is the
  plant's deviation, which a device does not know.
* **the tared axial load.** ``load = -(f_y - tare)``: the ankle cell's axial force
  minus its hanging-leg reading (compression reads negative ``y``), as the device's
  tare and osl_deploy's observation remove it.
* **the nominal body weight.** The robot entity's mass on the host model times
  ``|g|``: osl_deploy's bundle model sums to the same 82.289 kg.

Every osl_deploy episode starts in early stance. Training episodes start at random
clip frames, so by default (``initial_state="pose"``) the state is guessed from the
start pose on the first control step after a reset
(:meth:`~mjlab.tasks.osl_fsm.fsm.BatchedFsm.guess`), before that step's regular
update. A term's reset runs before the command writes the start frame and before
the forward that computes its sensors, so the guess waits for the next
``process_actions``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

import mujoco
import numpy as np
import torch
import tyro

from mjlab.envs.mdp.actions import JointEffortAction, JointEffortActionCfg
from mjlab.tasks.osl_fsm.fsm import (
  E_STANCE,
  JOINTS,
  STATES,
  BatchedFsm,
  RuleName,
  impedance_settings,
  rule_set,
  threshold_settings,
)
from mjlab.tasks.osl_split.config.sensors import (
  ANKLE_FORCE,
  ANKLE_POS,
  ANKLE_VEL,
  KNEE_POS,
  KNEE_VEL,
  scene_sensor_name,
  static_load_cell_tare,
)
from mjlab.tasks.osl_split.rl.layout import OSL_ACTION_TERM

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

InitialState = Literal["pose", "e_stance"]
Controller = Literal["fsm", "policy"]

READINGS: tuple[str, ...] = (
  "knee",
  "knee_velocity",
  "ankle",
  "ankle_velocity",
  "force",
)
"""The columns of :attr:`OslFsmAction.readings`: osl_deploy's ``SIGNALS`` keys."""
SIGNALS: dict[str, str] = {
  "knee": "osl_knee_pos",
  "knee_velocity": "osl_knee_vel",
  "ankle": "osl_ankle_pos",
  "ankle_velocity": "osl_ankle_vel",
  "force": "osl_ankle_load_force_y",
}
"""Each reading's channel in the ``osl`` observation, whose newest frame osl_deploy's
controller reads (``force`` is tared there, as here)."""
_SENSORS: dict[str, tuple[str, int]] = {
  "knee": (KNEE_POS, 0),
  "knee_velocity": (KNEE_VEL, 0),
  "ankle": (ANKLE_POS, 0),
  "ankle_velocity": (ANKLE_VEL, 0),
  "force": (ANKLE_FORCE, 1),
}
"""Each reading's entity-level sensor and component (the cell's axial ``y``)."""
FSM_DIAGNOSTICS: tuple[str, ...] = (
  "fsm_e_stance",
  "fsm_l_stance",
  "fsm_e_swing",
  "fsm_l_swing",
  "fsm_cycle",
  "fsm_knee_torque",
  "fsm_ankle_torque",
  "fsm_load_bw",
)
"""The per-step diagnostics in :attr:`OslFsmAction.components`: 1 in the state the
update left (one key per state), 1 on entering early stance from another state
(osl_deploy's ``cycles``), each motor's commanded torque magnitude after the ctrl
clip (N m, nominal gain), and the load in body weights."""
UPDATE = "once per control step, held over decimation"


@dataclass(kw_only=True)
class OslFsmActionCfg(JointEffortActionCfg):
  """``osl_motors`` under the walking controller (``controller="fsm"``) or, as a
  plain JointEffortAction, under the policy's rows (``controller="policy"``).

  The four controller settings are hidden from the command line: the env config's
  ``osl_fsm`` (``--env.osl-fsm.*``) is their one knob, copied here whenever the env
  config is built (``config.env_cfgs.sync_osl_fsm``), so a flag here would be
  overwritten silently.
  """

  rules: tyro.conf.Suppress[RuleName] = "myosuite"
  """``"myosuite"`` (a state leaves when any condition holds) or ``"sdk"`` (all)."""
  swing_load_exit: tyro.conf.Suppress[bool | None] = None
  """The sdk rules' load-based swing exit; None is their default, on."""
  body_mass: tyro.conf.Suppress[float | None] = None
  """kg, scaling the load thresholds; None is the robot's nominal mass."""
  initial_state: tyro.conf.Suppress[InitialState] = "pose"
  """``"pose"``: guessed from the start pose; ``"e_stance"``: osl_deploy's."""
  controller: Controller = "fsm"

  def build(self, env: ManagerBasedRlEnv) -> OslFsmAction:
    return OslFsmAction(self, env)


def entity_mass(model: mujoco.MjModel, prefix: str) -> float:
  """The summed mass (kg) of the bodies whose names start with ``prefix``: one
  entity of an mjlab scene, which attaches each under ``"<name>/"``. On the OSL
  scenes it equals the whole model's (the world and the terrain are massless), the
  sum osl_deploy takes over a bundle's model; a terrain with massive bodies would
  not move it."""
  ids = [i for i in range(model.nbody) if model.body(i).name.startswith(prefix)]
  if not ids:
    raise ValueError(f"The model has no bodies under the prefix {prefix!r}.")
  return float(model.body_mass[ids].sum())


def nominal_motor_gains(
  model: mujoco.MjModel, prefix: str, joints: list[str]
) -> tuple[list[float], list[tuple[float, float] | None]]:
  """Each joint's motor, found by its transmission: the torque per ctrl unit (gain
  times gear, so the robot model's gear and the musculoskeletal model's gain read
  alike) and its ctrl range (None when not ctrl-limited).

  Raises:
    ValueError: If a joint has no or several motors, or a motor is not a fixed-gain
      one (a bias or activation dynamics would make torque no multiple of ctrl).
  """
  gains: list[float] = []
  ranges: list[tuple[float, float] | None] = []
  for joint in joints:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + joint)
    motors = [
      a
      for a in range(model.nu)
      if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
      and int(model.actuator_trnid[a, 0]) == joint_id
    ]
    if joint_id < 0 or len(motors) != 1:
      raise ValueError(f"Expected one motor on joint {prefix + joint!r}, got {motors}.")
    a = motors[0]
    gain = float(model.actuator_gainprm[a, 0] * model.actuator_gear[a, 0])
    fixed = (
      model.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
      and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE
      and model.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE
    )
    if not (fixed and math.isfinite(gain) and gain != 0.0):
      raise ValueError(
        f"The motor on {prefix + joint!r} ({model.actuator(a).name}) is not a "
        "fixed-gain torque motor with a nonzero gain: the controller's torque has "
        "no ctrl to command it."
      )
    gains.append(gain)
    limited = bool(model.actuator_ctrllimited[a])
    lo, hi = (float(v) for v in model.actuator_ctrlrange[a])
    ranges.append((lo, hi) if limited else None)
  return gains, ranges


def _joint_of_target(target_names: list[str]) -> list[int]:
  """For each of the term's targets, in its order, the controller's joint index
  (0 knee, 1 ankle), by name: the action's order is the entity's joint order, not
  necessarily knee first.

  Raises:
    ValueError: Unless the targets are exactly one knee and one ankle.
  """
  joints = [
    [j for j, joint in enumerate(JOINTS) if joint in name] for name in target_names
  ]
  if len(target_names) != len(JOINTS) or sorted(sum(joints, [])) != [0, 1]:
    raise ValueError(
      f"The OSL controller drives {target_names}: it needs exactly one knee and one "
      "ankle motor."
    )
  return [match[0] for match in joints]


class OslFsmAction(JointEffortAction):
  """The OSL motors under the walking controller (see the module docstring).

  Attributes:
    rules: The resolved rule set.
    fsm: The batched controller; ``fsm.state`` is each env's state code.
    tare_n: The ankle cell's hanging-leg axial reading (N), float32-exact, as the
      ``osl`` observation subtracts it.
    body_mass_kg, body_weight_n: What scales the load thresholds.
    nm_per_ctrl: (2,) the knee and ankle motors' nominal N m per ctrl unit.
    readings: (N, 5) the last reading, columns :data:`READINGS` (``force`` tared).
    torque: (N, 2) the last knee and ankle torque in N m, before the ctrl clip.
    components: Each of :data:`FSM_DIAGNOSTICS` to its (N,) value from the last
      controller step; zero under ``controller="policy"``.

  A reset puts its envs in early stance and zeroes their command, which MuJoCo then
  receives as 0 until the next step, but keeps their ``readings``, ``torque`` and
  ``components``. The env resets inside ``step()``, after the controller ran, and
  the runners read the diagnostics after ``step()`` returns, as they read the
  reward terms', which keep theirs too: zeroing would put zeros in place of every
  episode's last step.
  """

  cfg: OslFsmActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: OslFsmActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg=cfg, env=env)
    self.rules = rule_set(cfg.rules, cfg.swing_load_exit)
    for name, value, choices in (
      ("initial_state", cfg.initial_state, get_args(InitialState)),
      ("controller", cfg.controller, get_args(Controller)),
    ):
      if value not in choices:
        raise ValueError(f"Unknown {name} {value!r}; choose from {list(choices)}")
    self._joint_of_target = _joint_of_target(self._target_names)
    target_of_joint = [self._joint_of_target.index(j) for j in range(len(JOINTS))]
    entity = cfg.entity_name
    prefix = f"{entity}/"

    self._sensors: dict[str, Any] = {}
    for reading, (sensor, _) in _SENSORS.items():
      try:
        self._sensors[reading] = env.scene[scene_sensor_name(sensor, entity)]
      except KeyError as error:
        raise ValueError(
          f"The OSL controller reads {scene_sensor_name(sensor, entity)!r}, which "
          "the scene lacks: the robot's spec_fn needs the OSL sensors "
          "(mjlab.tasks.osl_split.config.sensors.with_osl_sensors)."
        ) from error

    # The host model is the nominal one: the randomization writes the warp model.
    model = env.sim.mj_model
    tare = static_load_cell_tare(model, prefix)[ANKLE_FORCE]
    self.tare_n = float(np.float32(tare[1]))
    joints = [self._target_names[target_of_joint[j]] for j in range(len(JOINTS))]
    gains, ranges = nominal_motor_gains(model, prefix, joints)
    if cfg.body_mass is None:
      mass = entity_mass(model, prefix)
    else:
      mass = float(cfg.body_mass)
    if not (math.isfinite(mass) and mass > 0.0):
      raise ValueError(f"The body mass must be positive and finite, got {mass}.")
    self.body_mass_kg = mass
    self.body_weight_n = mass * float(np.linalg.norm(model.opt.gravity))
    self.fsm = BatchedFsm(self.rules, self.body_weight_n, self.num_envs, self.device)

    # The command's bounds per joint: the term's clip, as osl_deploy's action block
    # clips, within the motor's ctrl range, which MuJoCo enforces anyway.
    low = [-math.inf] * len(JOINTS)
    high = [math.inf] * len(JOINTS)
    if cfg.clip is not None:
      clip = self._clip[0].tolist()  # Every env's clip is the same.
      for j in range(len(JOINTS)):
        low[j], high[j] = clip[target_of_joint[j]]
    for j, limits in enumerate(ranges):
      if limits is not None:
        low[j], high[j] = max(low[j], limits[0]), min(high[j], limits[1])
    self._nominal = (gains, low, high)  # The model's float64 values, for settings().
    device = self.device
    self.nm_per_ctrl = torch.tensor(gains, dtype=torch.float32, device=device)
    self._ctrl_low = torch.tensor(low, dtype=torch.float32, device=device)
    self._ctrl_high = torch.tensor(high, dtype=torch.float32, device=device)

    n = self.num_envs
    self._pending = torch.full(
      (n,), cfg.initial_state == "pose", dtype=torch.bool, device=device
    )
    self.readings = torch.zeros(n, len(READINGS), device=device)
    self.torque = torch.zeros(n, len(JOINTS), device=device)
    self.components: dict[str, torch.Tensor] = {
      name: torch.zeros(n, device=device) for name in FSM_DIAGNOSTICS
    }

  # Properties.

  @property
  def state(self) -> torch.Tensor:
    """(N,) each env's state code, an index into :data:`~..fsm.STATES`."""
    return self.fsm.state

  @property
  def ctrl(self) -> torch.Tensor:
    """(N, 2) the held motor command in ctrl units, in the term's target order:
    what ``apply_actions`` writes on every substep."""
    return self._processed_actions

  # Methods.

  def process_actions(self, actions: torch.Tensor) -> None:
    """Read the sensors, take at most one transition and command the new state's
    torque; ``actions`` (the policy's OSL rows) are ignored under the controller."""
    if self.cfg.controller == "policy":
      super().process_actions(actions)
      return
    readings = self.readings
    for column, reading in enumerate(READINGS):
      _, component = _SENSORS[reading]
      readings[:, column] = self._sensors[reading].data[:, component]
    readings[:, 4] -= self.tare_n
    knee, knee_velocity, ankle, ankle_velocity, force = readings.unbind(dim=1)
    load = -force  # The cell reads the leg's compression as negative y.

    fsm = self.fsm
    # The pose guess only initializes an episode's state, so its step counts no
    # cycle, even when the step's update then enters early stance: osl_deploy
    # starts every episode in early stance and counts entries from there.
    first = self._pending.clone()
    if self.cfg.initial_state == "pose":
      fsm.start_from_pose(self._pending, load, knee_velocity)
    self._pending.zero_()
    before = fsm.state.clone()
    state = fsm.update(load, knee, knee_velocity, ankle)
    q = torch.stack((knee, ankle), dim=1)
    qd = torch.stack((knee_velocity, ankle_velocity), dim=1)
    self.torque.copy_(fsm.torque(q, qd))
    ctrl = torch.clamp(self.torque / self.nm_per_ctrl, self._ctrl_low, self._ctrl_high)
    command = ctrl[:, self._joint_of_target]  # The term's own (target) order.
    self._processed_actions.copy_(command)
    # The command in the policy's units: what the osl group's last_action records.
    self._raw_actions[:] = (command - self._offset) / self._scale

    components = self.components
    for code, name in enumerate(STATES):
      components[f"fsm_{name}"].copy_(state == code)
    entered = (state == E_STANCE) & (before != E_STANCE) & ~first
    components["fsm_cycle"].copy_(entered)
    commanded = (ctrl * self.nm_per_ctrl).abs()
    components["fsm_knee_torque"].copy_(commanded[:, 0])
    components["fsm_ankle_torque"].copy_(commanded[:, 1])
    components["fsm_load_bw"].copy_(load / self.body_weight_n)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Early stance for the reset envs, with the pose guess pending under
    ``initial_state="pose"``, and their command (raw and ctrl) zeroed; their last
    reading and diagnostics stay (see the class docstring)."""
    super().reset(env_ids)
    rows = slice(None) if env_ids is None else env_ids
    self.fsm.reset(rows)
    self._pending[rows] = self.cfg.initial_state == "pose"
    if self.cfg.controller == "fsm":
      # Under "policy" the base term reassigns the command at every step (an
      # inference tensor when stepped in inference mode, which a reset outside it
      # could not write to) and, as stock JointEffortAction, keeps it on reset.
      self._processed_actions[rows] = 0.0

  def settings(self) -> dict[str, Any]:
    """The resolved controller settings as plain data: what a checkpoint records.

    ``body_mass`` is the configured value (None: the model's), ``body_mass_kg`` and
    ``body_weight_n`` what the controller uses; ``swing_load_exit`` is resolved.
    """
    rules = self.rules
    gains, low, high = self._nominal
    return {
      "controller": self.cfg.controller,
      "rules": rules.name,
      "every": rules.every,
      "knee_velocity": rules.knee_velocity,
      "swing_load_exit": rules.swing_load_exit,
      "body_mass": None if self.cfg.body_mass is None else float(self.cfg.body_mass),
      "body_mass_kg": self.body_mass_kg,
      "body_weight_n": self.body_weight_n,
      "initial_state": self.cfg.initial_state,
      "tare_n": self.tare_n,
      "nm_per_ctrl": dict(zip(JOINTS, gains, strict=True)),
      "ctrl_limits": {j: [low[i], high[i]] for i, j in enumerate(JOINTS)},
      "impedance": impedance_settings(),
      "thresholds": threshold_settings(),
      "sensors": {
        reading: scene_sensor_name(sensor, self.cfg.entity_name)
        for reading, (sensor, _) in _SENSORS.items()
      },
      "update": UPDATE,
    }


def fsm_component(
  env: ManagerBasedRlEnv, name: str, action_name: str = OSL_ACTION_TERM
) -> torch.Tensor:
  """One per-env diagnostic of the OSL controller from the current step (zeros when
  the action term has none)."""
  term = env.action_manager.get_term(action_name)
  components = getattr(term, "components", None)
  if components is not None and name in components:
    return components[name]
  return torch.zeros(env.num_envs, device=env.device)
