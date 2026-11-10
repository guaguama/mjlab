"""The deployment contract of the musclemimic-style OSL policies, robot and MSK.

The deployment bundle (:mod:`mjlab.tasks.osl_gait_library.export`) is built on it.

The actor leaves as an ONNX graph from the raw observation to the raw action mean.
The frozen observation statistics are part of the graph; nothing else is. Its
metadata (``policy_metadata``) is the deployment contract:

* ``observation``: the actor's terms in order, with the names that resolve every
  joint, site body, muscle and touch sensor in the compiled model. A consumer
  rebuilds the input from simulator state and the reference;
* ``action``: one block per action term, in output order, with the term's affine
  map and clip into MuJoCo ``ctrl`` (``ctrl = clip(a * scale + offset, lo, hi)``),
  the actuators it writes, the actor's sampling std and the bounds of the clipped
  Gaussian that ``expected`` inference averages over;
* ``training``: what the policy saw in training but deployment does not apply by
  itself, i.e. each actuator group's command delay.

Two inference modes turn the mean into the action:

* ``"mean"``: the raw mean. The robot model's policies run on it: their std is
  small (0.2 at init) next to the [-1, 1] action range, so the mean is what they
  were trained around. So do MSK policies trained with the stage-1 recipe, whose
  muscle std ends at 0.07;
* ``"expected"``: ``E[clip(mu + sigma eps, lo, hi)]``, per output, with the learned
  sigma: still deterministic, but the action the policy trained against on
  average. An MSK policy with a large muscle std needs it (musclemimic's settings
  left 1.4-2.5): every sampled excitation is clamped to [0, 1], so noise gives each
  muscle 0.2-0.4 of drive on average that the raw mean lacks, and raw-mean rollouts
  fall within seconds.

``default_inference`` picks ``expected`` when the policy drives muscles whose
median std exceeds 0.3 (:data:`~.expected.EXPECTED_INFERENCE_MIN_STD`), ``mean``
otherwise.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import mujoco
import numpy as np
import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.tasks.osl_mimic.mdp.observations import TOUCH_SENSORS

from .expected import auto_inference

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg
  from mjlab.envs.mdp.actions.actions import BaseAction

Inference = Literal["mean", "expected"]

SCHEMA_VERSION = 1
"""Bumped on any change a consumer would have to follow."""
ENTITY = "robot"
"""The scene entity of the OSL full-body model; model names carry ``robot/``."""
ONNX_INPUT = "observation"
ONNX_OUTPUT = "mean"
METADATA_KEY = "osl_policy"
"""The ONNX ``metadata_props`` entry that holds the metadata as one JSON string."""
GAIT_TERMS = ("twist", "phase")
"""The gait-library terms that end the observation (``osl_mimic_gait_v1``)."""
MUSCLE_FIELDS = ("length", "velocity", "force", "control", "activation")
"""``muscle_state``'s five blocks, each one value per muscle."""
_TERM_PROCESSING = ("clip", "scale", "delay_min_lag", "delay_max_lag", "history_length")
"""Observation-term settings that change what the actor reads. A consumer builds
every term raw, so none may be set. Noise is allowed: it corrupts the training
input only, and deployment reads the clean value it was added to."""


@dataclasses.dataclass(frozen=True)
class PolicyParts:
  """What a policy's metadata is read from: the task rebuilt on CPU."""

  model: mujoco.MjModel
  """The compiled scene model."""
  joint_names: Sequence[str]
  """The entity's non-root joints, in model order (no ``robot/`` prefix)."""
  obs_terms: Sequence[str]
  """The actor group's terms, in order."""
  obs_term_cfgs: Mapping[str, Any]
  """Each actor term's config (an ``ObservationTermCfg`` or its saved dict)."""
  actions: Mapping[str, BaseAction]
  """The action terms, in action-manager (policy output) order."""
  site_bodies: Sequence[str]
  lookahead_steps: int
  decimation: int
  step_dt: float
  entity_cfg: EntityCfg | None = None
  """The entity config, for the actuator delays it trained with."""
  obs_group_history: int | None = None
  """The actor group's own ``history_length`` (it stacks every term)."""


def _prefixed(name: str) -> str:
  return f"{ENTITY}/{name}"


def muscle_actuators(model: mujoco.MjModel) -> list[int]:
  """The entity's muscle actuators in ``muscle_state``'s order (model order)."""
  return [
    i
    for i in range(model.nu)
    if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
    and model.actuator(i).name.startswith(f"{ENTITY}/")
  ]


def term_sizes(parts: PolicyParts) -> dict[str, int]:
  """Each actor term's width, from the model and the reference settings.

  Raises:
    ValueError: For a term this schema does not know.
  """
  joints = len(parts.joint_names)
  sites = len(parts.site_bodies) - 1  # Relative to the root site.
  lookahead = parts.lookahead_steps - 1  # The current frame is the first one.
  known = {
    "root_pos": 5,
    "joint_pos": joints,
    "root_vel": 6,
    "joint_vel": joints,
    "muscles": len(MUSCLE_FIELDS) * len(muscle_actuators(parts.model)),
    "touch": len(TOUCH_SENSORS),
    # Site rpos, rangles, rvel (12 each), the reference rpos, then per lookahead
    # frame the root position and qvel deltas and the site rpos, then the phase.
    "goal": 12 * sites + 3 * sites + lookahead * (9 + 3 * sites) + 1,
    "twist": 3,
    "phase": 2,
  }
  unknown = [t for t in parts.obs_terms if t not in known]
  if unknown:
    raise ValueError(f"Observation terms outside the OSL schema: {unknown}")
  return {t: known[t] for t in parts.obs_terms}


def observation_schema(obs_terms: Sequence[str]) -> str:
  if tuple(obs_terms[-len(GAIT_TERMS) :]) == GAIT_TERMS:
    return "osl_mimic_gait_v1"
  return "osl_mimic_v1"


def _setting(cfg: Any, key: str) -> Any:
  return cfg.get(key) if isinstance(cfg, Mapping) else getattr(cfg, key, None)


def unsupported_processing(
  term_cfgs: Mapping[str, Any], group_history: int | None = None
) -> list[str]:
  """``term.setting`` for every actor-term setting a raw-input consumer would miss
  (see ``_TERM_PROCESSING``), and ``history_length`` for a group-wide history.
  Works on configs and on their saved dicts."""
  issues = [
    f"{name}.{key}"
    for name, cfg in term_cfgs.items()
    for key in _TERM_PROCESSING
    if _setting(cfg, key) not in (None, 0)
  ]
  return issues + (["history_length"] if group_history else [])


def default_inference(parts: PolicyParts, std: np.ndarray) -> Inference:
  """``expected`` when the policy drives muscles whose median std (``std``: the
  actor's, per output) exceeds 0.3, ``mean`` otherwise (see
  :func:`~.expected.auto_inference`)."""
  muscles = set(muscle_actuators(parts.model))
  outputs = [
    i for i, (a, _) in enumerate(_actuators(parts.model, parts.actions)) if a in muscles
  ]
  return auto_inference(np.asarray(std, dtype=np.float64).reshape(-1)[outputs])


def action_std(actor: torch.nn.Module, action_dim: int) -> np.ndarray:
  """The actor's sampling std per output, as its distribution computes it (with
  its own clamping), for a state-independent Gaussian."""
  dist = getattr(actor, "distribution", None)
  if dist is None:
    raise ValueError("The actor has no output distribution.")
  param = next(iter(dist.parameters()), None)
  device = param.device if param is not None else torch.device("cpu")
  with torch.no_grad():
    dist.update(torch.zeros(1, action_dim, device=device))
    std = dist.std.detach().cpu().numpy().reshape(-1)
  if std.shape != (action_dim,) or not np.isfinite(std).all() or (std < 0).any():
    raise ValueError(f"Invalid actor std of shape {std.shape}.")
  return std.astype(np.float64)


def _target_kind(term: BaseAction) -> tuple[mujoco.mjtObj, mujoco.mjtTrn]:
  if term.cfg.transmission_type == TransmissionType.TENDON:
    return mujoco.mjtObj.mjOBJ_TENDON, mujoco.mjtTrn.mjTRN_TENDON
  if term.cfg.transmission_type == TransmissionType.JOINT:
    return mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtTrn.mjTRN_JOINT
  raise ValueError(f"Unsupported transmission {term.cfg.transmission_type}.")


def _actuators(
  model: mujoco.MjModel, actions: Mapping[str, BaseAction]
) -> list[tuple[int, str]]:
  """(actuator id, block name) for every policy output, in output order."""
  out: list[tuple[int, str]] = []
  for name, term in actions.items():
    kind, trn = _target_kind(term)
    for target in term.target_names:
      target_id = mujoco.mj_name2id(model, kind, _prefixed(target))
      matches = np.flatnonzero(
        (model.actuator_trntype == trn) & (model.actuator_trnid[:, 0] == target_id)
      )
      if target_id < 0 or len(matches) != 1:
        raise ValueError(f"Expected one actuator on {target!r}, found {len(matches)}.")
      out.append((int(matches[0]), name))
  return out


def _vector(value: Any, size: int) -> np.ndarray:
  """A term's scale or offset (a float, or a per-env tensor) for one env."""
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().double().reshape(-1, size)[0].numpy()
  return np.full(size, float(value))


def _clip(term: BaseAction) -> tuple[np.ndarray, np.ndarray]:
  size = term.action_dim
  clip = getattr(term, "_clip", None)
  if term.cfg.clip is None or clip is None:
    return np.full(size, -np.inf), np.full(size, np.inf)
  values = clip.detach().cpu().double().numpy()[0]
  return values[:, 0], values[:, 1]


def _effective_bounds(
  model: mujoco.MjModel, ids: list[int], scale, offset, low, high
) -> tuple[np.ndarray, np.ndarray]:
  """The range each output is effectively clipped to, in policy units.

  The action terms' clips map to [-1, 1] in policy units (the OSL motors scale by
  2.88 and clip to +-2.88, the servos map [-1, 1] onto their joint range and clip
  to it, the muscles clip to [-1, 1]), and MuJoCo clamps a muscle's excitation to
  [0, 1] inside its activation dynamics. That is checked here against each term's
  own map, so a term with another mapping fails instead of averaging wrongly.
  """
  muscle = model.actuator_dyntype[ids] == mujoco.mjtDyn.mjDYN_MUSCLE
  lo = np.where(muscle, 0.0, -1.0)
  hi = np.ones(len(ids))
  with np.errstate(divide="ignore", invalid="ignore"):
    derived_lo = (low - offset) / scale
    derived_hi = (high - offset) / scale
  derived_lo = np.where(
    muscle, np.maximum(derived_lo, (0.0 - offset) / scale), derived_lo
  )
  derived_hi = np.where(
    muscle, np.minimum(derived_hi, (1.0 - offset) / scale), derived_hi
  )
  bad = (np.abs(derived_lo - lo) > 1e-5) | (np.abs(derived_hi - hi) > 1e-5)
  if bad.any():
    names = [model.actuator(ids[i]).name for i in np.flatnonzero(bad)][:5]
    raise ValueError(
      f"Action outputs whose clip is not [-1, 1] (or [0, 1] for muscles) in policy "
      f"units: {names}"
    )
  return lo, hi


def action_metadata(
  parts: PolicyParts, std: np.ndarray, inference: Inference
) -> dict[str, Any]:
  """The ``action`` section: blocks, the affine maps, clips, std and bounds."""
  model = parts.model
  actuators = _actuators(model, parts.actions)
  ids = [a for a, _ in actuators]
  if len(set(ids)) != len(ids) or sorted(ids) != list(range(model.nu)):
    raise ValueError("The action terms must drive every actuator exactly once.")
  if std.shape != (len(ids),):
    raise ValueError(f"std has {std.shape} values for {len(ids)} outputs.")
  blocks, scale, offset, low, high = [], [], [], [], []
  start = 0
  for name, term in parts.actions.items():
    size = term.action_dim
    lo, hi = _clip(term)
    blocks.append(
      {"name": name, "start": start, "size": size, "targets": list(term.target_names)}
    )
    scale.append(_vector(term.scale, size))
    offset.append(_vector(term.offset, size))
    low.append(lo)
    high.append(hi)
    start += size
  scale_a, offset_a = np.concatenate(scale), np.concatenate(offset)
  low_a, high_a = np.concatenate(low), np.concatenate(high)
  expected_low, expected_high = _effective_bounds(
    model, ids, scale_a, offset_a, low_a, high_a
  )
  return {
    "size": len(ids),
    "actuator_names": [model.actuator(i).name for i in ids],
    "blocks": blocks,
    "scale": scale_a.tolist(),
    "offset": offset_a.tolist(),
    "clip_low": low_a.tolist(),
    "clip_high": high_a.tolist(),
    "std": std.tolist(),
    "expected_low": expected_low.tolist(),
    "expected_high": expected_high.tolist(),
    "default_inference": inference,
    "units": (
      "Policy units; ctrl = clip(a * scale + offset, clip_low, clip_high), with a "
      "the raw mean ('mean') or E[clip(N(mean, std^2), expected_low, "
      "expected_high)] ('expected'). OSL motor ctrl is not joint N m."
    ),
  }


def observation_metadata(parts: PolicyParts, size: int) -> dict[str, Any]:
  model = parts.model
  return {
    "schema": observation_schema(parts.obs_terms),
    "size": size,
    "terms": list(parts.obs_terms),
    "term_sizes": term_sizes(parts),
    "body_names": [_prefixed(b) for b in parts.site_bodies],
    "muscle_names": [model.actuator(i).name for i in muscle_actuators(model)],
    "touch_names": [_prefixed(s) for s in TOUCH_SENSORS],
    "normalization": "Frozen checkpoint RunningMeanStd, baked into actor.onnx",
    "frames": "wxyz quaternion; root qvel linear world/angular body; mimic rpos world",
  }


def training_metadata(parts: PolicyParts) -> dict[str, Any]:
  """Each action block's actuator command delay in training: a lag drawn per env
  (per episode when held), which deployment has to reproduce itself."""
  delays: dict[str, Any] = {}
  articulation = getattr(parts.entity_cfg, "articulation", None)
  groups: list[Any] = list(getattr(articulation, "actuators", None) or ())
  physics_dt = parts.step_dt / parts.decimation
  for name, term in parts.actions.items():
    owners = [
      g
      for g in groups
      if all(
        any(re.fullmatch(expr, t) for expr in g.target_names_expr)
        for t in term.target_names
      )
    ]
    if len(owners) != 1:
      delays[name] = None
      continue
    g = owners[0]
    lags = [int(g.delay_min_lag), int(g.delay_max_lag)]
    delays[name] = {
      "steps": lags,
      "seconds": [lag * physics_dt for lag in lags],
      "held_per_episode": float(g.delay_hold_prob) >= 1.0,
    }
  return {
    "actuator_delay": delays,
    "physics_dt": physics_dt,
    "note": (
      "Command delay of each action block in training, in physics steps. The "
      "in-runner validation plays the middle of each range; a deployment without "
      "it runs the policy outside its training distribution when the range "
      "excludes zero."
    ),
  }


def policy_metadata(
  parts: PolicyParts, std: np.ndarray, obs_dim: int, inference: Inference | None
) -> dict[str, Any]:
  """The deployment contract of one policy (see the module docstring).

  Raises:
    ValueError: If the actor's input width differs from the schema's, or an actor
      term is processed in a way a raw-input consumer cannot reproduce.
  """
  issues = unsupported_processing(parts.obs_term_cfgs, parts.obs_group_history)
  if issues:
    raise ValueError(f"Actor observation processing outside the schema: {issues}")
  size = sum(term_sizes(parts).values())
  if size != obs_dim:
    raise ValueError(f"The actor reads {obs_dim} values; the schema gives {size}.")
  inference = inference or default_inference(parts, std)
  return {
    "schema_version": SCHEMA_VERSION,
    "mujoco_version": mujoco.__version__,
    "decimation": parts.decimation,
    "policy_dt": parts.step_dt,
    "root_joint": _prefixed("root"),
    "joint_names": [_prefixed(j) for j in parts.joint_names],
    "io": {
      "input": ONNX_INPUT,
      "input_shape": [1, obs_dim],
      "output": ONNX_OUTPUT,
      "output_shape": [1, len(std)],
    },
    "observation": observation_metadata(parts, obs_dim),
    "action": action_metadata(parts, std, inference),
    "training": training_metadata(parts),
  }


def export_actor_onnx(actor: torch.nn.Module, path: str | Path) -> int:
  """Write the actor's raw mean, behind its frozen normalizer, as ONNX on CPU.

  The training actor is untouched (``as_onnx`` works on a copy). Returns the input
  width.
  """
  module = actor.as_onnx(verbose=False)  # type: ignore[operator]
  module.to("cpu")
  module.eval()
  obs_dim = int(module.input_size)
  torch.onnx.export(
    module,
    (torch.zeros(1, obs_dim),),
    str(path),
    export_params=True,
    opset_version=18,
    input_names=[ONNX_INPUT],
    output_names=[ONNX_OUTPUT],
    dynamic_axes={},
    dynamo=False,
  )
  return obs_dim


def attach_metadata(onnx_path: str | Path, metadata: Mapping[str, Any]) -> None:
  """Embed the metadata as one JSON string under ``METADATA_KEY``."""
  import onnx

  graph = onnx.load(str(onnx_path))
  entry = onnx.StringStringEntryProto()
  entry.key = METADATA_KEY
  entry.value = json.dumps(metadata, allow_nan=False)
  graph.metadata_props.append(entry)
  onnx.save(graph, str(onnx_path))


def onnx_parity(
  onnx_path: str | Path, actor: torch.nn.Module, observations: np.ndarray
) -> tuple[np.ndarray, float]:
  """Run ``observations`` (N, D) through onnxruntime and the frozen torch actor.

  Returns the ONNX means (N, A) and the largest absolute difference.
  """
  ort = cast(Any, importlib.import_module("onnxruntime"))
  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  options.inter_op_num_threads = 1
  session = ort.InferenceSession(
    str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
  )
  module = actor.as_onnx(verbose=False)  # type: ignore[operator]
  module.to("cpu")
  module.eval()
  obs = observations.astype(np.float32)
  means = np.stack(
    [np.asarray(session.run(None, {ONNX_INPUT: o[None]})[0])[0] for o in obs]
  )
  with torch.inference_mode():
    expected = module(torch.from_numpy(obs)).numpy()
  return means, float(np.max(np.abs(means - expected)))


def normalizer_samples(actor: torch.nn.Module, count: int, seed: int = 0) -> np.ndarray:
  """``count`` observations drawn from the actor's own input statistics
  (mean + std * z), the first one the mean itself: inputs the network has seen."""
  norm = getattr(actor, "obs_normalizer", None)
  mean = getattr(norm, "mean", None)
  var = getattr(norm, "var", None)
  if not isinstance(mean, torch.Tensor) or not isinstance(var, torch.Tensor):
    raise ValueError("The actor has no running observation statistics.")
  mean_np = mean.detach().cpu().double().numpy()
  std_np = np.sqrt(var.detach().cpu().double().numpy())
  z = np.random.default_rng(seed).standard_normal((count, mean_np.size))
  z[0] = 0.0
  return (mean_np + std_np * z).astype(np.float32)
