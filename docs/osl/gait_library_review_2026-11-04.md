---
orphan: true
---

# OSL gait-library review (2026-11-04)

This is a read-only review of five questions:

- why the OSL MSK mimic task fails here but trains in musclemimic;
- how sesteban's G1 unicycle gait-library task works compared with our draft port;
- which rewards to use when the gait-library task is rebuilt;
- the MSK finger issue;
- whether mj-nlp, clipper and mjlab interface correctly.

It ran in two rounds. Round 1 had six reviewers. Round 2 had four adversarial verifiers,
three independent reward proposals, a synthesis, and a numeric validation on the real
library. No code was changed.

**Units.** mjlab's RewardManager pays `weight × term × step_dt` with step_dt = 0.02.
Per-step numbers below are quoted as **rew/dt**, the weighted sum before the ×dt. That
makes them directly comparable with the sum of positive weights, which is 5.0-6.0.

## 1. Summary

| Topic | Verdict |
|---|---|
| MSK mimic | The per-step reward starts strongly negative (about −84 rew/dt against a +5 maximum), so ending the episode is optimal. The adaptive LR then sits at its 1e-5 floor. musclemimic clips its total at 0. Removing the fingers alone only moves init to about −70 rew/dt. |
| Gait-library draft | A faithful port of sesteban's command code, but with inherited flaws: an ill-posed twist reward, drifting egocentric root targets, a blend that slides the feet, and a pitch-blind tilt termination. Keep the loader and helpers; rebuild rewards, terminations and blend. |
| Rewards | A concrete set (section 4) scores 1.0 on every positive term for a perfect tracker of the current library. The init reward is about −0.05 rew/dt at std 1 (robot) and +1.29 (MSK). No training evidence yet. |
| Fingers | Designed and prototyped in scratch. MSK becomes nq 80/84, nu 341, with a skeleton bit-identical to the robot. |
| Three-repo interface | Mechanically correct for the forward library, which clipper's committed code reproduces bit-exactly. Open: a physics mismatch, the idle clip, provenance, and blockers for turning libraries. |
| Bugs in committed code | `anchor_ori` is pitch-blind in all four OSL tracking tasks. The servo ctrlrange clamp leaves the knee up to 174 N·m short. The self-collision reward was removed on a false premise. |

## 2. MSK mimic: musclemimic vs mjlab

The failed run is `logs/rsl_rl/osl_fullbody_msk_compliant_tracking/2026-10-20_14-07-01`:
7.7k iterations, 4096 envs, stock G1 PPO.

| | musclemimic (tracked on v1/v2) | mjlab MSK (failed) |
|---|---|---|
| Reward total | `max(Σ w·r + penalties, 0)`: never negative | Unclipped sum. At init about −84 rew/dt: action_rate about −71, joint_limit about −7, tracking about +1.7 |
| Action-rate penalty | none | −0.1 × Σ over 403 raw actions |
| Joint-limit penalty | none | −10 × Σ over 117 joints; 61% of its init cost is the 40 finger joints |
| Tracking signal | CLF over 102 pelvis-relative channels, including error velocity (tracked). MimicReward failed validation. | 6 BeyondMimic exp terms over 18 bodies, world-frame velocities |
| Learning rate | Muon, 4e-4 linear to 0, 1 epoch | Adam, adaptive (desired_kl 0.01, KL summed over 403 dims): at the 1e-5 floor for 78% of iterations, 96% in 1000-3000 |
| Terminations | Mean site deviation 1.0 m, root 1.0 m, root angle 1.0 rad | z error 0.25 m on pelvis, toes_l, OSL foot, both lunates, plus tilt. These fire 2-3× sooner on identical rollouts (16.6 vs 29.7 steps under N(0,1)) |
| Init and perturbations | Exact reference state; no noise, push or randomisation | RSI ±0.1 rad on every joint, pushes every 1-3 s, DR, observation noise |
| Actor observations | 5×339 muscle states, 0.8 s reference lookahead | No muscle state, no lookahead |
| Networks | 12×1024 + 3×2048, SiLU + LayerNorm | 512-256-128 |
| Simulation | Euler, 2 ms × 5 = 100 Hz | implicitfast, 2 ms × 10 = 50 Hz (warp drops the muscle velocity derivatives) |
| Fingers | deleted (nu 341) | kept (nu 403) |

**Mechanism.**
1. With a negative per-step reward and non-bootstrapped terminations, dying is optimal.
   Episodes fell from 16.4 to about 3 steps (iterations 50-2500).
2. Shrinking the policy std is the only way to cut a raw action-rate penalty, so std fell
   from 1.0 to 0.12.
3. The std collapse compounds with KL summed over 403 dims (KL scales as 1/σ²) and pins
   the adaptive LR at its floor.
4. The per-step sum turned positive only around iteration 1500-2000. Episodes then
   plateaued near 24 steps, with a body error of about 15 cm.

**musclemimic's rewards.** These are the configs that ran on v2.
- **MimicReward:**
  - Each term is `exp(−k·mean(err²))` over pelvis-relative quantities: qpos 0.2 (k 10),
    qvel 0.1 (k 2), root_pos 0.1, root_vel 0.1, site rpos 0.6 (k 100, equivalent to a
    0.17 m std), rquat 0.01, rvel 0.1 + 0.1 (k 0.1, nearly flat).
  - Penalties: out-of-bounds actions 0.01, OSL torque 1e-5·τ², floored at −1.
  - Maximum 1.31 per step at 100 Hz.
- **CLF:**
  - 102 channels: rpos 48 (scale 0.05 m), rangles 48 (0.2 rad), root_pos 3, root_ori 3.
  - V is a double-integrator CARE quadratic in the error and its time derivative (τ 0.1 s).
  - Level `exp(−min(V/V_max, 5))`, minus 0.25 × a decay-violation term, clipped at ≥ 0.
- v2 rigid CLF reached val rpos 9 mm at 1B steps (35 h); MimicReward runs failed
  validation.
- No musclemimic run exists on v3. v2 differs only by a too-short pylon, so v2 counts as
  valid evidence.

## 3. sesteban's G1 unicycle vs our draft

### 3.1 How sesteban's task works (fork at 9d2b702; origin has not moved)

- **Library.**
  - walk_unicycle has 379 clips × 70 frames (1.4 s):
    - forward: vx 0.5-0.9 × wz ±0.5, step 0.05;
    - backward: vx −0.8..−0.5 × wz;
    - pivots: |wz| 0.5-1.5;
    - idle.
  - Clips loop at their own length and are phase-aligned offline on ankle height.
- **Sampler.**
  - Modes: idle 15%, pivot 30%, arc 55% (40% of arcs backward). Resample every 3-8 s.
  - Snap to the nearest clip by span-normalised weighted L2.
  - The twist observation steps immediately; the reference blends over 0.4 s at the
    shared phase.
- **Egocentric root targets.** The anchor velocity is integrated and re-based to the robot
  at each resample, because looping clips' absolute pose sawtooths.
- **Actor (98 dims, reference-free).**
  - IMU angular velocity, biased joint pos, joint vel, last action;
  - commanded twist, `(sin, cos)` phase (`(0, 0)` at idle), projected gravity.
- **Critic.**
  - Adds the reference and body poses.
  - Flaw: its anchor terms read the clip's absolute looping anchor, not the egocentric
    error.
- **Base layer (g1_custom).**
  - DR: COM ±5 cm, encoder bias ±0.015, foot friction 0.3-1.2, Kp/Kd ×0.8-1.2,
    armature ×0.95-1.05, torso mass ×0.9-1.1.
  - Actuator delay 0-40 ms; pushes every 1-3 s.
  - Init state and action offset are the idle pose. Episodes are 10 s.
  - PPO: stock G1 (512-256-128, lr 1e-3 adaptive, init std 1.0).

### 3.2 His walk_unicycle rewards

| Term | Weight | Definition |
|---|---|---|
| motion_body_pos / ori / lin_vel / ang_vel | 1.0 each | exp(−mean over 14 bodies of err² / σ²), σ 0.3 m / 0.4 / 1.0 / 3.14 |
| egocentric root pos / ori | 0.5 each | exp(−err²/σ²), σ 0.3 m / 0.4 rad against the egocentric targets |
| twist | 0.3 | exp(−‖[v_fwd, v_lat, ω_z] − cmd‖² / 0.5²), instantaneous, pelvis body-x heading, not gated |
| joint_limit | −10 | Σ violation over 29 joints, soft factor 0.9 |
| self_collisions | −10 | number of the 4 substeps with any pelvis-subtree self-contact above 10 N |
| action_rate limbs / waist | −0.1 / −0.3 | Σ (a_t − a_{t−1})² on raw actions over 26 / 3 dims |
| foot_slip / soft_landing | −0.1 / −0.001 | stance-foot speed² / landing force, gated on \|cmd\| > 0.05 |

Variants:
- SwingFoot adds toe_clearance −1 and swing_foot_pitch −1.
- Jog uses twist σ 0.7 with foot terms 10× and 100× lighter.
- WalkJog uses σ 0.6 ("midpoint, not a measurement").
- He lowered action rate from −0.2/−0.4 to −0.1/−0.3 in 9d2b702 with no stated reason.
- His walk twist σ 0.5 was measured on a different library (walk_diffdrive). On
  walk_unicycle a perfect replay scores only about 0.60 (0.49 on pivots).

### 3.3 What our draft is

`src/mjlab/tasks/gait_library/` is uncommitted and was never trained. An AST diff shows
its command, loader and blend behave the same as sesteban's. It adds two things:
- the twist heading taken from a configurable forward axis (body −y on OSL);
- egocentric critic anchors.

It is layered on the single-clip OSL tracking config, so it also inherits that config's
choices:
- action centre WALK_START;
- raw-torque OSL actions;
- G1 PPO;
- 20 s episodes;
- `ee_body_pos` terminations on the hands.

It leaves out sesteban's foot terms, action-rate split, DR set and actuator delay.

### 3.4 Problems measured in the draft

| Problem | Measurement |
|---|---|
| Twist reward is ill-posed: the stride-mean label is compared with the instantaneous twist | A perfect tracker scores 0.59-0.71 on the slow and mid bands and 0.25-0.29 on the fast band. It scores 0.61 at 0.95 m/s vs 0.62 at 1.0, so it cannot separate commands. It pays the policy to damp natural pelvis yaw. |
| Egocentric root targets drift: full 3D re-base, plain-lerped anchor velocity, data inconsistency | Heading vs clip: mean 3.2°, p95 10.4°, max 13.6°. z: mean 7 mm, p95 30 mm, max 52 mm. Under them a perfect tracker scores body_ori 0.96. |
| Egocentric lag is never forgiven until the next resample (3-8 s) | Inferred: a 1 m/s command step leaves about 0.5 m of path error; the position term sits near 0.06 with no gradient. |
| Blend replays the outgoing clip at a rate ratio with unscaled anchor velocity | Served stance foot slides 4-13 cm per switch. The served joint step reaches 0.295 rad, vs 0.191 rad in steady state. |
| Tilt termination is pitch-blind (section 8) | Never fires on pitch, at any angle. |
| Action centre WALK_START has the knee on its extension stop | Reproducing library poses needs \|a\| up to 14.4 (9.5 with a library-midrange centre). |
| Adaptive sampling | 2 bins pooled over all clips, so effectively uniform. |
| Init reward at policy std 1 | Robot −5.1 rew/dt (action_rate −5.6, joint_limit −1.5; 86% of the latter is the raw-torque OSL ankle). MSK −84 rew/dt. |

### 3.5 Salvage verdict

| Component | Verdict |
|---|---|
| `LibraryMotionLoader` | Keep and extend: manifest, provenance, an `fps`/period check. |
| `OslReferenceMixin` (uncommitted diff in `tracking/config/osl/mdp.py`) | Keep. It handles (S, T, J) padding and coupled-joint resets. |
| `LibraryMotionCommand` | Rework: bounded random-walk sampler, snap hysteresis, twist window buffers. Remove the egocentric integration. |
| `BlendingLibraryMotionCommand` | Rebuild as a time-warp: shared phase at the blended frequency, each clip's velocities scaled by its own playback rate. |
| Egocentric targets, rewards and critic obs | Drop. |
| Observations | Keep `commanded_twist` and `motion_phase`. Add the critic terms in 4.4. |
| Rewards | Keep the `root_twist()` helper (its body −y heading is verified). Rebuild everything else. |
| Config | Keep the entity, sim, DR and actor surgery. Replace the rewards and terminations wholesale; do not inherit them. |
| Tests | Keep the contract, loop and forward-axis tests. Add stats-pass, twist, blend-slip and tilt tests. |

## 4. Recommended rewards for the rebuilt task

Principles:
- An imitation backbone, plus a well-posed speed signal and prosthesis-specific terms.
- **Every std and normaliser is derived by a load-time stats pass over the loaded
  library.** The pass writes `library_stats.json` with the library hash. Values below are
  for the current KIT library; the stage-2 library re-derives its own.
- Positive weights sum to W+ = 6.0.

### 4.1 Positive terms (robot and MSK)

| Term | Weight | Definition | Std procedure (KIT value) |
|---|---|---|---|
| body_pos | 1.0 | Stock relative-body kernel over 17 bodies (TRACKED_BODIES minus the duplicate `Full Body`). Reference re-anchored at the robot pelvis xy with the lifted z*, yaw-rotated. | σ = √2 × phase RMS, clip-weighted by the sampler's π (0.179 m) |
| body_ori | 1.0 | Same 17 bodies; arm bodies weighted 0.5 inside the mean, because the subject boundaries live in the arms | 0.442 rad |
| body_lin_vel | 1.0 | Same bodies, heading-rotated served reference | 1.128 m/s |
| body_ang_vel | 1.0 | Same | 3.29 rad/s |
| root_height_tilt | 0.5 | exp(−½[(z − z*)²/σz² + θ²/σθ²]). z* is the served pelvis z plus a per-frame lift c(t) that cancels the reference feet sitting in the floor; θ is the pelvis gravity-tilt angle. Replaces the egocentric root terms. | σz 25 mm, σθ 0.080 rad |
| osl_joint | 0.5 | exp(−¼ Σ over OSL knee and ankle of [(q − q̂)²/σq² + (q̇ − q̂̇)²/σq̇²]) | σq (0.472, 0.229) rad, σq̇ (4.11, 2.38) rad/s |
| twist_stride | 0.5 | Box mean over one stride (the snapped clip's period) of the root twist, heading from `Full Body` −y. Target: the box mean of the served reference's root twist + (cmd − L_snap). Triangular window (box of box) on ω_z. Buffers prefilled at reset. | σ per axis = Δ/√(−ln 0.78) ≈ 2 grid steps (0.1003) |
| contact_match | 0.5 | Mean over both feet of 1[robot foot state compatible with the reference flag]. Contact if the lowest foot-geom point is below 3 mm, swing if above 15 mm, don't-care in between. | Physical bands, not tuned |

Twist definition details (verified):
- The reference sample must use `Full Body` body −y (body index 0), or pelvis +x.
- Pelvis −y points down; using it scored 0.005-0.49 for a perfect tracker.
- The ω_z triangular window keeps the fast band's ±5% cadence error at 0.96-0.98, against
  0.81-0.87 with a single box.

### 4.2 Penalties

| Term | Model | Weight | Definition |
|---|---|---|---|
| servo_action_rate (limbs / lumbar) | robot | ≈ −0.0058 / −0.0117 | Σ (a_t − a_{t−1})² on raw servo actions; OSL dims excluded. Weight set so a perfect tracker pays G1's share (0.155% of W+). The stock −0.1 would cost a perfect tracker 0.14-0.44 rew/dt. |
| joint_limit | robot: 29 actuated joints; MSK: the 2 OSL joints | −10 | Σ violation, limits widened to the library's own range |
| self_collisions | robot | −1 → −4 per substep | Legs-only scope (femur_r vs femur_l subtrees), threshold 24.7 N, history 10; ramp to −4 once episodes reach 50% length. Full-body scope only after a QA-passing idle exists. |
| self_collisions | MSK | −0.25 → −4 per substep | Ramp over ≥ 1000 iterations, gated on the measured contact count (< ~1 substep), not on episode length |
| osl_torque_rate | both | −0.25 | −w(1 − e^(−x/x0)), x = Σ ((τ_t − τ_{t−1})/dt)² over the OSL knee and ankle; x0 per speed band |
| osl_impact | both | −0.1 | Bounded, on the OSL first-contact vertical speed |
| osl_swing_clearance / intact_swing_clearance | both | −0.4 / −0.1 | Deficit below h*(t) = max(lifted reference lowest point, intact minimum toe clearance) during reference mid-swing |
| foot_slip | both | −0.2 | Bounded; stance-foot horizontal speed², not gated by the command |
| muscle_out_of_bounds | MSK | −0.25 | Mean over 339 muscles of relu(\|a\| − 1)² |
| nonneg_floor | MSK | last term | relu(−sum of all earlier terms), so the MSK total is never negative |
| osl_power, twist_coarse | both | 0 now | Candidates for phase 3b (−0.1 and +0.5) |

### 4.3 Terminations (rebuilt, not inherited)

- **time_out:** 20 s.
- **anchor_pos:** |z* − z_pelvis| > 0.25 m, against the lifted target.
- **tilt:** `bad_osl_anchor_ori` at **0.5** (60°), frame-agnostic.
- **fall:** any terrain contact on the non-foot collision geoms (tibia, pelvis, thorax,
  femur capsules).
- **ee_body_pos removed.** With raw-torque OSL it fired first in 998 of 1010 noisy episodes
  and halved them. With impedance OSL it fires rarely.

### 4.4 Coupled settings

- **Command.**
  - Ranges are read from the library labels.
  - Resample every U(3, 8) s: 15% idle, 20% uniform jumps, otherwise a bounded random walk
    (±4 grid steps).
  - Snap hysteresis of 0.25 grid step. Uniform start phase.
  - With independent draws, 34% of walk→walk switches cross two or more subject
    boundaries.
- **Blend.** Time-warp: 0.4 s walk↔walk, one outgoing stride for walk↔idle.
- **Servos.**
  - Scale 0.25·F/Kp (G1 convention; |a| of 10-14 is learnable and unclipped).
  - Centre on the solved idle pose; fallback is the library midrange. Never WALK_START.
  - Unclamping the ctrlrange is a decision (section 5).
- **OSL actions.** Impedance or raw torque is a decision (section 5).
- **Init std.**
  - Robot: 1.0, falling back to 0.5 if the init total drops below −1.0 rew/dt.
  - Robot PPO: G1 settings.
  - MSK PPO:
    - fixed or linear LR 3e-4 → 3e-5, or desired_kl ≈ 0.12 (never the 1e-5 adaptive
      floor);
    - entropy ≤ 4e-4;
    - nets ≥ 1024-512-256;
    - muscle activation in the actor obs.
- **Critic additions:**
  - twist-window error, cmd − L_snap, z − z*, θ;
  - reference OSL q and q̇, contact flags and swing masks, foot heights;
  - previous OSL torque, feet contact, blend α.
- **Library QA gates at load:**
  - idle static, on the coupler manifold, soles within ±2 mm, zero pair penetration;
  - every perfect tracker ≥ 0.95 on twist, with |δ_c| ≤ 0.5σ;
  - perfect-tracker penalty ≤ 2% of W+;
  - band-edge jump report.

### 4.5 MSK deltas

- Fingerless model (section 6).
- The same positive and bounded terms as the robot.
- joint_limit on the OSL joints only. No muscle action rate.
- muscle_out_of_bounds and the non-negative floor.
- RSI noise and pushes off at first.
- self_collisions ramped.
- Integrator: Euler at 100 Hz or implicitfast at 50 Hz, decided by a short A/B run. The
  spec is rate-invariant apart from γ and the twist window.

### 4.6 Validation on the real library (robot-rigid unless noted)

| Check | Result |
|---|---|
| Perfect replay, every positive term, every band | 1.000; total 5.96-5.99 rew/dt; no terminations |
| Perfect-tracker penalty budget | 0.04-0.06 rew/dt (slow and mid bands), 0.118 (fast band, at the 2% gate) |
| Init, std 1.0, impedance OSL | −0.52 rew/dt, 40-step episodes; **−0.05 with legs-only self-collision** |
| Init, std 0.5 / zero action | +1.16 / +2.10 rew/dt |
| Init, std 1.0, raw-torque OSL | −2.63 rew/dt (joint_limit −1.82) |
| MSK fingerless, std 1.0 | +1.29 rew/dt; the floor binds on 5.5% of steps |
| Converged single-clip robot policy on these stds | body terms 0.92-0.94, root_height_tilt 0.80, osl_joint 0.88 |
| Twist discrimination (1.0 m/s tracker commanded 1.0 / 0.975 / 0.95 / 0.9 / 0.8 / 0.5 / idle) | 1.00 / 0.94 / 0.78 / 0.35 / 0.02 / 0 / 0 |
| G1 arcs and pivots (proxy) | perfect tracker 1.00 on all 378 clips; one grid step 0.78 |
| Lazy stand under walk commands (0.5 / 1.0 / 1.5 m/s) | walking earns 1.96 / 2.52 / 3.37× standing |
| Current idle clip | −40 rew/dt even for a perfect tracker (self-collision on every substep) |

These fixes from validation are already folded into the tables above:
- the twist reference frame;
- the ω_z triangular window;
- legs-only self-collision with a ramp, for robot and MSK;
- clip-weighted std pooling;
- clearance targets in the lifted frame;
- per-band torque-rate x0.

## 5. Open decisions (recommended default first)

1. **OSL action space.** Recommended: impedance, as position actuators (Kp 406.5 / 134.3,
   Kd 7.97 / 1.79, forcerange ±142.3 / ±168.2 N·m). The alternative is raw torque.
   - Init: −0.52 vs −2.63 rew/dt.
   - Impedance keeps the robot a pure JointPositionAction, so ONNX export works.
   - Phase 3b inherits this choice.
2. **Prosthesis-specific terms** (osl_joint, contact_match, clearance, torque rate,
   impact): include them. They are the least proven part of the design.
3. **Egocentric root targets:** drop them. Add a leaky yaw-hold in phase 3b if
   straight-line holding becomes a criterion.
4. **Unclamp the 27 servo ctrlranges** (G1 convention). The clamp leaves knee_angle_l up
   to 174 N·m short in up to 18.6% of steps; unclamped, the worst is 5.5 N·m. It diverges
   from musclemimic's servo twin.
5. **Idle clip:** mj-nlp's solved Bones median stance. The current clipper `stand`
   keyframe is off the coupler manifold, self-penetrating and unstable.
6. **When to train:** a robot-rigid shake-down on the KIT library is fine. Wait for the
   stage-2 unicycle library before a run meant to be kept.
7. **Defaults taken unless you object:** legs-only self-collision ramp, arms weighted 0.5
   in body_ori, a bounded random-walk command, a time-warp blend.

## 6. Finger removal plan (designed, not applied)

- **What to delete.**
  - musclemimic's exact lists: 40 joints, 62 muscles, and their tendons, matched by the
    deleted actuators' `.target`.
  - Do it inside `get_spec` (default fingerless for msk), **after** `_strip_keyframes`.
    Doing it before fails with "keyframe 'stand': invalid qpos size".
  - Raise unless exactly 40 + 62 are found.
- **Prototype result:**
  - rigid nq 80, nv 79, nu 341, ntendon 347; compliant nq 84;
  - the joints are exactly the robot's 73/77 in the same order, and kinematics, masses and
    equalities are bit-identical;
  - warp vs CPU drift is about 500× smaller.
  - The extrinsic finger muscles also cross the wrist, so wrist capacity roughly halves.
    That matches musclemimic.
- **Why the default lives in `get_spec`:** `tracking/config/osl/env_cfgs.py:127` builds a
  second spec to size the per-joint noise lists.
- **Must change with it:**
  - `test_osl_constants` DIMS → (80, 73, 341) and (84, 77, 341), 339 muscles;
  - `test_osl_tracking` 113/117 → 73/77, 401 → 339;
  - the `test_osl_rollout` MSK tolerances;
  - the docstrings, and the unreleased changelog line "401 muscles".
- **Consequences:**
  - MSK action dim 403 → 341;
  - padding becomes a no-op for MSK-rigid;
  - old MSK checkpoints are incompatible;
  - the robot needs nothing.
- **It does not fix the MSK reward sign by itself.**

## 7. mj-nlp → clipper → mjlab

**Verified OK:**
- Model XMLs are byte-paired across musclemimic, clipper, mj-nlp and mjlab, apart from one
  banner line and the `meshdir`.
- Node schema `state` = [qpos 80 | qvel 79].
- Joint order (73) and body order (105) match all four mjlab variants. FK error ≤ 1.3e-7 m.
- Velocities are consistent with mjlab's link convention.
- Seams are within tolerance. Yaw per stride is 0. The forward axis (body −y) and the
  signs (vy left, ω_z CCW) agree in all three repos.
- clipper `39a1a13` reproduces all 22 files bit-exactly.

**Problems:**

| Area | Problem | Owner |
|---|---|---|
| Contact physics | Solved with soft contact (solref 0.04, solimp 0/0.95/0.01), an elliptic cone, impratio 3 and μ 1.0. mjlab uses stiff XML contact, a pyramidal cone, impratio 1 and foot μ 0.6 (DR 0.3-1.2). Reference stance feet sit 10-20 mm inside mjlab's floor and slide 15-51 mm per stride; fast-band friction demand reaches 0.94 (p95). | mjlab compensates (lift); mj-nlp records its physics |
| Actuation | Solved with ideal torques; mjlab clamps servo ctrl to the joint range | mjlab (decision 4) |
| Idle | The `stand` keyframe is 52 mm off the coupler manifold, has 11.4 mm arm/thorax overlap on active pairs and is unstable. clipper only accepts a keyframe name. | mj-nlp (solved idle) + clipper (accept a node) |
| Subject boundaries | Four KIT subjects: matched-phase jumps of 0.94 / 0.62 / 0.52 rad at 0.65/0.70, 0.90/0.95, 1.20/1.25 m/s, mostly arms. Stage 2 keeps 3-5 forward boundaries. | mj-nlp (report continuity), mjlab (blend, arm weighting) |
| Turning libraries | clipper's seam check is all-or-nothing (the arc library fails it); it never reads `index.json`, so 18 unconverged arcs would be served; compliant libraries are unreadable; there is no merge tool | clipper |
| Turning in mjlab | No unicycle sampler or turning ranges | mjlab |
| Provenance | mj-nlp outputs and solver config are uncommitted; `index.json` git_sha points at a dirty tree; clipper writes no manifest; mjlab reads a gitignored sibling path | all three |
| Frame rate | mjlab ignores `fps`; 50 Hz is implicit and unchecked | mjlab |

## 8. Bugs in committed code

1. **`anchor_ori` is pitch-blind in all four OSL tracking tasks** (`66f84ebec`).
   - The stock term compares projected-gravity z in the anchor frame. The OSL pelvis frame
     is x forward, y up, z lateral.
   - It never fires on pitch; it fires on roll only past about 53°.
   - Falls were still caught by `anchor_pos` and `ee_body_pos`; bent-over postures were
     not.
   - Fix: `bad_osl_anchor_ori` at 0.5 in `osl_fullbody_tracking_env_cfg`.
2. **The servo ctrlrange clamp** in `_reparameterize_servos` is the torque shortfall
   described in decision 4.
3. **The self-collision reward was removed** on the premise that self-collision is "off by
   construction". In fact the 46 explicit contact pairs are active.

## 9. Method and evidence

- **Round 1.** Six reviewers, each read-only, with tiny CPU/GPU probes:
  - msk-mimic-compare;
  - sesteban-unicycle;
  - mjlab-gait-library-port;
  - fingers;
  - interface-contract;
  - library-data-verify.
- **Round 2.**
  - Verifiers: anchor_ori confirmed; twist partially confirmed and reconciled (0.23-0.48
    was exp-of-mean, which the reward never pays); PD feasibility partially confirmed;
    library boundaries confirmed.
  - Three reward proposals: sesteban-faithful, MSK-first, prosthesis-goal.
  - A synthesis, then a numeric validation.
- **Scratch reports, scripts and outputs** are in
  `/tmp/claude-1000/-home-dmma-src-00-AMBER-mjlab/743dd65c-1055-45df-800b-0bcd8ab61c80/scratchpad/`
  (`wf1/`, `wf2/`, one folder per agent). They live in `/tmp` and may not survive a reboot.
- **Agent memories written or corrected:**
  - `osl-gait-library-reward-design`;
  - `osl-msk-finger-removal-plan`;
  - `osl-library-interface-contract`;
  - `osl-pelvis-frame-anchor-ori-trap`;
  - `osl-msk-tracking-failure-diagnosis`;
  - `clipper-trajectory-conversion`;
  - `sesteban-gait-library-fork`;
  - `osl-tracking-phase2`;
  - `osl-gait-library-walk-fwd-task`;
  - `osl-two-stage-rl-plan`.
