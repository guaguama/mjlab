"""Stage 2 of the OSL plan on the stairs clip: a learned human and a learned OSL actor.

Stairs stage 1 (:mod:`mjlab.tasks.osl_stairs`) trains one policy for the whole body
to follow one clip up a flight of stairs. Stage 2 splits it as unicycle stage 2
(:mod:`mjlab.tasks.osl_split`) splits the unicycle policy: the stage-1 actor minus
its two OSL rows becomes the human actor, a fresh OSL actor reads only the
prosthesis's own sensors, and two-agent PPO trains both, each on its own reward
channel (RL only; there is no FSM variant here).

Everything is shared with those two packages:

* the env is stairs stage 1's (the clip's terrain boxes, the blind stage-1
  observation as the human's, the gait-library physics randomization), turned into
  stage 2 by unicycle stage 2's own split stage, so the OSL actor's sensors, history,
  lags, noise and switches are unicycle stage 2's by construction;
* the agent settings are unicycle stage 2's (networks, learning rates, freeze,
  blend, channels, schedule), with stairs stage 1's episode handling and one-clip
  validation;
* the runner is unicycle stage 2's with its motion-library assumptions replaced by
  the clip's (evaluation envs on the training clip and terrain, one clip length of
  validation), the terrain copied into the run, and a warm start that must come from
  a stairs stage-1 run.

``config`` holds the env config, the runner settings and the task registration
(``Mjlab-Split-Stairs-OSL-Fullbody{-ROBOT,-MSK}[-CLF][-Compliant]``), ``rl`` the
runner. There is no export or deploy path: osl_deploy has no terrain, so the stairs
actors are judged in training (validation, stress validation, videos). The OSL actor
is still exported as ONNX with every checkpoint.
"""
