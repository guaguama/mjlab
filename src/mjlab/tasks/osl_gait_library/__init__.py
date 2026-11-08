"""Twist-commanded imitation over a looping OSL gait library.

One network with full observability drives the model over a library of looping
strides plus an idle pose: either the robot model (27 position servos and the two OSL
motors) or the musculoskeletal model (339 muscles and the two OSL motors), with
MimicReward or CLFReward. Each is the matching imitation task of
:mod:`mjlab.tasks.osl_mimic` (same simulation, network, PPO, observations and
imitation reward), with sesteban's G1 unicycle command in place of the single clip,
a stride-averaged twist reward, the commanded twist and the stride phase as extra
observations, and 10 s episodes.

Two variants: round 1's forward-walking tasks (``Mjlab-Gait-Library-OSL-Fullbody-*``),
and the unicycle tasks (``Mjlab-Gait-Library-Unicycle-OSL-Fullbody-*``) over the full
library with domain randomization. Both are stage 1 of the plan, the unified policy
that stage 2 splits into separate human and OSL actors.

``mdp`` holds the library command and the added terms, events and curriculum;
``config`` the task registration and randomization; ``rl`` the runner's validation
over the library.
"""
