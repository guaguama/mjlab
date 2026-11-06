"""Twist-commanded imitation over a looping OSL gait library.

One network with full observability drives the model over a library of
forward-walking strides plus an idle pose: either the robot model (27 position
servos and the two OSL motors) with MimicReward, or the musculoskeletal model (339
muscles and the two OSL motors) with CLFReward. Each is the matching imitation task
of :mod:`mjlab.tasks.osl_mimic` (same simulation, network, PPO, observations and
imitation reward), with sesteban's G1 unicycle command in place of the single clip,
a stride-averaged twist reward, the commanded twist and the stride phase as extra
observations, and 10 s episodes.

``mdp`` holds the library command and the added terms; ``config`` the task
registration; ``rl`` the runner's validation over the library.
"""
