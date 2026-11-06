"""Twist-commanded imitation over a looping OSL gait library: round 1.

One network with full observability drives the musculoskeletal model, the 339
muscles and the two OSL motors, over a library of forward-walking strides plus an
idle pose. It is the MSK CLF imitation task of :mod:`mjlab.tasks.osl_mimic` (same
simulation, network, PPO, observations and CLF reward), with sesteban's G1 unicycle
command in place of the single clip, a stride-averaged twist reward, the commanded
twist and the stride phase as extra observations, and 10 s episodes.

``mdp`` holds the library command and the added terms; ``config`` the task
registration; ``rl`` the runner's validation over the library.
"""
