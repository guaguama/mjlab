"""musclemimic's OSL full-body imitation recipe, ported to mjlab.

``mdp`` holds the reference command, observations, reward (MimicReward and
CLFReward) and terminations; ``rl`` the network, optimizer, PPO variant and runner;
``config`` the environment and runner settings and the task registration.
"""
