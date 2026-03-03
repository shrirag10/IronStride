"""
IronStride — Humanoid Locomotion RL Environment Package.

Registers the custom Gymnasium environment for the Unitree H1 humanoid.
"""

import gymnasium as gym

gym.register(
    id="IronStrideEnv-v0",
    entry_point="ironstride.envs.ironstride_env:IronStrideEnv",
    max_episode_steps=1000,
)
