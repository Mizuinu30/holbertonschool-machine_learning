#!/usr/bin/env python3
"""
Defines function that loads pre-made FrozenLakeEnv environment
from OpenAI's gym
"""


import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ loads pre-made FrozenLakeEnv environment from OpenAI's gym"""
    if desc is not None:
        # Load environment with custom map description
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)
    elif map_name is not None:
        # Load environment with a pre-made map
        env = gym.make(
            "FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    else:
        # Load a randomly generated 8x8 map if both desc and map_name are None
        env = gym.make(
            "FrozenLake-v1", map_name="8x8", is_slippery=is_slippery)

    return env
