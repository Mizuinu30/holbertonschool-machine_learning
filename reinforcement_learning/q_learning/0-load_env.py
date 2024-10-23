#!/usr/bin/env python3
"""
Defines function that loads pre-made FrozenLakeEnv environment
from OpenAI's gym
"""


import gymnasium as gym

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads pre-made FrozenLakeEnv environment from OpenAI's gym

    returns:
        the environment
    """
    ENV = gym.make(
        'FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery,
          render_mode="ansi")

    return ENV
