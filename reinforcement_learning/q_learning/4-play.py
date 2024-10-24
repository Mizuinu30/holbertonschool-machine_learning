#!/usr/bin/env python3
"""
Defines function that has trained agent play an episode
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Has trained agent play an episode

    returns:
        total rewards for the episode
    """
    state, _ = env.reset()
    done = False
    total_reward = 0
    rendered_outputs = []
    rendered_outputs.append(env.render())
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, truncated, _ = env.step(action)
        rendered_outputs.append(env.render())
        state = new_state
        total_reward += reward
        if done or truncated:
            break
    return total_reward, rendered_outputs
