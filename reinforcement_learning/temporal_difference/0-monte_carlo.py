#!/usr/bin/env python3
"""
Monte Carlo policy evaluation for the FrozenLake8x8 environment.
This module provides a function to estimate the state-value function
for a given policy using a Monte Carlo approach.
"""

import numpy as np


def monte_carlo(env, V, policy, gamma=0.99, episodes=10000):
    """
    Estimates the state-value function using Monte Carlo method.

    Args:
        env: Gymnasium environment.
        V (np.array): Initial value function.
        policy (function): Policy function for choosing actions.
        gamma (float): Discount factor.
        episodes (int): Number of episodes for evaluation.

    Returns:
        np.array: Updated value function.
    """
    # Initialize return tracking for each state
    returns = {s: [] for s in range(env.observation_space.n)}

    for _ in range(episodes):
        # Generate an episode
        episode = []
        state = env.reset()[0]
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, reward))
            state = next_state

        # Calculate returns and update V
        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            if state not in [x[0] for x in episode[:-1]]:
                returns[state].append(G)
                V[state] = np.mean(returns[state])

    return V
