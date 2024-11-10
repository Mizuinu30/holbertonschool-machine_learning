#!/usr/bin/env python3
"""This module contains the function for the SARSA(λ) algorithm."""
import numpy as np


def sarsa_lambtha(
    env,
    Q,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05,
):
    """
    Performs the SARSA(λ) algorithm to update the Q table.

    Args:
        env: The environment instance.
        Q: A numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: The eligibility trace factor.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.
        epsilon: The initial threshold for epsilon greedy.
        min_epsilon: The minimum value that epsilon should decay to.
        epsilon_decay: The decay rate for updating epsilon between episodes.

    Returns:
        Q: The updated Q table.
    """

    def epsilon_greedy(state, Q, epsilon):
        """Selects an action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    for episode in range(episodes):
        # Reset the environment to start a new episode
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        # Select the initial action using epsilon-greedy policy
        action = epsilon_greedy(state, Q, epsilon)

        # Initialize the eligibility trace to zeros
        E = np.zeros_like(Q)

        for step in range(max_steps):
            # Take the action and observe the next state,
            # reward, and whether the episode is done
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Select the next action using epsilon-greedy policy
            next_action = epsilon_greedy(next_state, Q, epsilon)

            # Calculate the TD error
            delta = reward + gamma * Q[next_state,
                                       next_action] - Q[state, action]

            # Update the eligibility trace for the current state-action pair
            E[state, action] += 1

            # Update the Q table and eligibility trace
            # for all state-action pairs
            Q += alpha * delta * E
            E *= gamma * lambtha

            # If the episode is terminated or truncated, end the episode
            if terminated or truncated:
                break

            # Move to the next state and action
            state = next_state
            action = next_action

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))

    return Q
