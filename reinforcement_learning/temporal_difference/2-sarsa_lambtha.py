#!/usr/bin/env python3
"""This module contains the function for the SARSA(λ) algorithm."""


import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) on a given environment using eligibility traces.

    Parameters:
    env -- the environment instance
    Q -- a numpy.ndarray of shape (s, a) containing the Q table
    lambtha -- the eligibility trace factor
    episodes -- the total number of episodes to train over
    max_steps -- the maximum number of steps per episode
    alpha -- the learning rate
    gamma -- the discount rate
    epsilon -- the initial threshold for epsilon greedy
    min_epsilon -- the minimum value that epsilon should decay to
    epsilon_decay -- the decay rate for updating epsilon between episodes

    Returns:
    Q -- the updated Q table
    """
    # Initialize number of states and actions
    num_states, num_actions = Q.shape

    # Loop through episodes
    for episode in range(episodes):
        # Reset the environment to start a new episode
        state = env.reset()

        # Initialize eligibility trace for each state-action pair to zero
        E = np.zeros((num_states, num_actions))

        # Select the first action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(Q[state])

        # Loop for each step within an episode
        for step in range(max_steps):
            # Take action, observe reward and next state
            next_state, reward, done, _ = env.step(action)

            # Choose the next action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                next_action = np.random.randint(num_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # Calculate TD error (delta)
            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Update eligibility trace for the current state-action pair
            E[state, action] += 1

            # Update Q values and eligibility traces for all state-action pairs
            Q += alpha * td_error * E
            E *= gamma * lambtha  # Decay eligibility traces

            # Move to the next state and action
            state = next_state
            action = next_action

            # Break if episode is finished
            if done:
                break

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

    return Q
