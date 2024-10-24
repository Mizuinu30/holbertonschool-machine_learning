#!/usr/bin/env python3
"""
Q-learning algorithm for training on the FrozenLake environment
"""

import numpy as np


def train(
    env,
    Q,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05):
    """
    Trains an agent using Q-learning on a FrozenLakeEnv environment.

    Parameters:
    - env: The FrozenLakeEnv instance
    - Q: A numpy.ndarray containing the Q-table
    - episodes: The total number of episodes to train over (default: 5000)
    - max_steps: The maximum number of steps per episode (default: 100)
    - alpha: The learning rate (default: 0.1)
    - gamma: The discount factor (default: 0.99)
    - epsilon: The initial epsilon value for the epsilon-greedy strategy (default: 1)
    - min_epsilon: The minimum value for epsilon (default: 0.1)
    - epsilon_decay: The decay rate for epsilon after each episode (default: 0.05)

    Returns:
    - Q: The updated Q-table
    - total_rewards: A list containing the rewards per episode
    """
    epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_rewards = 0

        for step in range(max_steps):
            # Choose an action using epsilon-greedy strategy
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action, observe next state and reward
            next_state, reward, done, info = env.step(action)

            # If the agent falls in a hole, set the reward to -1
            if done and reward == 0:
                reward = -1

            # Update Q-table using the Q-learning formula
            best_next_action = np.argmax(Q[next_state, :])
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

            # Update the state
            state = next_state
            episode_rewards += reward

            # If the episode ends, break the loop
            if done:
                break

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
        total_rewards.append(episode_rewards)

    return Q, total_rewards
