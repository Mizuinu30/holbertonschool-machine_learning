#!/usr/bin/env python3
""" This module contains the epsilon_greedy function """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ returns the action with the highest Q-value """

    if np.random.uniform(0, 1) < epsilon:
        # Explore
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit
        action = np.argmax(Q[state])

    return action

def train(
    env,
    Q,
    episodes=1000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    min_epsilon=0.01,
    epsilon_decay=0.995,
):
    """ Trains a Q-table using the Q-learning algorithm. """

    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        rewards_current_episode = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _, info = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )
            state = new_state
            rewards_current_episode += reward
            if done:
                break
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_rewards.append(rewards_current_episode)
    return Q, total_rewards
