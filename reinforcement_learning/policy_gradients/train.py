#!/usr/bin/env python3
import numpy as np
from policy_gradient import policy_gradient

def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Train a policy gradient agent.

    Args:
        env: The environment.
        nb_episodes (int): Number of episodes for training.
        alpha (float): Learning rate.
        gamma (float): Discount factor.

    Returns:
        list: Scores for each episode.
    """
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_gradients = []

        done = False
        while not done:
            action, gradient = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_rewards.append(reward)
            episode_gradients.append(gradient)
            state = next_state

        # Update weights
        for t in range(len(episode_rewards)):
            Gt = sum(gamma ** i * r for i, r in enumerate(episode_rewards[t:]))
            weight += alpha * Gt * episode_gradients[t]

        score = sum(episode_rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    return scores
