#!/usr/bin/env python3
""" Have a function perform SARSA(λ) """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Determine next action to take using epsilon-greedy
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, Q.shape[1])
    else:
        return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """ Performs SARSA(λ)
    """
    fepsilon = epsilon

    for episode in range(episodes):
        es = np.zeros_like(Q)
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)

        for _ in range(max_steps):
            ns, r, term, trunc, _ = env.step(action)

            naction = epsilon_greedy(Q, ns, epsilon)
            δ = r + gamma * Q[ns, naction] - Q[state, action]
            es[state, action] += 1
            es *= lambtha * gamma
            Q += alpha * δ * es

            if term or trunc:
                break

            state = ns
            action = naction

            epsilon_range = fepsilon - min_epsilon
            decay_factor = np.exp(-epsilon_decay * episode)

        epsilon = min_epsilon + epsilon_range * decay_factor
    return Q
