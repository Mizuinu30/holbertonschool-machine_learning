#!/usr/bin/env python3
""" Policy Gradient """


import numpy as np


def policy(matrix, weight):
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)

def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient.

    Args:
        state (np.ndarray): The current state.
        weight (np.ndarray): The weight matrix.

    Returns:
        tuple: The action taken and its gradient.
    """
    probs = policy(state[None, :], weight)  # Add batch dimension
    action = np.random.choice(len(probs[0]), p=probs[0])
    d_softmax = probs.copy()
    d_softmax[0, action] -= 1  # Subtract 1 from chosen action's probability
    gradient = np.dot(state[:, None], d_softmax)
    return action, gradient
