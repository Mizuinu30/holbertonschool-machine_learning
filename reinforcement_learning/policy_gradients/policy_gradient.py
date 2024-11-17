#!/usr/bin/env python3
"Compute the policy using the weight matrix and the state matrix."

import numpy as np


def policy(matrix, weight):
    """
    Compute the policy using the weight matrix and the state matrix.

    Args:
        matrix (np.ndarray): The state matrix (1 x n).
        weight (np.ndarray): The weight matrix (n x m).

    Returns:
        np.ndarray: The policy distribution (1 x m).
    """

    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))  # Avoid overflow by subtracting max
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
