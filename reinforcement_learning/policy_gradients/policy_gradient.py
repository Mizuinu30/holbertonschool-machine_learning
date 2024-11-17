#!/usr/bin/env python3
import numpy as np

def policy(matrix, weight):
    """
    Computes the policy using a weight matrix and a given state matrix.
    Uses the softmax function to compute the probability distribution.

    Args:
        matrix (ndarray): State matrix.
        weight (ndarray): Weight matrix.

    Returns:
        ndarray: Probability distribution for actions.
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)

def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and weight matrix.

    Args:
        state (ndarray): Current state of the environment (1D array).
        weight (ndarray): Weight matrix for computing policy.

    Returns:
        int: Selected action.
        ndarray: Gradient for the policy.
    """
    # Compute policy probabilities
    probs = policy(state[np.newaxis, :], weight)

    # Sample an action based on the probability distribution
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Compute the gradient
    d_softmax = np.diag(probs[0]) - np.outer(probs[0], probs[0])
    gradient = state[:, np.newaxis] * d_softmax[:, action]

    return action, gradient
