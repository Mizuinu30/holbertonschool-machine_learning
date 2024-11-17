#!/usr/bin/env python3
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
