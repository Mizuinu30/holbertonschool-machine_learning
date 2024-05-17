#!/usr/bin/env python3
""" Module to normalize a matrix using batch normalization. """


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a 
    neural network using batch normalization.

    Parameters:
    Z (numpy.ndarray): Shape (m, n) that should be normalized.
        m (int): The number of data points.
        n (int): The number of features in Z.
    gamma (numpy.ndarray): Shape (1, n) containing
    the scales used for batch normalization.
    beta (numpy.ndarray): Shape (1, n) containing
    the offsets used for batch normalization.
    epsilon (float): A small number used to
    avoid division by zero.

    Returns:
    numpy.ndarray: The normalized Z matrix.
    """
    # Compute the mean of Z
    mean = np.mean(Z, axis=0)

    # Compute the variance of Z
    variance = np.var(Z, axis=0)

    # Normalize Z
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift the normalized Z
    Z_batch_norm = gamma * Z_normalized + beta

    return Z_batch_norm
