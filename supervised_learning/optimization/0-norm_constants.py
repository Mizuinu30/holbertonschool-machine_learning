#!/usr/bin/env python3
""" Module to compute the norm constants for the 0-norm. """


import numpy as np


def normalization_constants(X):
    """ Calculate the normalization constants of a matrix.

    Parameters:
    X (numpy.ndarray): The matrix to calculate the norm constants.

    Returns:
    Tuple: The mean and the standard of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
