#!/usr/bin/env python3
""" Module to shuffle data points."""


import numpy as np


def shuffle_data(X, Y):
    """
    Shuffle the data points in two matrices the same way

    Parameters:
    X (numpy.ndarray): The first input matrix of shape (m, nx) to shuffle
    Y (numpy.ndarray): The second input matrix of shape (m, ny) to shuffle

    Returns:
    tuple: The shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
