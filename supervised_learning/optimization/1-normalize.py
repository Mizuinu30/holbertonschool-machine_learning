#!/usr/bin/env python3
""" Module to compute the norm constants for the 1-norm. """


import numpy as np


def normalize(X, m, s):
    """
    Normalize a matrix

    Parameters:
    X (numpy.ndarray): The input matrix of shape (d, nx) to normalize
    m (numpy.ndarray): The mean of all features of X
    s (numpy.ndarray): The standard deviation of all features of X

    Returns:
    numpy.ndarray: The normalized X matrix
    """
    X_normalized = (X - m) / s
    return X_normalized
