#!/usr/bin/env python3
""" Modile that converts a numeric label vector into a one-hot matrix """
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    Args:
        Y: a numpy.ndarray with shape (m,) containing numeric class labels
        classes: the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m), or None on
            failure
    """

    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None

    encoded_array = np.zeros((classes, Y.size), dtype=float)

    encoded_array[Y, np.arange(Y.size)] = 1

    return encoded_array
