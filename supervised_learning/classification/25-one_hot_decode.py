#!/usr/bin/env python3
""" Modile that converts a one-hot matrix into a numeric label vector """
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a numeric label vector
    Args:
        one_hot: a one-hot encoded numpy.ndarray with shape (classes, m)
                where classes is the maximum number of classes
                m is the number of examples
    Returns: a numpy.ndarray with shape (m,) containing the numeric labels
            for each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    else:
        return np.argmax(one_hot, axis=0)
