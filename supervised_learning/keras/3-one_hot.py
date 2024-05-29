#!/usr/bin/env python3
""" One Hot Module """


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Parameters:
    - labels: array-like, label vector to convert
    - classes: int, the number of classes (optional)

    Returns:
    - one_hot_matrix: the one-hot matrix
    """
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_matrix
