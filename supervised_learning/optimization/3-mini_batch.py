#!/usr/bin/env python3
""" Module to create mini-batches. """


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from input data and labels.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx), where m is the number of data points and nx is the number of features.
        Y (numpy.ndarray): Labels of shape (m, ny), where m is the number of data points and ny is the number of classes.
        batch_size (int): Number of data points in a batch.

    Returns:
        list: List of mini-batches, each mini-batch is a tuple (X_batch, Y_batch).
    """
    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X, Y = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
