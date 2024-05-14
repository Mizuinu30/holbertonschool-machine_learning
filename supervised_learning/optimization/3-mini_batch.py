#!/usr/bin/env python3
""" Module to create mini-batches. """


import numpy as np


def mini_batch(X, Y, batch_size):
    """ Create mini-batches from input data."""
    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X, Y = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
