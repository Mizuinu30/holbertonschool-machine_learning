#!/usr/bin/env python3
""" Initializes cluster centroids for K-means. """

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
    X (numpy.ndarray): The dataset of shape (n, d) to
    be used for K-means clustering.
        n is the number of data points.
        d is the number of dimensions for each data point.
    k (int): A positive integer containing the number of clusters.

    Returns:
    numpy.ndarray: A numpy array of shape (k, d) containing
    the initialized centroids for each cluster,
                   or None on failure.
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0:
        return None

    if X.ndim != 2:
        return None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(min_vals, max_vals, (k, d))

    return centroids
