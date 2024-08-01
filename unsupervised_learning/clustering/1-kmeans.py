#!/usr/bin/env python3
""" Performs K-means clustering on a dataset. """

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
    X (numpy.ndarray): The dataset of shape (n, d) to be used for K-means clustering.
        n is the number of data points.
        d is the number of dimensions for each data point.
    k (int): A positive integer containing the number of clusters.

    Returns:
    numpy.ndarray: A numpy array of shape (k, d) containing the initialized centroids for each cluster,
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

def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    X (numpy.ndarray): The dataset of shape (n, d) containing the dataset.
        n is the number of data points.
        d is the number of dimensions for each data point.
    k (int): A positive integer containing the number of clusters.
    iterations (int): A positive integer containing the maximum number of iterations that should be performed.

    Returns:
    numpy.ndarray: A numpy array of shape (k, d) containing the centroid means for each cluster.
    numpy.ndarray: A numpy array of shape (n,) containing the index of the cluster in C that each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0 or not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        # Calculate distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        new_C = np.array([X[clss == i].mean(axis=0) if np.any(clss == i) else np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), d) for i in range(k)])

        # Check for convergence
        if np.all(C == new_C):
            break

        C = new_C

    return C, clss
