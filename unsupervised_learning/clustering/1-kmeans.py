#!/usr/bin/env python3
"""K-means clustering module"""


import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a multivariate uniform distribution.

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer containing the number of clusters

    Returns:
    - numpy.ndarray of shape (k, d) containing the initialized centroids for each cluster
    - None on failure (e.g., if k is not a positive integer or X is not a valid array)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2 or not isinstance(k, int) or k <= 0:
        return None

    # Find the minimum and maximum values for each dimension
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # Generate k random centroids within the specified range
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, X.shape[1]))

    return centroids

def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer containing the number of clusters
    - iterations: positive integer containing the maximum number of iterations

    Returns:
    - C: numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    - clss: numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to
    - (None, None) on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    # Initialize variables
    prev_centroids = np.copy(centroids)
    for i in range(iterations):
        # Step 1: Assign clusters
        # Compute the distance from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Assign each point to the nearest centroid
        clss = np.argmin(distances, axis=1)

        # Step 2: Update centroids
        for j in range(k):
            if np.any(clss == j):
                # Update centroid to mean of assigned points
                centroids[j] = X[clss == j].mean(axis=0)
            else:
                # Reinitialize centroid if it has no points
                centroids[j] = np.random.uniform(X.min(axis=0), X.max(axis=0))

        # Step 3: Check for convergence
        if np.allclose(centroids, prev_centroids):
            break
        prev_centroids = np.copy(centroids)

    return centroids, clss
