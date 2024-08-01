#!/usr/bin/env python3
"""K-means clustering"""


import numpy as np

def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    return np.random.uniform(X.min(axis=0), X.max(axis=0), (k, d))

def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters:
    X (numpy.ndarray): Dataset of shape (n, d)
    k (int): Number of clusters
    iterations (int): Maximum number of iterations

    Returns:
    C (numpy.ndarray): Centroid means for each cluster
    clss (numpy.ndarray): Index of the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
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
