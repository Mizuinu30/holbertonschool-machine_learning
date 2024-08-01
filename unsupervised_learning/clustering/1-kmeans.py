#!/usr/bin/env python3
import numpy as np


def kmeans(X, k, iterations):
    """
    Perform K-means clustering on the dataset X.

    Args:
        X (np.ndarray): The dataset.
        k (int): The number of clusters.
        iterations (int): The maximum number of iterations.

    Returns:
        tuple: (C, clss) where C is the array of centroids and clss is the array of cluster assignments.
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0 or not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        new_C = np.array([X[clss == i].mean(axis=0) if np.any(clss == i) else np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), d) for i in range(k)])
        if np.all(C == new_C):
            break
        C = new_C

    return C, clss

def initialize(X, k):
    """
    Initialize centroids for K-means.

    Args:
        X (np.ndarray): The dataset.
        k (int): The number of clusters.

    Returns:
        np.ndarray: The initialized centroids.
    """
    n, d = X.shape
    return X[np.random.choice(n, k, replace=False)]