#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    return np.random.uniform(X.min(axis=0), X.max(axis=0), (k, d))

def kmeans(X, k, iterations=1000):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    centroids = initialize(X, k)
    if centroids is None:
        return None, None
    prev_centroids = np.copy(centroids)
    for i in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)
        for j in range(k):
            if np.any(clss == j):
                centroids[j] = X[clss == j].mean(axis=0)
            else:
                centroids[j] = np.random.uniform(X.min(axis=0), X.max(axis=0))
        if np.allclose(centroids, prev_centroids):
            break
        prev_centroids = np.copy(centroids)
    return centroids, clss
