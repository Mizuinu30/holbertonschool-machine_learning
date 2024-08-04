#!/usr/bin/env python3
"""This modlue contains the function maximization(X, g)
"""
import numpy as np


def maximization(X, g):
    """This function calculates the maximization step in the
    EM algorithm fr a GMM
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape

    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None
    k = g.shape[0]
    if g.shape[0] != k:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), np.ones(n,)).all():
        return None, None, None

    pi = np.sum(g, axis=1) / n
    m = np.matmul(g, X) / np.sum(g, axis=1).reshape(-1, 1)

    S = np.zeros((k, d, d))
    for cluster in range(k):
        X_m = X - m[cluster]
        S[cluster] = np.dot(g[cluster] * X_m.T, X_m) / np.sum(g[cluster])

    return pi, m, S
