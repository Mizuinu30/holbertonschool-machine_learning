#!/usr/bin/env python3
"""This modle  test fr the optimum number
of clusters by variance
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """This function tests fr the optimum number of clusters by variance
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin <= 0 or n <= kmin:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or n < kmax:
        return None, None
    if kmin >= kmax:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []
    d_vars = []

    for k in range(kmin, kmax + 1):

        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        V = variance(X, C)
        variances.append(V)

    for var in variances:
        d_vars.append(np.abs(variances[0] - var))

    return results, d_vars
