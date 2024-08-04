#!/usr/bin/env python3
"""This modlue contains the function maximization that performs the expectation
maximization fr a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """This functions performs the expectation maximization fr a GMM
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape

    l_prev = 0

    pi, m, S = initialize(X, k)

    for i in range(iterations + 1):
        if i != 0:
            l_prev = likelihood
            pi, m, S = maximization(X, g)
        g, likelihood = expectation(X, pi, m, S)

        if verbose:
            if i % 10 == 0 or i == iterations or np.abs(
                    likelihood - l_prev) <= tol:
                print(f"Log Likelihood after {i} iterations: {likelihood:.5f}")
        if np.abs(likelihood - l_prev) < tol:
            break
        l_prev = likelihood

    return pi, m, S, g, likelihood
