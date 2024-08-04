#!/usr/bin/env python3
"""This modulle contains the function expectation(X, pi, m, S)
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """This function calculates the expectation step in the
    EM algorithm fr a GMM
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape

    if pi.shape[0] > n:
        return None, None
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    pos = np.zeros((k, n))

    for cluster in range(k):

        PDF = pdf(X, m[cluster], S[cluster])

        pos[cluster] = pi[cluster] * PDF

    sum_pos = np.sum(pos, axis=0, keepdims=True)
    pos /= sum_pos

    li = np.sum(np.log(sum_pos))

    return pos, li
