#!/usr/bin/env python3
"""
This module contains a function that conducts propagation using
Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {'A0': X}
    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]

        Z = np.dot(W, A_prev) + b

        if layer == L:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)

            D = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A = np.multiply(A, D) / keep_prob
            cache['D' + str(layer)] = D

        cache['A' + str(layer)] = A

    return cache
