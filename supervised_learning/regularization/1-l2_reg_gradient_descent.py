#!/usr/bin/env python3
"""
module l2_reg_gradient_descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient descent
    with L2 regularization
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for idx in range(L, 0, -1):
        A = cache['A' + str(idx - 1)]
        W = weights['W' + str(idx)]
        b = weights['b' + str(idx)]
        dw = np.matmul(dz, A.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dz = np.matmul(W.T, dz) * (1 - A * A)
        weights['W' + str(idx)] = W - alpha * (dw + (lambtha/m) * W)
        weights['b' + str(idx)] = b - alpha * db
