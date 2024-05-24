#!/usr/bin/env python3
"""
module l2_reg_gradient_descent
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network using gradient"""
    m = Y.shape[1]
    A_prev = cache[f'A{L-1}']
    dZ = cache[f'A{L}'] - Y

    for l in range(L, 0, -1):
        A_prev = cache[f'A{l-1}'] if l > 1 else cache['A0']
        W = weights[f'W{l}']
        b = weights[f'b{l}']

        dW = (np.matmul(dZ, A_prev.T) + lambtha * W) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights[f'W{l}'] = W - alpha * dW
        weights[f'b{l}'] = b - alpha * db

        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)
