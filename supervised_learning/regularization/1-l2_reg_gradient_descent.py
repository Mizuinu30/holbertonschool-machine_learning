#!/usr/bin/env python3
""" module l2_reg_gradient_descent """


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural
    network using gradient descent"""

    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)] if i > 1 else cache['A0']
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        # Calculate gradients
        dW = (np.dot(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update parameters
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db

        if i > 1:
            # Calculate dZ for the next layer
            dZ = np.dot(W.T, dZ) * (1 - cache['A' + str(i - 1)] ** 2)

    return None
