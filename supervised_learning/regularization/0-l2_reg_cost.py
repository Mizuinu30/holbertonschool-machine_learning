#!/usr/bin/env python3
"""
module l2_reg_cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization
    """
    l2 = 0
    for idx in range(1, L+1):
        l2 += np.linalg.norm(weights['W' + str(idx)])
    return cost + (l2 * lambtha / (2 * m))
