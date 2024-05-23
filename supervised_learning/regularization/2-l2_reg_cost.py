#!/usr/bin/env python3
"""
module l2_reg_cost
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (tensor): A tensor containing the cost of the network without L2 regularization.
    model (Keras Model): A Keras model that includes layers with L2 regularization.

    Returns:
    tensor: A tensor containing the total cost for each layer of the network, accounting for L2 regularization.
    """
    l2_loss = sum(layer.losses for layer in model.layers)
    total_cost = cost + l2_loss
    return total_cost
