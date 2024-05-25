#!/usr/bin/env python3
"""L2 regularization in tensorflow"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """ calculates the cost of a neural network with L2 regularization"""

    l2_losses = []

    # Iterate over each layer to get L2 losses
    for layer in model.layers:
        if layer.losses:  # Check if the layer has L2 regularization losses
            l2_loss = tf.reduce_sum(layer.losses)
            l2_losses.append(l2_loss)

    # Convert the list of L2 losses to a tensor
    l2_losses_tensor = tf.convert_to_tensor(l2_losses)

    # Calculate total costs by adding the original cost to each L2 loss
    total_costs = cost + l2_losses_tensor

    return total_costs
