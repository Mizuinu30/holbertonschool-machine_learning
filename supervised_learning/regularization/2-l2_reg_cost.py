#!/usr/bin/env python3
"""L2 regularization in tensorflow"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """ calculates the cost of a neural network with L2 regularization"""

    l2_losses = [tf.reduce_sum(layer.losses)
                 for layer in model.layers if layer.losses]

    # Add the L2 losses to the original cost
    total_cost = tf.constant([cost + l2_loss for l2_loss in l2_losses])

    return total_cost
