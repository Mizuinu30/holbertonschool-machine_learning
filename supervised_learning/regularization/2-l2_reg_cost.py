#!/usr/bin/env python3
"""L2 regularization in tensorflow"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """ calculates the cost of a neural network with L2 regularization"""

    l2_losses = tf.add_n(model.losses)
    total_cost = cost + l2_losses
    return total_cost
