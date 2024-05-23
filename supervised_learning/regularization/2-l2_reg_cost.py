#!/usr/bin/env python3
"""
module l2_reg_cost
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    calculates the cost of a neural network with L2 regularization
    """
    reg_losses = tf.add_n(model.losses)
    total_cost = cost + reg_losses
    return total_cost
