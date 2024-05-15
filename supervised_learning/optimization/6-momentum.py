#!/usr/bin/env python3
""" Module that calculates the momentum of a data set."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with
    momentum optimization algorithm in TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight.

    Returns:
    optimizer: A TensorFlow optimizer configured
    with the given learning rate and momentum.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
