#!/usr/bin/env python3
"""L2 regularization in tensorflow"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a dense layer with L2 regularization.

    Args:
    prev: tensor, the output of the previous layer.
    n: int, number of nodes in the new layer.
    activation: function, the activation function to be used.
    lambtha: float, the L2 regularization parameter.

    Returns:
    The output tensor of the new layer.
    """
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=init,
                             kernel_regularizer=regularizer)
    return tensor(prev)
