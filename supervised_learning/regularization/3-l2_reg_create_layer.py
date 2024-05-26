#!/usr/bin/env python3
""" L2 Regularization Cost """
import tensorflow as tf
from tensorflow.keras import layers


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ creates a tensorflow layer that includes L2 regularization
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        lambtha: L2 regularization parameter
        Returns: output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation, kernel_initializer=init,
                            kernel_regularizer=reg)
    return layer(prev)
