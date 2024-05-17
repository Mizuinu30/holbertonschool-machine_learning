#!/usr/bin/env python3
""" Module to create a batch normalization layer. """


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural
    network in TensorFlow.

    Parameters:
    prev (tf.Tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (function): The activation function
    that should be used on the output of the layer.

    Returns:
    tf.Tensor: A tensor of the activated output for the layer.
    """
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg')
    )

    Z = dense_layer(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')

    epsilon = 1e-7
    Z_batch_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon)

    if activation is not None:
        return activation(Z_batch_norm)
    else:
        return Z_batch_norm
