#!/usr/bin/env python3
""" L2 Regularization Cost """
import tensorflow as tf
from tensorflow.keras import layers, regularizers


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tensorflow layer that includes L2 regularization
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg', distribution='untruncated_normal')

    layer = layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(lambtha)
    )
    return layer(prev)
