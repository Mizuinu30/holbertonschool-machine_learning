#!/usr/bin/env python3
"""Script that creates a tf layer that
    includes l2 regularization
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tf layer that"""
    layer = layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizers.l2(lambtha)
    )
    return layer(prev)
