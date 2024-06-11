#!/usr/bin/env python3
"""
Deep CNNs Module
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block as described in Going Deeper with
    Convolutions(2014)
    Args:
        A_prev (ndarray): output from the previous layer
        filters (tuple or list): contains F1, F3R, F3, F5R, F5, FPP
    Returns:
        concatenated output of the inception block
    """
    activation = 'relu'
    init = K.initializers.he_normal(seed=None)
    F1, F3R, F3, F5R, F5, FPP = filters
    convly_1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                               activation=activation,
                               kernel_initializer=init)(A_prev)
    convly_2P = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                                activation=activation,
                                kernel_initializer=init)(A_prev)
    convly_2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                               activation=activation,
                               kernel_initializer=init)(convly_2P)
    convly_3P = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                                activation=activation,
                                kernel_initializer=init)(A_prev)
    convly_3 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                               activation=activation,
                               kernel_initializer=init)(convly_3P)
    layer_pool = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(1, 1),
                                       padding='same')(A_prev)
    layer_poolP = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                                  activation=activation,
                                  kernel_initializer=init)(layer_pool)
    mid_layer = K.layers.concatenate([convly_1, convly_2, convly_3,
                                      layer_poolP])
    return mid_layer
