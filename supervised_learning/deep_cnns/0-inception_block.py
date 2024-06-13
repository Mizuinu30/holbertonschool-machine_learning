#!/usr/bin/env python3
"""
Deep CNNs Module
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    Going Deeper with Convolutions (2014)

    Args:
        A_prev (ndarray): Output from the previous layer
        filters (tuple or list): Contains F1, F3R, F3, F5R, F5, FPP

    Returns:
        The concatenated output of the inception block
    """
    activation = 'relu'
    init = K.initializers.he_normal(seed=None)
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 Convolution branch
    conv_1x1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                               activation=activation,
                               kernel_initializer=init)(A_prev)

    # 1x1 Convolution followed by 3x3 Convolution branch
    conv_3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                                      activation=activation,
                                      kernel_initializer=init)(A_prev)
    conv_3x3 = K.layers.Conv2D(filters=F3, kernel_size=3,
                               padding='same', activation=activation,
                               kernel_initializer=init)(conv_3x3_reduce)

    # 1x1 Convolution followed by 5x5 Convolution branch
    conv_5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                                      activation=activation,
                                      kernel_initializer=init)(A_prev)
    conv_5x5 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                               activation=activation,
                               kernel_initializer=init)(conv_5x5_reduce)

    # Max pooling followed by 1x1 Convolution branch
    max_pool = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same')(A_prev)
    max_pool_conv = K.layers.Conv2D(filters=FPP, kernel_size=1,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=init)(max_pool)

    # Concatenate all the branches
    inception_output = K.layers.concatenate(
        [conv_1x1, conv_3x3, conv_5x5, max_pool_conv], axis=-1)

    return inception_output
