#!/usr/bin/env python3
"""
Deep CNNs Module
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning for Image Recognition (2015)

    Args:
        A_prev (tensor): output from the previous layer
        filters (tuple or list): contains F11, F3, F12
            - F11 is the number of filters in the first 1x1 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F12 is the number of filters in the second 1x1 convolution
        s (int): stride to be used for the first convolution in both the main and shortcut paths

    Returns:
        tensor: the activated output of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)

    # First component of the main path
    X = K.layers.Conv2D(F11, (1, 1), strides=s, padding='same',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of the main path
    X = K.layers.Conv2D(F3, (3, 3), strides=(
        1, 1), padding='same', kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of the main path
    X = K.layers.Conv2D(F12, (1, 1), strides=(
        1, 1), padding='same', kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    X_shortcut = K.layers.Conv2D(
        F12, (1, 1), strides=s, padding='same', kernel_initializer=init)(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add shortcut to the main path
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
