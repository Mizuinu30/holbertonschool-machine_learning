#!/usr/bin/env python3
"""Identity Block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in Deep Residual Learning for Image Recognition (2015)

    Args:
        A_prev (tensor): output from the previous layer
        filters (tuple or list): contains F11, F3, F12
            - F11 is the number of filters in the first 1x1 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F12 is the number of filters in the second 1x1 convolution

    Returns:
        tensor: the activated output of the identity block
    """
    F11, F3, F12 = filters

    # Save the input value for the shortcut
    X_shortcut = A_prev

    # First component of main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                        strides=(1, 1), padding='valid',
                        kernel_initializer=K.initializers.he_normal(seed=0))(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                        strides=(1, 1), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                        strides=(1, 1), padding='valid',
                        kernel_initializer=K.initializers.he_normal(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut value to main path, and pass it through a ReLU activation
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
