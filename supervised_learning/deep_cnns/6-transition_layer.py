#!/usr/bin/env python3
"""transition layer"""


from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in Densely Connected
    Convolutional Networks

    Args:
        X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        compression is the compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of filters within
        the output, respectively
    """
    # He et. al initialization
    init = K.initializers.he_normal()
    # Batch Normalization
    batch_norm = K.layers.BatchNormalization()(X)
    # ReLU activation function
    activation = K.layers.Activation('relu')(batch_norm)
    # 1x1 Convolution
    nb_filters = int(nb_filters * compression)

    conv = K.layers.Conv2D(filters=nb_filters,
                           kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer=init)(activation)
    # Average Pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         strides=(2, 2))(conv)
    # Return the output of the transition layer and the number of filters
    return avg_pool, nb_filters
