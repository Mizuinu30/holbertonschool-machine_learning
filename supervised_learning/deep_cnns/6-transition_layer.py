#!/usr/bin/env python3
"""transition layer"""


from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in Densely Connected
    Convolutional Networks"""
    # 1x1 Convolution
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    nb_filters = int(nb_filters * compression)
    # 1x1 Convolution
    X = K.layers.Conv2D(nb_filters, (1, 1),
                        padding='same',
                        kernel_initializer=K.initializers.he_normal(
        seed=0))(X)
    # Average Pooling
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)
    # Return the output of the transition layer and the number of filters
    return X, nb_filters
