#!/usr/bin/env python3
""" Module to build a neural network with the Keras library. """


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Parameters:
    - nx: number of input features to the network
    - layers: list containing the number of nodes in each layer of the network
    - activations: list containing the activation functions used for each
    layer of the network
    - lambtha: L2 regularization parameter
    - keep_prob: probability that a node will be kept for dropout

    Returns:
    - model: the keras model
    """
    model = K.models.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[0], activation=activations[0],
                                     input_shape=(nx,), kernel_initializer=K.
                                     initializers.he_normal(),
                                     kernel_regularizer=K.
                                     regularizers.l2(lambtha)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer=K.regularizers.l2(lambtha)))
    return model
