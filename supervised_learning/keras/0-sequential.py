#!/usr/bin/env python3
""" a function that builds a neural network with the Keras library"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ a function that builds a neural network with the Keras library"""
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(layers.Dense(layers[i], activation=activations[i],
                                   kernel_regularizer=L2, input_shape=(nx,)))
        else:
            model.add(layers.Dropout(1 - keep_prob))
            model.add(layers.Dense(layers[i], activation=activations[i],
                                   kernel_regularizer=L2))
    return model
