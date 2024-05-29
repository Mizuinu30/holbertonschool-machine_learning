#!/usr/bin/env python3
""" Input Module """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with the Keras library """
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            output = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=L2)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=L2)(dropout)
    return K.models.Model(inputs=inputs, outputs=output)
