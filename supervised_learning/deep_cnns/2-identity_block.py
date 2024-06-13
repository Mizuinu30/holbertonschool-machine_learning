#!/usr/bin/env python3
"""
Deep CNNs Module
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    builds an identity block as described in Deep Residual Learning for Image
    Recognition(2015)
    """
    init = K.initializers.he_normal()
    activation = 'relu'
    F11, F3, F12 = filters
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                            kernel_initializer=init)(A_prev)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(batch1)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                            kernel_initializer=init)(relu1)
    batch2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation('relu')(batch2)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                            kernel_initializer=init)(relu2)
    batch3 = K.layers.BatchNormalization(axis=3)(conv3)
    add = K.layers.Add()([batch3, A_prev])
    final_relu = K.layers.Activation('relu')(add)
    return final_relu
