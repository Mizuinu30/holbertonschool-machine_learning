#!/usr/bin/env python3
""" Weights Module """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ saves a model weights """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """ loads a model weights """
    network.load_weights(filename)
    return None
