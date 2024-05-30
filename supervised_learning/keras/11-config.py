#!/usr/bin/env python3
""" Config Module """
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model configuration in JSON format """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    f.close()
    return None


def load_config(filename):
    """  loads a model with a specific configuration """
    with open(filename, 'r') as f:
        model = K.models.model_from_json(f.read())
    f.close()
    return model
