#!/usr/bin/env python3
"""
    Forward propagation graph for a neural network
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network
    Arguments:
        x (tf.placeholder): input data placeholder
        layer_sizes (List[int]): number of nodes in each layer
        activations (List[Callable]): activation functions for each layer
    Returns:
        tf.Tensor: prediction of the network in tensor form
    """
    layer = x
    for size, activation in zip(layer_sizes, activations):
        layer = create_layer(layer, size, activation)
    return layer
