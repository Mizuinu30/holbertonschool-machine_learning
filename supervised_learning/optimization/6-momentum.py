#!/usr/bin/env python3
""" Module that calculates the momentum of a data set."""

import tensorflow as tf
import tensorflow.compat.v1 as tf


def create_momentum_op(alpha, beta1):
    """ Function that creates the training operation for a neural network
    in tensorflow using the gradient descent with momentum optimization algorithm.

    - alpha is the learning rate
    - beta1 is the momentum weight

    Returns: optmizer operation
    """
    optimizer = tf.compat.v1.train.MomentumOptimizer(alpha, beta1)
    return optimizer
