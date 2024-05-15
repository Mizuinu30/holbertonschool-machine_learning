#!/usr/bin/env python3
"""
module create_RMSProp_op
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """ Function that creates the RMSProp optimization operation in Tensorflow
    - alpha is the learning rate
    - beta2 is the RMSProp weight
    - epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
    return optimizer
