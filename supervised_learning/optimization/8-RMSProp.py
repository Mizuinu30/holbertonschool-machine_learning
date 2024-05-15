#!/usr/bin/env python3
""" Module to create the RMSProp optimizer in TensorFlow."""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the RMSProp optimizer in TensorFlow 2.x.

    Parameters:
    loss -- the loss function
    alpha -- learning rate
    beta2 -- RMSProp weight (discounting factor)
    epsilon -- small number to avoid division by zero

    Returns:
    optimizer -- the configured RMSProp optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
    return optimizer
