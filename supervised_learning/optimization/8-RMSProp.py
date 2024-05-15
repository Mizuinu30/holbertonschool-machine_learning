#!/usr/bin/env python3
""" Module to create the RMSProp optimizer in TensorFlow."""


import tensorflow as tf
import tensorflow.keras as keras


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates the RMSProp optimizer in TensorFlow 2.x and uses it to minimize the loss.

    Parameters:
    loss -- the loss function
    alpha -- learning rate
    beta2 -- RMSProp weight (discounting factor)
    epsilon -- small number to avoid division by zero

    Returns:
    train_op -- the operation that minimizes the loss
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
    train_op = optimizer.minimize()
    return train_op
