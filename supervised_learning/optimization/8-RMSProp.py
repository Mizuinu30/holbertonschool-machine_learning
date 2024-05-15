#!/usr/bin/env python3
""" Module to create the RMSProp optimization operation in TensorFlow."""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the RMSProp optimization operation in TensorFlow.

    Parameters:
    loss -- the loss to minimize
    alpha -- learning rate
    beta2 -- RMSProp weight (discounting factor)
    epsilon -- small number to avoid division by zero

    Returns:
    optimizer -- the optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha, decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
