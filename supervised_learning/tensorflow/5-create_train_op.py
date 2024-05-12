#!/usr/bin/env python3
""" function that creates the trainig operation for a network"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """loss is the loss of network's prediction
    alpha is the learning rate
    Returns: an operation that trains the network using
    gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
