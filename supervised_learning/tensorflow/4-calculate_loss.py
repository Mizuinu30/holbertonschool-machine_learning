#!/usr/bin/env python3
""" function that calculates loss"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """ calculates loss
    y is a placeholder for labes of input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the loss of the prediction
    """

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
