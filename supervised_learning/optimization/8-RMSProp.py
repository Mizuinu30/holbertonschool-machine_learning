#!/usr/bin/env python3
"""
module create_RMSProp_op
"""
import tensorflow as tf
import tensorflow.keras as keras


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm
    """
    op = tf.keras.optimizers.RMSprop(learning_rate=alpha, decay=beta2,
                                     epsilon=epsilon)
    step_op = op.get_updates(loss=None, params=None)
    return step_op
