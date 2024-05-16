#!/usr/bin/env python3
"""  
Module hat creates a learning rate decay
operation in tensorflow using inverse time decay
"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time decay.

    Parameters:
    alpha -- original learning rate
    decay_rate -- weight used to determine the rate at which alpha will decay
    decay_step -- number of passes of gradient descent that should occur before alpha is decayed further

    Returns:
    learning_rate -- the learning rate decay operation
    """
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)(global_step)
    return learning_rate
