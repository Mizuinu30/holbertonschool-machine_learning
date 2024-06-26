#!/usr/bin/env python3
"""Module that that updates the learning rate
using inverse time decay in numpy"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.

    Parameters:
    alpha (float): The original learning rate.
    decay_rate (float): The weight used to
    determine the rate at which alpha will decay.
    global_step (int): The number of passes of
    gradient descent that have elapsed.
    decay_step (int): The number of passes of
    gradient descent that should occur before alpha is
    decayed further.

    Returns:
    float: The updated value for alpha.
    """
    # Compute the learning rate decay
    decayed_learning_rate = alpha / \
        (1 + decay_rate * np.floor(global_step / decay_step))

    return decayed_learning_rate
