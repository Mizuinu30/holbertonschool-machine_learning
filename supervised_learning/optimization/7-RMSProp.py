#!/usr/bin/env python3
""" Module that calculates the RMSProp optimization algorithm."""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    alpha (float): The learning rate.
    beta2 (float): The RMSProp weight.
    epsilon (float): A small number to avoid division by zero.
    var (numpy.ndarray): A numpy array containing the variable to be updated.
    grad (numpy.ndarray): A numpy array containing the gradient of var.
    s (numpy.ndarray): The previous second moment of var.

    Returns:
    tuple: The updated variable and the new moment, respectively.
    """
    s = beta2 * s + (1 - beta2) * np.square(grad)
    var_updated = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var_updated, s
