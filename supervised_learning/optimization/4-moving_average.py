#!/usr/bin/env python3
""" Module that calculates the weigthed moving
average of a data set.
"""


import numpy as np


def moving_average(data, beta):
    """ Function that calculates the weigthed moving
    average of a data set.

    - data is the list of data to calculate the moving average of
    - beta is the weight used for the moving average

    Returns: a list containing the moving averages of data
    """
    V = 0
    moving_averages = []
    for i in range(len(data)):
        V = beta * V + (1 - beta) * data[i]
        moving_averages.append(V / (1 - beta ** (i + 1)))
    return moving_averages
