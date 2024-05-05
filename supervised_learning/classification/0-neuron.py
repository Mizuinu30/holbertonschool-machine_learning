#!/usr/bin/env python3
""" A neuron that performs binary classification """


import numpy as np


class Neuron:
    """ Class Neuron """

    def __init__(self, nx):
        """ Class constructor """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.b = 0
        self.w = np.random.normal(size=(1, nx))
        self.A = 0
