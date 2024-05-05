#!/usr/bin/env python3
"""
    Module Content:
        - A Neuron class that defines a single neuron
        performing binary classification
"""
import numpy as np


class Neuron:

    """
        Function:
            - def __init__(self, nx): Class constructor
    """

    def __init__(self, nx):
        """
            __init__: class constructor

            Input:
                @nx: is the number of input features to the neuron.

            Public Instances:
                @W: The weight vector for the neuron.
                @b: The bias for the neuron.
                @A: The activated output of the neuron (prediction).

            Raises:
                TypeError: nx must be an integer.
                ValueError: nx must be a positive integer.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__b = 0
        self.__W = np.random.normal(size=(1, nx))
        self.__A = 0

    @property
    def W(self):
        """
            Getter function

            Return: The weight vector for the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
            Getter function

            Return: The bias for the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
            Getter function

            Return: The activated output of the neuron (prediction).
        """
        return self.__A
