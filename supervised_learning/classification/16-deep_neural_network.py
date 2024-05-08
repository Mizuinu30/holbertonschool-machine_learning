#!/usr/bin/env python3
""" Module that defines a deep neural network """

import numpy as np


class DeepNeuralNetwork:
    """ Class that defines a deep neural network """

    def __init__(self, nx, layers):
        """
        Initializes the DeepNeuralNetwork instance
        Args:
            nx: is the number of input features
            layers: is a list representing the number of nodes in each
                    layer of the network
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            key_W = "W" + str(i + 1)
            key_b = "b" + str(i + 1)

            if i == 0:
                self.weights[key_W] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights[key_W] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            self.weights[key_b] = np.zeros((layers[i], 1))
