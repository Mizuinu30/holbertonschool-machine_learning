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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            key_W = "W" + str(i + 1)
            key_b = "b" + str(i + 1)

            if i == 0:
                self.__weights[key_W] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[key_W] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            self.__weights[key_b] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Getter function for L """
        return self.__L

    @property
    def cache(self):
        """ Getter function for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter function for weights """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            key_W = "W" + str(i + 1)
            key_b = "b" + str(i + 1)
            key_A = "A" + str(i)
            key_newA = "A" + str(i + 1)

            Z = np.matmul(self.__weights[key_W], self.__cache[key_A]) + self.__weights[key_b]
            self.__cache[key_newA] = 1 / (1 + np.exp(-Z))

        return self.__cache[key_newA], self.__cache
