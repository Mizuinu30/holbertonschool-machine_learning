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

            Z = np.matmul(
                self.__weights[key_W], self.__cache[key_A]) + self.__weights[
                    key_b]
            self.__cache[key_newA] = 1 / (1 + np.exp(-Z))

        return self.__cache[key_newA], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            A: numpy.ndarray with shape (1, m) containing the activated output
        """
        m = Y.shape[1]
        cost = -np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions """

        output, cache = self.forward_prop(X)
        cost = self.cost(Y, output)
        prediction = np.where(output >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network
        Args:
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
               labels for the input data
            cache: is a dictionary containing all the intermediary values of
                   the network
            alpha: is the learning rate
        """
        m = Y.shape[1]
        dZ = cache["A" + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            A = cache["A" + str(i - 1)]
            dW = np.matmul(dZ, A.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W = self.__weights["W" + str(i)]
            dZ = np.matmul(W.T, dZ) * (A * (1 - A))
            self.__weights["W" +
                           str(i)] = self.__weights["W" + str(i)] - alpha * dW
            self.__weights["b" +
                           str(i)] = self.__weights["b" + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the deep neural network
        Args:
            X: is a numpy.ndarray with shape (nx, m) that contains the input
               data
                nx: is the number of input features to the neuron
                m: is the number of examples
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
               labels for the input data
            iterations: is the number of iterations to train over
            alpha: is the learning rate
        Returns: the evaluation of the training data after iterations of
                 training have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)

        return self.evaluate(X, Y)
