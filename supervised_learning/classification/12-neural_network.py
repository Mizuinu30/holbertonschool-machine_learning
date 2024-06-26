#!/usr/bin/env python3
"""defines a neural network with one hidden
layer performing binary classification"""
import numpy as np


class NeuralNetwork:
    """class NeuralNetwork"""

    def __init__(self, nx, nodes):
        """nx = the number of input features to the neuron
        nodes = the number of nodes found in the hidden layer
        W1: The weights vector for the hidden layer. Upon instantiation,
        it should be initialized using a random normal distribution.
        b1: The bias for the hidden layer. Upon instantiation,
        it should be initialized with 0’s.
        A1: The activated output for the hidden layer. Upon instantiation,
        it should be initialized to 0.
        W2: The weights vector for the output neuron. Upon instantiation,
        it should be initialized using a random normal distribution.
        b2: The bias for the output neuron. Upon instantiation,
        it should be initialized to 0.
        A2: The activated output for the output neuron (prediction).
        Upon instantiation, it should be initialized to 0."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """returns private instance weight"""
        return self.__W1

    @property
    def b1(self):
        """returns private instance bias"""
        return self.__b1

    @property
    def A1(self):
        """returns private instance output"""
        return self.__A1

    @property
    def W2(self):
        """returns private instance weight"""
        return self.__W2

    @property
    def b2(self):
        """returns private instance bias"""
        return self.__b2

    @property
    def A2(self):
        """returns private instance output"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        X = a numpy.ndarray with shape (nx, m) that contains the input data
        nx = the number of input features to the neuron
        m = the number of examples
        It updates the private attributes __A1 and __A2
        neurons should use a sigmoid activation function"""
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y = a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A = a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        To avoid division by zero errors, it will be used 1.0000001 - A
        instead of 1 - A"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions
        X = numpy.ndarray with shape (nx, m) that contains the input data
        Y = numpy.ndarray with shape (1, m) that contains the correct labels
        nx = the number of input features to the neuron
        m = the number of examples
        It returns the neuron’s prediction and the cost of the network"""
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return prediction, cost
