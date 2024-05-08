#!/usr/bin/env python3
""" Module that defines a deep neural network"""
import numpy as np
import matplotlib.pyplot as plt


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
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if (not isinstance(layers, list) or
                not all(map(lambda x: isinstance(x, int) and x > 0, layers))):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if i == 0:
                self.weights["W" + str(i+1)] = (np.random.randn(layers[i], nx)
                                                * np.sqrt(2 / nx))
            else:
                self.weights["W" + str(i+1)] = (np.random.randn(layers[i],
                                                                layers[i - 1])
                                                * np.sqrt(2 / layers[i - 1]))
            self.weights["b" + str(i+1)] = np.zeros((layers[i], 1))

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
        """ Calculates the forward propagation of the neural network
        Args:
            X: is a numpy.ndarray with shape (nx, m) that contains the input
               data
                nx: is the number of input features to the neuron
                m: is the number of examples
        Returns: the output of the neural network and the cache, respectively
        """

        if 'A0' not in self.__cache:
            self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            if i == 1:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                Z = np.matmul(W, X) + b
            else:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                X = self.__cache['A{}'.format(i-1)]
                Z = np.matmul(W, X) + b

            self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(i)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression
        Args:
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
               labels for the input data
            A: is a numpy.ndarray with shape (1, m) containing the activated
               output of the neuron for each example
        Returns: the cost
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions
        Args:
            X: is a numpy.ndarray with shape (nx, m) that contains the input
               data
                nx: is the number of input features to the neuron
                m: is the number of examples
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
               labels for the input data
        Returns: the neuron’s prediction and the cost of the network
            prediction is a numpy.ndarray with shape (1, m) containing the
            predicted labels for each example
            label values are 1 if the output of the network is >= 0.5 0
            otherwise
        """
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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        Args:
            X: is a numpy.ndarray with shape (nx, m) that contains the input
                data
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
                labels for the input data
            iterations: is the number of iterations to train over
            alpha: is the learning rate
            verbose: is a boolean that defines whether or not to print
                information about the training
            graph: is a boolean that defines whether or not to graph information
                about the training once the training has completed
            step: is the number of iterations between printing verbose information
        Returns: the evaluation of the training data after iterations of training
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 1 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        count = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)
            costs.append(cost)
            count.append(i)

            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")

        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
