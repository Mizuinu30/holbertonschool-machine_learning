#!/usr/bin/env python3
"""A Module defines a deep neural network performing binary classification"""


import pickle
import numpy as np
from matplotlib import pyplot as plt


class DeepNeuralNetwork:
    """class DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
                self.weights['b1'] = np.zeros((layers[i], 1))
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Getter function for L"""
        return self.__L

    @property
    def cache(self):
        """ Getter function for cache"""
        return self.__cache

    @property
    def weights(self):
        """ Getter function for weights"""
        return self.__weights
def forward_prop(self, X):
        """
            method calculate forward propagation of neural network
        """
        self.__cache['A0'] = X
        L1 = self.__L

        for lopper in range(1, L1):
            Z = (np.matmul(self.__weights["W" + str(lopper)],
                           self.__cache['A' + str(lopper - 1)]) +
                 self.__weights['b' + str(lopper)])
            if self.__activation == 'sig':
                A = 1 / (1 + np.exp(-Z))
            else:
                A = np.tanh(Z)
            self.__cache['A' + str(lopper)] = A

        Z = (np.matmul(self.__weights["W" + str(L1)],
                       self.__cache['A' + str(L1 - 1)]) +
             self.__weights['b' + str(L1)])
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        self.__cache['A' + str(L1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
            Calculate cross-entropy cost for multiclass
        """
        m = Y.shape[1]
        log_loss = -(1 / m) * np.sum(Y * np.log(A))

        return log_loss

    def evaluate(self, X, Y):
        """
            Method to evaluate the network's prediction
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A == np.max(A, axis=0), 1, 0), cost


    def gradient_descent(self, Y, cache, alpha=0.05):
        """  Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        for i in range(self.__L, 0, -1):
            A = cache['A' + str(i)]
            A_prev = cache['A' + str(i - 1)]
            W = weights_copy['W' + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                dz = da * (A * (1 - A))
            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            da = np.dot(weights_copy['W' + str(i)].T, dz)
            self.__weights['W' + str(i)] = weights_copy[
                'W' + str(i)] - alpha * dw
            self.__weights['b' + str(i)] = weights_copy[
                'b' + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the deep neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph is True:
                    costs.append(cost)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph is True:
            plt.plot(costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object"""
        if not filename.endswith('.pkl'):
            return None
        try:
            with open(filename, 'rb') as file:
                obj = pickle.load(file)
            return obj
        except FileNotFoundError:
            return None
