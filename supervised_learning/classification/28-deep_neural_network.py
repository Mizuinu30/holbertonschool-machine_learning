#!/usr/bin/env python3
"""
    Class DeepNeuralNetwork : deep NN performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Tuple, Union


class DeepNeuralNetwork:
    """class DeepNeuralNetwork"""

    def __init__(self, nx, layers, activation='sig'):
        """
        Class constructor
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache: Dict[str, np.ndarray] = {}
        self.__weights: Dict[str, np.ndarray] = {}
        self.__activation = activation

        for i in range(self.__L):
            if i == 0:
                self.__weights["W" + str(i + 1)] = np.random.randn(layers[i], nx) / np.sqrt(nx)
            else:
                self.__weights["W" + str(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self) -> int:
        """
        The number of layers in the neural network
        """
        return self.__L

    @property
    def cache(self) -> Dict[str, np.ndarray]:
        """
        Dictionary to hold all intermediary values
        Upon instantiation, empty
        """
        return self.__cache

    @property
    def weights(self) -> Dict[str, np.ndarray]:
        """
        Dictionary holding all weights and biases of the network
        """
        return self.__weights

    @property
    def activation(self):
        """getter for __activation"""
        return self.__activation

    def forward_prop(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            key_W = "W" + str(i + 1)
            key_b = "b" + str(i + 1)
            key_A_prev = "A" + str(i)
            key_A = "A" + str(i + 1)

            Z = np.dot(self.__weights[key_W], self.__cache[key_A_prev]) + self.__weights[key_b]
            if i == self.__L - 1:
                # softmax activation for the last layer
                exp_Z = np.exp(Z)
                self.__cache[key_A] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # activation for the other layers
                if self.__activation == 'sig':
                    self.__cache[key_A] = 1 / (1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    self.__cache[key_A] = np.tanh(Z)

        return self.__cache[key_A], self.__cache

    def cost(self, Y: np.ndarray, A: np.ndarray) -> float:
        """
        Calculate cross-entropy cost for multiclass
        """
        m = Y.shape[1]
        log_loss = -(1 / m) * np.sum(Y * np.log(A))
        return log_loss

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Method to evaluate the network's prediction
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A == np.max(A, axis=0), 1, 0)
        accuracy = np.sum(predictions == Y) / Y.size
        print(f'Accuracy: {accuracy:.2%}')
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dz = cache["A" + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache["A" + str(i - 1)]
            dw = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if self.__activation == 'sig':
                dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (A_prev * (1 - A_prev))
            elif self.__activation == 'tanh':
                dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (1 - A_prev**2)

            self.__weights["W" + str(i)] -= alpha * dw
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 5000, alpha: float = 0.05,
              verbose: bool = True, graph: bool = True, step: int = 100) -> Tuple[np.ndarray, float]:
        """
        Method to train a deep neural network
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        count = []

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {current_cost:.6f}")
                costs.append(current_cost)
                count.append(i)

        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename: str) -> None:
        """
        Method to save instance object to a file in pickle format
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> Union[None, 'DeepNeuralNetwork']:
        """
        Method to load a pickled DeepNeuralNetwork object
        """
        try:
            with open(filename, 'rb') as file:
                loaded_object = pickle.load(file)
            return loaded_object
        except FileNotFoundError:
            return None
