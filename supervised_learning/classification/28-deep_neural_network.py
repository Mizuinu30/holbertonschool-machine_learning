#!/usr/bin/env python3
"""
    Class DeepNeuralNetwork : deep NN performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Tuple, Union


class DeepNeuralNetwork:
    """
        Class DeepNeuralNetwork
    """

    def __init__(self, nx: int, layers: List[int], activation: str = 'sig') -> None:
        """
            Class constructor
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
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache: Dict[str, np.ndarray] = {}
        self.__weights: Dict[str, np.ndarray] = {}
        self.__activation = activation
        for i in range(self.__L):
            if i == 0:
                self.__weights["W" + str(i + 1)] = (np.random.randn(layers[i], nx) / np.sqrt(nx))
            else:
                self.__weights["W" + str(i + 1)] = (np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1]))
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
    def activation(self) -> str:
        """
            Activation function used in the hidden layers
        """
        return self.__activation

    def forward_prop(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
            Method calculates the forward propagation of the neural network
        """

        self.__cache['A0'] = X
        L = self.__L

        for looper in range(1, L):
            Z = (np.matmul(self.__weights["W" + str(looper)],
                           self.__cache['A' + str(looper - 1)]) +
                 self.__weights['b' + str(looper)])
            if self.__activation == 'sig':
                A = 1 / (1 + np.exp(-Z))
            else:
                A = np.tanh(Z)
            self.__cache['A' + str(looper)] = A

        Z = (np.matmul(self.__weights["W" + str(L)],
                       self.__cache['A' + str(L - 1)]) +
             self.__weights['b' + str(L)])
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        self.__cache['A' + str(L)] = A

        return A, self.__cache

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

    def gradient_descent(self, Y: np.ndarray, cache: Dict[str, np.ndarray], alpha: float = 0.05) -> None:
        """
            Method to calculate one pass of gradient descent
            on the neural network
        """
        L = self.__L
        m = Y.shape[1]
        dZ = cache['A' + str(L)] - Y
        dW = np.matmul(dZ, cache['A' + str(L - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W_prev = np.copy(self.__weights['W' + str(L)])
        self.__weights['W' + str(L)] -= alpha * dW
        self.__weights['b' + str(L)] -= alpha * db

        for loops in range(L - 1, 0, -1):
            dA = np.matmul(W_prev.T, dZ)
            A = cache['A' + str(loops)]
            if self.__activation == 'sig':
                dZ = dA * A * (1 - A)
            else:
                dZ = dA * (1 - (A ** 2))
            dW = np.matmul(dZ, cache['A' + str(loops - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W_prev = np.copy(self.__weights['W' + str(loops)])
            self.__weights['W' + str(loops)] -= alpha * dW
            self.__weights['b' + str(loops)] -= alpha * db

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
