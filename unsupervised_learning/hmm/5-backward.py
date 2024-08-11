#!/usr/bin/env python3
"""This module contains the function backward that performs the backward
algorithm for a hidden Markov model."""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden Markov model.

    Args:
        Observation: numpy.ndarray of shape (T,) that contains
          the index of the observations.
            T: the number of observations.
        Emission: numpy.ndarray of shape (N, M) containing
          the emission probability of a specific
                  observation given a hidden state.
            Emission[i, j]: the probability of observing j
            given the hidden state i.
            N: the number of hidden states.
            M: the number of all possible observations.
        Transition: 2D numpy.ndarray of shape (N, N) containing
          the transition probabilities.
            Transition[i, j]: the probability of transitioning
            from the hidden state i to j.
        Initial: numpy.ndarray of shape (N, 1) containing
          the probability of starting in a particular hidden state.

    Returns:
        P: the likelihood of the observations given the model.
        B: numpy.ndarray of shape (N, T) containing the
          backward path probabilities.
           B[i, j]: the probability of generating the
             future observations from hidden state i at time j.
    """
    # Step 1: Validate the inputs
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2):
        return None, None

    if (Emission.shape[0] != Transition.shape[0] or
            Transition.shape[0] != Transition.shape[1] or
            Initial.shape[0] != Transition.shape[0] or Initial.shape[1] != 1):
        return None, None

    if (not np.isclose(np.sum(Emission, axis=1), 1).all() or
            not np.isclose(np.sum(Transition, axis=1), 1).all() or
            not np.isclose(np.sum(Initial), 1)):
        return None, None

    # Step 2: Initialize variables
    N = Initial.shape[0]  # Number of hidden states
    T = Observation.shape[0]  # Number of observations

    B = np.zeros((N, T))

    # Initialize the last column of B (B[:, T-1] = 1 for all states)
    B[:, T-1] = 1

    # Step 3: Recursion
    for t in range(T-2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                B[:, t+1] * Transition[i, :] * Emission[:, Observation[t+1]])

    # Step 4: Calculate the likelihood of the observations given the model
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
