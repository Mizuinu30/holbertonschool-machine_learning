#!/usr/bin/env python3
"""This module contains the function forward that performs the forward
algorithm for a hidden Markov model."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """This function calculates the forward algorithm for a hidden Markov
    model.

    Args:
        Observation: numpy.ndarray of shape (T,) that contains the index of
                    the observations.
            T: the number of observations.
        Emission: numpy.ndarray of shape (N, M) containing the emission
                  probability of a specific observation given a hidden state.
            Emission[i, j]: the probability of observing j given the hidden
                            state i.
            N: the number of hidden states.
            M: the number of all possible observations.
        Transition: 2D numpy.ndarray of shape (N, N) containing the
                    transition probabilities.
            Transition[i, j]: the probability of transitioning from the
                              hidden state i to j.
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
                 starting in a particular hidden state.

    Returns:
        P: the likelihood of the observations given the model.
        F: numpy.ndarray of shape (N, T) containing the forward path
           probabilities.
           F[i, j]: the probability of being in hidden state i at time j
           given the previous observations.
        None, None on failure.
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

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Step 3: Iterate over the observations
    #  to calculate the forward probabilities
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(
                F[:, t-1] * Transition[:, j] * Emission[j, Observation[t]])

    # Step 4: Calculate the likelihood of the observations given the model
    P = np.sum(F[:, -1])

    return P, F
