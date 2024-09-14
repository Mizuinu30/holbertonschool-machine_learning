#!/usr/bin/env python3
"""This module contains the function viterbi that calculates the most likely
sequence of hidden states for a hidden Markov model."""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """This function calculates the most likely sequence of hidden states for a
    hidden Markov model.

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
        path: a list of length T containing the most likely sequence of hidden
              states.
        P: the probability of obtaining the path sequence.
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

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Initialize base cases (t == 0)
    viterbi[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Step 3: Recursion
    for t in range(1, T):
        for s in range(N):
            transition_probs = viterbi[:, t-1] * Transition[:, s]
            viterbi[s, t] = np.max(
                transition_probs) * Emission[s, Observation[t]]
            backpointer[s, t] = np.argmax(transition_probs)

    # Step 4: Termination
    P = np.max(viterbi[:, T-1])
    Last_state = np.argmax(viterbi[:, T-1])

    # Step 5: Backtrack to find the most likely path
    path = [Last_state]
    for t in range(T-1, 0, -1):
        Last_state = backpointer[Last_state, t]
        path.insert(0, Last_state)

    return path, P
