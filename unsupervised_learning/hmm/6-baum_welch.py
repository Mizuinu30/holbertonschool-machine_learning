#!/usr/bin/env python3
"""This module contains the function baum_welch that performs the Baum-Welch
algorithm for a hidden Markov model."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """This function calculates the forward algorithm for a hidden Markov model."""
    N = Transition.shape[0]
    T = Observation.shape[0]

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(
                F[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]]
            )

    P = np.sum(F[:, -1])
    return P, F


def backward(Observation, Emission, Transition, Initial):
    """This function performs the backward algorithm for a hidden Markov model."""
    N = Transition.shape[0]
    T = Observation.shape[0]

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for s in range(N):
            B[s, t] = np.sum(
                B[:, t + 1] * Transition[s, :] * Emission[:, Observation[t + 1]]
            )

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B
