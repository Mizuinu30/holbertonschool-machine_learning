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
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]])

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
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :] * Emission[:, Observation[t + 1]])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden Markov model."""
    N = Transition.shape[0]
    T = Observations.shape[0]
    M = Emission.shape[1]

    for iteration in range(iterations):
        # E-step: Compute forward and backward probabilities
        _, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        # Calculate Xi and Gamma
        Xi = np.zeros((N, N, T - 1))
        Gamma = np.zeros((N, T))

        for t in range(T - 1):
            denom = np.dot(np.dot(F[:, t].T, Transition) * Emission[:, Observations[t + 1]].T, B[:, t + 1])
            for i in range(N):
                Gamma[i, t] = F[i, t] * B[i, t] / np.sum(F[:, t] * B[:, t])
                Xi[i, :, t] = F[i, t] * Transition[i, :] * Emission[:, Observations[t + 1]] * B[:, t + 1] / denom

        Gamma[:, T - 1] = F[:, T - 1] * B[:, T - 1] / np.sum(F[:, T - 1] * B[:, T - 1])

        # M-step: Update the parameters
        Transition = np.sum(Xi, axis=2) / np.sum(Gamma[:, :-1], axis=1).reshape(-1, 1)
        Emission = np.zeros((N, M))

        for k in range(M):
            mask = Observations == k
            Emission[:, k] = np.sum(Gamma[:, mask], axis=1) / np.sum(Gamma, axis=1)

        # Update the Initial distribution
        Initial[:, 0] = Gamma[:, 0]

        # Convergence check (optional early stopping)
        if np.allclose(Transition, Transition) and np.allclose(Emission, Emission):
            break

    return Transition, Emission
