#!/usr/bin/env python3
"""This module hoes the epsilo_greedy funtion"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ returns the action with the highest Q-value """

    if np.random.uniform(0, 1) < epsilon:
        # Explore
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit
        action = np.argmax(Q[state])

    return action
