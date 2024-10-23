#!/usr/bin/env python3
"""This module contines the q_init function"""
import numpy as np


def q_init(env):
    """ initializes the Q-table for the given environment"""

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    return q_table
