#!/usr/bin/env python3
"""
Defines function that performs forward propagation for bidirectional RNN
"""

import numpy as np

def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for bidirectional RNN
    """
    t, m, i = X.shape
    _, h = h_0.shape

    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_forward[step] = h_prev

    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_backward[step] = h_next

    H = np.concatenate((H_forward, H_backward), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
