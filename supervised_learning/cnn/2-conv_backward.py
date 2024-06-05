#!/usr/bin/env python3
""" CNN Module """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ performs back propagation over a convolutional
    layer of a neural network """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (m, h_new, w_new, c_new) = dZ.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    else:
        ph = pw = 0
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    A_prev_pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    dA_prev_pad = np.pad(dA_prev, pad_width=((0, 0), (ph, ph), (pw, pw),
                                             (0, 0)), mode='constant')
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    a_slice = a_prev_pad[v_start:v_end, h_start:h_end]
                    da_prev_pad[v_start:v_end, h_start:h_end] +=\
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
        if padding == 'same':
            dA_prev[i, :, :, :] += da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] += da_prev_pad
    return dA_prev, dW, db
