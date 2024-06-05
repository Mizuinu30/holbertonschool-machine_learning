#!/usr/bin/env python3
""" CNN Module """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs forward propagation over a pooling layer
    of a neural network  """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    sh, sw = stride
    ch = int((h_prev - kh) / sh) + 1
    cw = int((w_prev - kw) / sw) + 1
    conv = np.zeros((m, ch, cw, c_prev))
    for x in range(ch):
        for y in range(cw):
            if mode == 'max':
                conv[:, x, y] = (np.max(A_prev[:, x*sh:((x*sh)+kh),
                                        y*sw:((y*sw)+kw)], axis=(1, 2)))
            else:
                conv[:, x, y] = (np.mean(A_prev[:, x*sh:((x*sh)+kh),
                                         y*sw:((y*sw)+kw)], axis=(1, 2)))
    return conv
