#!/usr/bin/env python3
""" CNN Module """
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """
    performs forward propagation over a
    convolutional layer of a neural network
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph = pw = 0
    else:
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    img_pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')
    ch = int(((h_prev + 2 * ph - kh) / sh) + 1)
    cw = int(((w_prev + 2 * pw - kw) / sw) + 1)
    conv = np.zeros((m, ch, cw, c_new))
    for i in range(ch):
        for j in range(cw):
            for k in range(c_new):
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw
                img_slice = img_pad[:, v_start:v_end, h_start:h_end]
                kernel = W[:, :, :, k]
                conv[:, i, j, k] = (np.sum(np.multiply(img_slice, kernel),
                                           axis=(1, 2, 3)))
    Z = conv + b
    return activation(Z)
