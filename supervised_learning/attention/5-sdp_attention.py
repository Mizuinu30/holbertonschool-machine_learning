#!/usr/bin/env python3
""" SDP Attention decoding"""


import tensorflow as tf


def sdp_attention(Q, K, V mask=None)
""" SDP Attention decoding

Args:
    Q: tensor with it's last two dimensions are (..., seq_len_q, dk)
        containing the query matrix
    K: tensor with it's last two diensions are (..., seq_len_v, dk)
        containing the key matrix
    V: tensor with it's last two dimensions are (..., seq_len_v, dv)
        containing the value matrix
    mask: tensor with it's last two dimensions are (..., seq_len_q, seq_len_v)
        containing the optional mask, or default None
        output: output, weights

    Returns: output, weights """

    q = tf.matmul(Q, K, transpose_b=True)
    # scale q
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = q / tf.math.sqrt(dk)

    if mask is not None"
        scaled += (mask * -1e9)

    weights = tf.nn.softmax(scaled_q, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
