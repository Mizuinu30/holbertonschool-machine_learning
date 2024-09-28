#!/usr/bin/env python3
""" Multihead attention """


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiheadAttention(tf.keras.layers.Layer):
    """ Multihead attention """

    def__init__(self, dm, h):
        """ Initializer """
        super(MultiheadAttention, self).__init__()
        self.h = h
        self.dm = dm

        self.depth = dm // h

        self.wq = tf.keras.layers.Dense(dm)
        self.wk = tf.keras.layers.Dense(dm)
        self.wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """ Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """ Call function """

        q = self.wq(Q) # (batch_size, seq_len, dm)
        k = self.wk(K) # (batch_size, seq_len, dm)
        v = self.wv(V) # (batch_size, seq_len, dm)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = sdp_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        concat_attention = tf.transpose(scaled_attention,
                                        (batch_size, -1
                                        self.dm))

        output = self.linear(concat_attention)

        return output, attention_weights
