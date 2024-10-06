#!/usr/bin/env python3
"""
Transformer model for Portuguese to English machine translation.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LayerNormalization, Dropout, Dense, Embedding
from tensorflow.keras.optimizers import Adam

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention mechanism."""
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = Dense(dm)
        self.Wk = Dense(dm)
        self.Wv = Dense(dm)
        self.dense = Dense(dm)

    def split_heads(self, x, batch_size):
        """Splits the input tensor into multiple attention heads."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """Performs the multi-head attention mechanism."""
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.Wq(q), batch_size)
        k = self.split_heads(self.Wk(k), batch_size)
        v = self.split_heads(self.Wv(v), batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        scaled_attention = tf.matmul(attention_weights, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))
        output = self.dense(concat_attention)

        return output

class EncoderLayer(tf.keras.layers.Layer):
    """Encoder layer."""
    def __init__(self, dm, h, hidden, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.ffn = tf.keras.Sequential([
            Dense(hidden, activation='relu'),
            Dense(dm)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """Forward pass for encoder layer."""
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    """Decoder layer."""
    def __init__(self, dm, h, hidden, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.ffn = tf.keras.Sequential([
            Dense(hidden, activation='relu'),
            Dense(dm)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Forward pass for decoder layer."""
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

class Encoder(tf.keras.layers.Layer):
    """Encoder stack."""
    def __init__(self, N, dm, h, hidden, input_vocab_size, max_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N

        self.embedding = Embedding(input_vocab_size, dm)
        self.pos_encoding = self.positional_encoding(max_len, dm)

        self.enc_layers = [EncoderLayer(dm, h, hidden, dropout_rate) for _ in range(N)]
        self.dropout = Dropout(dropout_rate)

    def positional_encoding(self, position, d_model):
        """Positional encoding."""
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        """Helper function to calculate angles for positional encoding."""
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x, training, mask):
        """Forward pass for the encoder."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.enc_layers[i](x, training, mask)

        return x

class Decoder(tf.keras.layers.Layer):
    """Decoder stack."""
    def __init__(self, N, dm, h, hidden, target_vocab_size, max_len, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N

        self.embedding = Embedding(target_vocab_size, dm)
        self.pos_encoding = self.positional_encoding(max_len, dm)

        self.dec_layers = [DecoderLayer(dm, h, hidden, dropout_rate) for _ in range(N)]
        self.dropout = Dropout(dropout_rate)

    def positional_encoding(self, position, d_model):
        """Positional encoding."""
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        """Helper function to calculate angles for positional encoding."""
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Forward pass for the decoder."""
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        return x

class Transformer(tf.keras.Model):
    """Transformer model."""
    def __init__(self, N, dm, h, hidden, input_vocab_size, target_vocab_size, max_len, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab_size, max_len, dropout_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab_size, max_len, dropout_rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """Forward pass for the Transformer."""
        enc_output = self.encoder(inputs[0], training, enc_padding_mask)
        dec_output = self.decoder(inputs[1], enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output
