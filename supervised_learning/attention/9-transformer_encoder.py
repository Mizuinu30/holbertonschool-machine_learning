#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create the encoder for a transformer.
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Class to create the encoder for a transformer.

    Args:
        N (int): The number of blocks in the encoder.
        dm (int): The dimensionality of the model.
        h (int): The number of heads.
        hidden (int): The number of hidden units in fully connected layer.
        input_vocab (int): The size of the input vocabulary.
        max_seq_len (int): The maximum sequence length possible.
        drop_rate (float): The dropout rate. Default is 0.1.

    Attributes:
        N (int): The number of blocks in the encoder.
        dm (int): The dimensionality of the model.
        embedding (Layer): The embedding layer for the inputs.
        positional_encoding (ndarray):
        Positional encodings of shape (max_seq_len, dm).
        blocks (list): List of length N containing EncoderBlocks.
        dropout (Layer): Dropout layer applied to positional encodings.
    """

    def __init__(
                 self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """Initializes the Encoder."""
        if not isinstance(N, int):
            raise TypeError("N must be an integer representing the number of blocks.")
        if not isinstance(dm, int):
            raise TypeError("dm must be an integer representing the model dimensionality.")
        if not isinstance(h, int):
            raise TypeError("h must be an integer representing the number of heads.")
        if not isinstance(hidden, int):
            raise TypeError("hidden must be an integer representing hidden units.")
        if not isinstance(input_vocab, int):
            raise TypeError("input_vocab must be an integer representing the input vocabulary size.")
        if not isinstance(max_seq_len, int):
            raise TypeError("max_seq_len must be an integer representing the max sequence length.")
        if not isinstance(drop_rate, float):
            raise TypeError("drop_rate must be a float representing the dropout rate.")

        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Calls the encoder and returns the encoder's output.

        Args:
            x (tensor): Input tensor of shape (batch, input_seq_len, dm).
            training (bool): Boolean indicating if the model is in training mode.
            mask: Mask to be applied for multi-head attention.

        Returns:
            Tensor: Output tensor of shape (batch, input_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]

        # Apply embedding and scale
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        # Apply each encoder block
        for block in self.blocks:
            x = block(x, training, mask)

        return x
