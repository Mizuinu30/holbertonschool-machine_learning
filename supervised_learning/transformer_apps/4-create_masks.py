#!/usr/bin/env python3
"""
Dataset
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for the given sequence.

    Args:
        seq: A tf.Tensor of shape (batch_size, seq_len).

    Returns:
        padding_mask: A tf.Tensor of shape (batch_size, 1, 1, seq_len)
                      where padded positions are 1 and
                      non-padded positions are 0.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    Creates a lookahead mask to prevent the decoder
    from attending to future tokens.

    Args:
        size: An integer representing the sequence length.

    Returns:
        look_ahead_mask: A tf.Tensor of shape (size, size),
        with 1s in the upper triangular
                         part and 0s in the lower part
                         (including the diagonal).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inputs, target):
    """
    Creates all the masks for training/validation.

    Args:
        inputs: A tf.Tensor of shape (batch_size, seq_len_in)
        containing the input sentence.
        target: A tf.Tensor of shape (batch_size, seq_len_out)
        containing the target sentence.

    Returns:
        encoder_mask: Padding mask for the encoder
        (batch_size, 1, 1, seq_len_in).
        combined_mask: Lookahead and padding mask
        for the 1st attention block in the decoder
                       (batch_size, 1, seq_len_out, seq_len_out).
        decoder_mask: Padding mask for the 2nd attention block in the decoder
                      (batch_size, 1, 1, seq_len_in).
    """
    # Encoder padding mask
    encoder_mask = create_padding_mask(inputs)

    # Decoder padding mask
    decoder_padding_mask = create_padding_mask(target)

    # Lookahead mask to mask future tokens in the target
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)

    # Combine lookahead mask and decoder padding mask
    combined_mask = tf.maximum(look_ahead_mask, decoder_padding_mask)

    # Decoder mask for the second attention block
    # Ensures that the decoder doesn't
    # attend to the padding tokens in the encoder's input
    decoder_mask = create_padding_mask(inputs)

    return encoder_mask, combined_mask, decoder_mask
