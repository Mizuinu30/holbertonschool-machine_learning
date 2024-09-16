#!/usr/bin/env python3
"""
This module contains a function to convert
a gensim Word2Vec model to a Keras Embedding layer.
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.

    Parameters:
    model (gensim.models.Word2Vec): A trained gensim Word2Vec model.

    Returns:
    tf.keras.layers.Embedding: The trainable Keras Embedding layer.
    """
    # Extract the weights from the gensim model
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    # Create the Keras Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
