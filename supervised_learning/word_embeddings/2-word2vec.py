#!/usr/bin/env python3
"""
This module contains a function to create, build,
and train a gensim Word2Vec model.
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim Word2Vec model.

    Parameters:
    sentences (list): List of sentences to be trained on.
    vector_size (int): Dimensionality of the embedding layer.
    min_count (int): Minimum number of occurrences of a word for use in training.
    window (int): Maximum distance between the current and predicted word within a sentence.
    negative (int): Size of negative sampling.
    cbow (bool): Boolean to determine the training type; True is for CBOW; False is for Skip-gram.
    epochs (int): Number of iterations to train over.
    seed (int): Seed for the random number generator.
    workers (int): Number of worker threads to train the model.

    Returns:
    gensim.models.Word2Vec: The trained Word2Vec model.
    """

    sg = 0 if cbow else 1  # 0 for CBOW, 1 for Skip-gram
    """ Create the Word2Vec model """

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )
    """ Build the vocabulary """

    model.build_vocab(sentences)
    """ Train the model """
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    """ Save the model """

    return model
