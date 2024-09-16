#!/usr/bin/env python3
"""
This module contains a function to create, build, and train a gensim FastText model.
"""

from gensim.models import FastText

def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim FastText model.

    Parameters:
    - sentences: list of sentences to be trained on
    - vector_size: dimensionality of the embedding layer
    - min_count: minimum number of occurrences of a word for use in training
    - window: maximum distance between the current and predicted word within a sentence
    - negative: size of negative sampling
    - cbow: boolean to determine the training type; True is for CBOW; False is for Skip-gram
    - epochs: number of iterations to train over
    - seed: seed for the random number generator
    - workers: number of worker threads to train the model

    Returns:
    - The trained FastText model
    """

    sg = 0 if cbow else 1  # 0 for CBOW, 1 for Skip-gram

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    model.train(sentences, total_examples=len(sentences), epochs=epochs)

    return model
