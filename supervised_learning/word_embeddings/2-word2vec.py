#!/usr/bin/env python3
"""
This module contains a function to create, build, and train a gensim Word2Vec model.
"""

from gensim.models import Word2Vec

def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5,
                   cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a Word2Vec model using gensim."""
    from gensim.models import Word2Vec

    # Set the training algorithm: 0 for CBOW, 1 for Skip-gram
    sg = 0 if cbow else 1

    # Initialize the Word2Vec model without training
    model = Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # Build the vocabulary from the sentences
    model.build_vocab(sentences)

    # Train the Word2Vec model
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
