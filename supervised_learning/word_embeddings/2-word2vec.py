#!/usr/bin/env python3
"""
Train Word2Vec

creates and trains a gensim word2vec model
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    ARGS:
        *sentences: {list} : sentences to be trained on
        *size :dimensionality of embedding layer
        *min_count: minimum number of occurrences of a word
        *window:  maximum distance between the current and
                  predicted word within a sentence
        *negative : size of negative sampling
        *cbow: {boolean} :training type; True  CBOW; False  Skip-gram
        *iterations: number of iterations to train over
        *seed: seed for the random number generator
        workers: number of worker threads to train the model
    Returns: the trained model
    """
    w2v_model = Word2Vec(size=size, min_count=min_count, window=window,
                         negative=negative, sg=cbow,
                         seed=seed, workers=workers)
    # Building the Vocabulary Table
    w2v_model.build_vocab(sentences)
    # Training of the model
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count,
                    epochs=iterations)

    return w2v_model
