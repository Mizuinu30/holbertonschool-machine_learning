#!/usr/bin/env python3
"""
This module provides a function to create,
build, and train a gensim Word2Vec model.

Functions:
    word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
        Creates and trains a Word2Vec model using the given parameters.
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim Word2Vec model.

    Args:
        sentences (list of list of str): The corpus for training, where each sentence is a list of words.
        vector_size (int, optional): Dimensionality of the word vectors. Defaults to 100.
        min_count (int, optional): Ignores all words with total frequency lower than this. Defaults to 5.
        window (int, optional): Maximum distance between the current and predicted word within a sentence. Defaults to 5.
        negative (int, optional): If > 0, negative sampling will be used; the int specifies how many "noise words"
            should be drawn (usually between 5-20). Defaults to 5.
        cbow (bool, optional): Defines the training algorithm. If True, uses CBOW (Continuous Bag of Words);
            if False, uses Skip-gram. Defaults to True.
        epochs (int, optional): Number of iterations (epochs) over the corpus. Defaults to 5.
        seed (int, optional): Seed for the random number generator. Defaults to 0.
        workers (int, optional): Number of worker threads to train the model. Defaults to 1.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.

    Detailed Description:
        This function initializes a Word2Vec model using the gensim library, builds its vocabulary
        from the provided sentences, and trains the model.

        - **Word2Vec Class**:
            The `Word2Vec` class constructs the word embedding model based on either the CBOW
            (Continuous Bag-of-Words) or Skip-gram architecture.

        - **Parameters**:
            - `vector_size`:
                - Type: `int`
                - Description: Dimensionality of the word vectors.
            - `min_count`:
                - Type: `int`
                - Description: Ignores all words with total frequency lower than this.
            - `window`:
                - Type: `int`
                - Description: Maximum distance between the current and predicted word within a sentence.
            - `negative`:
                - Type: `int`
                - Description: If > 0, negative sampling will be used; the int specifies how many "noise words" should be drawn.
            - `sg`:
                - Type: `int`
                - Description: Training algorithm; 1 for Skip-gram, 0 for CBOW.
            - `seed`:
                - Type: `int`
                - Description: Seed for the random number generator to ensure reproducibility.
            - `workers`:
                - Type: `int`
                - Description: Number of worker threads to train the model (faster training with multicore machines).

        - **Methods Used**:
            - `build_vocab(sentences)`:
                - Builds the vocabulary from a sequence of sentences.
            - `train(sentences, total_examples, epochs)`:
                - Trains the model on the given sentences.

    Example:
        ```python
        sentences = [['hello', 'world'], ['machine', 'learning'], ['word2vec', 'model']]
        model = word2vec_model(sentences, vector_size=50, epochs=10)
        vector = model.wv['hello']  # Access the vector for the word 'hello'
        ```
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
