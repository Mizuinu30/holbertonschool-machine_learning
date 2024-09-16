def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5,
                   cbow=True, epochs=5, seed=0, workers=1):
    """Creates, builds, and trains a Word2Vec model using gensim."""
    import random
    import numpy as np
    from gensim.models import Word2Vec

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Set the training algorithm: 0 for CBOW, 1 for Skip-gram
    sg = 0 if cbow else 1

    # Initialize and train the Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=epochs
    )

    return model
