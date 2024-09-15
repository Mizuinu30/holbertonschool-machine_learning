#!/usr/bin/env python
""" Bag of words model """

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ Bag of words model """

    vectorizer = CountVectorizer(vocabulary=vocab)
    X_train_counts = vectorizer.fit_transform(sentences)
    embeddings = X_train_couints.toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
