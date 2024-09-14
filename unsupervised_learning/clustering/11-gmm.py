#!/usr/bin/env python3
"""This modlue contains the GMM function"""
import sklearn.mixture


def gmm(X, k):
    """This function calculates the expectation maximization fr a GMM
    """
    Gmm = sklearn.mixture.GaussianMixture(n_components=k)
    parameters = Gmm.fit(X)
    pi = parameters.weights_
    m = parameters.means_
    S = parameters.covariances_
    clss = Gmm.predict(X)
    bic = Gmm.bic(X)
    return pi, m, S, clss, bic
