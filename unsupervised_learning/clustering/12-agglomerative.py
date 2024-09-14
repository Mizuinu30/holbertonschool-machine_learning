#!/usr/bin/env python3
"""This module contains the function agglomerative
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """This function performs agglomerative clustering on a dataset
    """
    hierarchy = scipy.cluster.hierarchy
    links = hierarchy.linkage(X, method='ward')
    clss = hierarchy.fcluster(links, t=dist, criterion='distance')

    plt.figure()
    hierarchy.dendrogram(links, color_threshold=dist)
    plt.show()
    return clss
