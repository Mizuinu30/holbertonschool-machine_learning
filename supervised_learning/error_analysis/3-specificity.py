#!/usr/bin/env python3
""" Module for the function specificity """


import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    spec = TN / (TN + FP)
    return spec
