#!/usr/bin/env python3
"""Module for the function precision"""


import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    prec = TP / (TP + FP)
    return prec
