#!/usr/bin/env python3
"""
module sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    sens = TP / (TP + FN)
    return sens
