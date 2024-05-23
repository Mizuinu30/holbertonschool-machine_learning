#!/usr/bin/env python3
""" Module for the function f1_score """


import numpy as np


def f1_score(confusion):
    """ calculates the F1 score of each class in a confusion matrix """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    f1 = 2 * prec * rec / (prec + rec)
    return f1
