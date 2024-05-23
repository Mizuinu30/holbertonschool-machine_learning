#!/usr/bin/env python3
""" Module for the function f1_score """


import numpy as np


def f1_score(confusion):
    # F1 score is the harmonic mean of precision and sensitivity
    # F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    # Import the required functions
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    # Calculate sensitivity and precision
    sens = sensitivity(confusion)
    prec = precision(confusion)

    # Calculate F1 score
    f1 = 2 * (prec * sens) / (prec + sens)

    return f1
