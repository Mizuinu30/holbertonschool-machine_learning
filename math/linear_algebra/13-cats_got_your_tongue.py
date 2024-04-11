#!/usr/bin/env python3
""" Concatenates two arrays """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two arrays
    """
    return np.concatenate((mat1, mat2), axis=axis)