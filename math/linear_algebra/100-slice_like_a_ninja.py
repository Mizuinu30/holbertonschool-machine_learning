#!/usr/bin/env python3
""" A function that slices a matrix along specific axes"""


import numpy as np

def np_slice(matrix, axes={}):
    """ Return a sliced matrix along specific axes """
    slices = [slice(None)] * matrix.ndim
    for axis, slice_range in axes.items():
        slices[axis] = slice(*slice_range)
    return matrix[tuple(slices)]
        