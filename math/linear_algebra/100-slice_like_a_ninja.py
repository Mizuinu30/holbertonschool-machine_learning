#!/usr/bin/env python3
""" A Function that slices Matces on its axes """


def np_slice(matrix, axes={}):
    """ Returns a numpy.ndarray, the slice of a matrix"""
    dimensions = len(matrix.shape)
    slices_matrix = dimensions * [slice(None)]
    for axis, value in axes.items():
        slices_matrix[axis] = slice(*value)
    return matrix[tuple(slices_matrix)]
