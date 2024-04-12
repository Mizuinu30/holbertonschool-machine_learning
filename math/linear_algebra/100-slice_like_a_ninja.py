#!/usr/bin/env python3
""" Function to slice a matrix along specific axes """


def np_slice(matrix, axes={}):
    """ A function that slices a matrix along specific axes """
    slices = [slice(None)] * len(matrix)
    for axis, slice_range in axes.items():
        slices[axis] = slice(*slice_range)
    return [row[tuple(slices)] for row in matrix]
