#!/usr/bin/env python3
""" A Function that slices Matces on its axes"""


def np_slice(matrix, axes={}):
    """ Return a manually sliced matrix along specific axes using list slicing. """
    if not all(isinstance(axis, int) and axis < 2 for axis in axes):
        raise ValueError("Axis index out of range for a 2D matrix")

    # Apply slicing for rows
    row_range = axes.get(0, (None, None, None))
    sliced_matrix = matrix[slice(*row_range)]

    # Apply slicing for columns if specified
    col_range = axes.get(1, (None, None, None))
    if any(x is not None for x in col_range):
        sliced_matrix = [row[slice(*col_range)] for row in sliced_matrix]

    return sliced_matrix
