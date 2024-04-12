#!/usr/bin/env python3
""" Function to slice a matrix along specific axes using pure Python lists"""

def np_slice(matrix, axes={}):
    """"""
    max_dims = 2  # Assumption: the matrix is 2D
    if any(axis >= max_dims for axis in axes):
        raise ValueError("Axis index out of range for a 2D matrix")

    # Extracting the row and column ranges from the axes dictionary
    sliced_matrix = []  # Initialize sliced_matrix as an empty list

    row_range = axes.get(0, (None, None))  # Slicing info for rows
    col_range = axes.get(1, (None, None))  # Slicing info for columns

    # Apply row slicing
    start = row_range[0] if row_range[0] is not None else 0
    stop = row_range[1] if row_range[1] is not None  else len(sliced_matrix[0])
    step = row_range[2] if row_range[2] is not None else 1
    sliced_matrix = matrix[start:stop:step]

    # Apply column slicing if specified
    if col_range != (None, None, None):
        start = row_range[0] if row_range[0] is not None else 0
        stop = row_range[1] if row_range[1] is not None  else len(sliced_matrix[0])
        step = row_range[2] if row_range[2] is not None else 1
        sliced_matrix = [row[col_range[0]:col_range[1]] for row in sliced_matrix]

    return sliced_matrix
