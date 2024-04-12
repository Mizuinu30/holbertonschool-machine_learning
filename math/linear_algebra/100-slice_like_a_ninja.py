#!/usr/bin/env python3
""" Function to slice a matrix along specific axes using pure Python lists"""

def np_slice(matrix, axes={}):
    """
    Return a manually sliced matrix along specific axes using pure Python lists.
    This implementation assumes matrices are 2D (list of lists).
    """
    max_dims = 2  # Assumption: the matrix is 2D
    if any(axis >= max_dims for axis in axes):
        raise ValueError("Axis index out of range for a 2D matrix")
    
    # Extracting the row and column ranges from the axes dictionary
    row_range = axes.get(0, (None, None))  # Slicing info for rows
    col_range = axes.get(1, (None, None))  # Slicing info for columns
    
    # Apply row slicing
    sliced_matrix = matrix[row_range[0]:row_range[1]]
    
    # Apply column slicing if specified
    if col_range != (None, None):
        sliced_matrix = [row[col_range[0]:col_range[1]] for row in sliced_matrix]
    
    return sliced_matrix
