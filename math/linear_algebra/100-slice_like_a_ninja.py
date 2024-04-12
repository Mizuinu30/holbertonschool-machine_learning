def np_slice(matrix, axes={}):
    """ Return a manually sliced matrix along specific axes using pure Python lists """
    max_dims = 2  # this implementation assumes matrices are 2D (list of lists)
    if any(axis >= max_dims for axis in axes):
        raise ValueError("Axis index out of range for a 2D matrix")
    
    # Generate slices for each dimension
    row_range = axes.get(0, (None, None))  # Get row slicing info, default to slicing everything
    col_range = axes.get(1, (None, None))  # Get column slicing info, default to slicing everything
    
    # Apply slicing to rows
    sliced_matrix = matrix[row_range[0]:row_range[1]]
    
    # Apply slicing to columns
    if col_range != (None, None):
        sliced_matrix = [row[col_range[0]:col_range[1]] for row in sliced_matrix]
    
    return sliced_matrix

# Example usage:
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

axes = {0: (1, 3), 1: (1, 3)}  # Slice rows 1 to 2 and columns 1 to 2
sliced = manual_slice(matrix, axes)
print(sliced)  # Output will be [[6, 7], [10, 11]]
