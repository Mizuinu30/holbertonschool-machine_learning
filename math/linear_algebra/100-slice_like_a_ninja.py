def np_slice(matrix, axes={}):
    """Return a manually sliced matrix along specific axes using list slicing."""
    # Ensure matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix) or not isinstance(matrix, list):
        raise ValueError("The matrix should be a 2D list.")

    max_dims = 2  # Supports only 2D lists
    if any(axis >= max_dims for axis in axes.keys()):
        raise ValueError("Axis index out of range for a 2D matrix")

    # Extract row and column slicing details with better default handling
    row_range = axes.get(0, (None, None, None))  # Defaults to slicing whole rows if not specified
    col_range = axes.get(1, (None, None, None))  # Defaults to slicing whole columns if not specified

    # Apply row slicing
    try:
        sliced_matrix = matrix[slice(*row_range)]
    except IndexError:
        raise ValueError("Row slicing indices are out of range.")

    # Apply column slicing if specified
    if any(x is not None for x in col_range):
        try:
            sliced_matrix = [row[slice(*col_range)] for row in sliced_matrix if row]  # checks if row is not empty
        except IndexError:
            raise ValueError("Column slicing indices are out of range.")

    return sliced_matrix
