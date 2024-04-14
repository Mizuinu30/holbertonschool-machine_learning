#!/usr/bin/env python3
""" defines function that adds two matrices """

def matrix_shape(matrix):
    """ returns list of integers representing dimensions of given matrix """
    shape = []
    while isinstance(matrix, list):  # Checks if the input is a list (matrix)
        shape.append(len(matrix))    # Appends the size of the current dimension
        matrix = matrix[0]           # Moves to the next dimension
    return shape

def add_matrices(mat1, mat2):
    """ returns new matrix that is sum of two matrices added element-wise """

    # Check if the dimensions of both matrices are the same
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    # Handle the addition based on the dimensionality of the matrices
    if isinstance(mat1[0], list):  # If the element is a list, proceed recursively
        return [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]
    else:  # Base case: elements of the matrices are not lists (i.e., actual numbers)
        return [x + y for x, y in zip(mat1, mat2)]
