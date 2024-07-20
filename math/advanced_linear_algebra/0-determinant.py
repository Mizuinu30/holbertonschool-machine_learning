#!/usr/bin/env python3
""" Determinant """


def determinant(matrix):
    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    if len(matrix) == 0:
        return 1  # The determinant of a 0x0 matrix is 1
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for a 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    # Base case for a 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for matrix of size 3x3 or greater
    def minor(matrix, row, col):
        """Return the minor of the matrix after removing the given row and column."""
        return [row[:col] + row[col+1:] for row in (matrix[:row] + matrix[row+1:])]

    det = 0
    for c in range(len(matrix)):
        det += ((-1) ** c) * matrix[0][c] * determinant(minor(matrix, 0, c))
    return det
