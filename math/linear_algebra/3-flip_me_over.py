#!/usr/bin/env python3
""" Function to transpose a matrix """


def matrix_transpose(matrix):
    """ Transposes a matrix"""
    rows = len(matrix)
    cols = len(matrix[0])

    # Create a new matrix with swapped dimensions
    transpose = [[0 for _ in range(rows)] for _ in range(cols)]

    # Populate the transpose matrix
    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = matrix[i][j]

    return transpose
