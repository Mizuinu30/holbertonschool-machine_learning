#!/usr/bin/env python3
""" Function to calculate the shape of a matrix """

def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    if not isinstance(matrix, list):
        return []
    if len(matrix) == 0:
        return [0]
    shape = matrix_shape(matrix[0])
    return [len(matrix)] + shape

matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
print(matrix_shape(matrix))
