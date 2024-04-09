#!/usr/bin/env python3
""" Function to calculate the shape of a matrix """

def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = [] # Initialize the shape of the matrix
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix [0]
    return shape

matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
print(matrix_shape(matrix))
