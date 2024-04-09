#!/usr/bin/env python3
""" Function to calculate the shape of a matrix """

def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = [] # Initialize the shape of the matrix
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix [0]
    return shape
