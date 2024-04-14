#!/usr/bin/env python3
""" A function that adds 2 matrices"""


def matrix_shape(matrix):
    """returns a list os integers representing the dimensions of a matrix"""
    matrix_shape = []
    while type(matrix) is list:
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape

def add_matrices(mat1, mat2):
    """Sum of matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2)
        return None
    if len(matrix_shape(mat1)) is 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
