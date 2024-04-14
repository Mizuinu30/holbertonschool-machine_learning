#!/usr/bin/env python3
""" defines function that adds two matrices """


def matrix_shape(matrix):
    """ returns list of integers representing dimensions of given matrix """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape

def add_matrices(mat1, mat2):
    """ returns new matrix that is sum of two matrices added element-wise """


    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    if isinstance(mat1[0], list):
        return [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]
    else:
        return [x + y for x, y in zip(mat1, mat2)]
