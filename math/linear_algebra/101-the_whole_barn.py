#!/usr/bin/env python3
""" A function that adds 2 matrices"""


def add_matrices(mat1, mat2):
    """Sum of matrices"""
    if len(mat1) != len(mat2) or any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None

    result = [[0]*len(mat1[0]) for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            result[i][j] = mat1[i][j] + mat2[i][j]

    return result
        