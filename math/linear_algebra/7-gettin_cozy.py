#!/usr/bin/env python3
""" Function that concatenates two that concatenates two matrices
along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis"""
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    if axis == 1 and len(mat1) == len(mat2):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    return None
