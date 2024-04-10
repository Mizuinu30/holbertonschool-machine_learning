#!/usr/bin/env python3
""" Function that performs element-wise addition,
subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """ Performs element-wise addition, subtraction, multiplication, and division"""
    if mat1.shape != mat2.shape:
        return None
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
