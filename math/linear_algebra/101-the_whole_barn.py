#!/usr/bin/env python3
""" A function that adds 2 matrices"""


def add_matrices(mat1, mat2):
    """Sum of matrices"""
    m1 = len(mat1)
    n1 = len(mat1[0])
    m2 = len(mat1)
    n2 = len(mat2[0])
    add = []
    
    if m1 == m2 and n1 == n2
        for i in range(m1):
            lis = [0] * n1
            add.append(lis)
        for i in range(m1):
            for j in range(n1):
                add[i][j] = mat1[i][j] + mat2[i][j]
        return add
    else:
        return None
