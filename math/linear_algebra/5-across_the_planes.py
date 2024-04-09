#!/user/bin/env python3
""" function to add two 2D matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ Adds two 2D matrices element-wise """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[x][y] + mat2[x][y] for y in range(len(mat1[0]))] for x in range(len(mat1))]
