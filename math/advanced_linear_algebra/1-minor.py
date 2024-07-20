#!/usr/bin/env python3
"""This module contains the minor function"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """ Calculates the minor of a matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    height = len(matrix)
    width = len(matrix[0])

    if height != width or (height == 1 and width == 0):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            sub_matrix = [
                row[:j] + row[j + 1:] for k, row in enumerate(matrix) if k != i
            ]
            if len(sub_matrix) > 1:
                minor_row.append(determinant(sub_matrix))
            else:
                minor_row.append(sub_matrix[0][0])
        minor_matrix.append(minor_row)

    return minor_matrix
