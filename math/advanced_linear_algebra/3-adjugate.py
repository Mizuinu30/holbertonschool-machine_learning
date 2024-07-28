#!/usr/bin/env python3
"""This module calculates the adjugate matrix
of a square matrix
"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """ Calculates the adjugate matrix of a matrix"""
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []
    for i in range(len(matrix)):
        adjugate_row = []
        for j in range(len(matrix)):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix
