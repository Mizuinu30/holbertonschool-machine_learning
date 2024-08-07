#!/usr/bin/env python3
"""This module calculates the definitness of a
matrix"""
import numpy as np


def definiteness(matrix):
    """ Calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    if np.all(np.linalg.eigvals(matrix) > 0):
        return "Positive definite"
    if np.all(np.linalg.eigvals(matrix) >= 0):
        return "Positive semi-definite"
    if np.all(np.linalg.eigvals(matrix) < 0):
        return "Negative definite"
    if np.all(np.linalg.eigvals(matrix) <= 0):
        return "Negative semi-definite"
    return "Indefinite"
