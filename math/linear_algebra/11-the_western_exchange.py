#!/usr/bin/env python3
""" a function that transposes matrix"""


def np_tranpose(matrix):
    """ Returns a transposed matrix"""
    return list(map(list, zip(*matrix)))