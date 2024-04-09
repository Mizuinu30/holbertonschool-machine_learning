#!/usr/bin/env python3
""" Function to add two matrices """


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise """
    if len(arr1) != len(arr2):
        return None
    sum_arrays = [x + y for x, y in zip(arr1, arr2)]
    return (sum_arrays)
