#!/usr/bin/env python3
""" Function to add two matrices """


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise """
    sum_arrays = [x + y for x, y in zip(arr1, arr2)]
 
    return(sum_arrays)
