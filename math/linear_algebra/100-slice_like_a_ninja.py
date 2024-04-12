#!/usr/bin/env python3
""" Function to slice a matrix along specific axes using pure Python lists"""

def np_slice(matrix, axes={}):
    """ Slicing Matrices"""
    num_axes = matrix.ndim
   
    slices =[slice(None)] * num_axes
   
    for axis, slice_range in axes.item():
        slices[axis] = slice(*slice_range)
       
    return matrix[tuple(slices)]
