#!/usr/bin/env python3
""" function def create_placeholders(nx, classes): 
that returns two placeholders, x and y, for the neural network """


import tensorflow as tf

def create_placeholders(nx, classes):
    """ function def create_placeholders(nx, classes): 
    that returns two placeholders, x and y, for the neural network """
    x = tf.compat.v1.placeholder("float", [None, nx], name="x")
    y = tf.compat.v1.placeholder("float", [None, classes], name="y")
    return x, y
