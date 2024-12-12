#!/usr/bin/env python3
""" Defines function that changes the brightness of an image """


import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Changes the brightness of an image

    Args:
        image [3D tf.Tensor]: contains the image to change brightness
        max_delta [float]: maximum change in brightness

    Returns:
        the adjusted image
    """
    return tf.image.random_brightness(image, max_delta)
