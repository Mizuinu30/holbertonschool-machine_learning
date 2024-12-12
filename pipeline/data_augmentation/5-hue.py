#!/usr/bin/env python3
""" function that changes the hue of an image """


import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image

    Args:
        image [3D tf.Tensor]: contains the image to change hue
        delta [float]: the amount the hue should change

    Returns:
        the adjusted image
    """
    return tf.image.adjust_hue(image, delta)
