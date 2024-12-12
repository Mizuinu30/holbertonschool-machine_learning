#!/usr/bin/env python3
"""
"""


import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise

    Args:
        image [3D td.Tensor]:
            contains the image to rotate

    Returns:
        the rotated image
    """
    return (tf.image.rot90(image))
