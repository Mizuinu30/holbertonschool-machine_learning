#!/usr/bin/env python3
"""
    Defines function that changes the contrast of an image
"""


import tensorflow as tf


def change_contrast(image, lower, upper):
    """ Changes the contrast of an image

    Args:
        image [3D td.Tensor]:
            contains the image to change contrast
        lower [float]:
            lower bound of the contrast range
        upper [float]:
            upper bound of the contrast range
    Returns:
        the adjusted image
    """
    return (tf.image.adjust_contrast(image, lower, upper))
