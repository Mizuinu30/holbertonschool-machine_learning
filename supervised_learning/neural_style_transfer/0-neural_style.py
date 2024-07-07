#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.image import resize

class NST:
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray) or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not (isinstance(alpha, (int, float)) and alpha >= 0):
            raise TypeError("alpha must be a non-negative number")
        if not (isinstance(beta, (int, float)) and beta >= 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        # Normalize the image's pixel values to the range [0, 1]
        image = image.astype('float32')
        image /= 255.0

        # Get original dimensions
        h, w, _ = image.shape

        # Calculate new dimensions while keeping the aspect ratio
        if h > w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        # Resize the image using bicubic interpolation
        scaled_image = resize(image, (new_h, new_w), method='bicubic')

        # Rescale pixel values to [0, 1] range
        scaled_image = np.expand_dims(scaled_image, axis=0)  # Add batch dimension

        return tf.convert_to_tensor(scaled_image)
