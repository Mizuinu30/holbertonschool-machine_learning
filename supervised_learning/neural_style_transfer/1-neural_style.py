#!/usr/bin/env python3
""" This is the module 1-neural_style.py """


import numpy as np
import tensorflow as tf


class NST:
    """This is the class NST"""

    # Public class attributes
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    content_layer = "block5_conv2"

    # Class constructor
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializer
        Arguments:
            style_image {np.ndarray} -- the image style
            content_image {np.ndarray} -- the image content
            alpha {float} -- the weight for content cost
            beta {float} -- the weight for style cost
        """
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        error3 = "alpha must be a non-negative number"
        error4 = "beta must be a non-negative number"

        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError(error1)
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[-1] != 3:
            raise TypeError(error2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError(error3)
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError(error4)

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = None

    @staticmethod
    def scale_image(image):
        """Preprocesses an image for the VGG19 model
        Arguments:
            image {np.ndarray} -- the original image
        Returns:
            np.ndarray -- the preprocessed image
        """
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.vgg19.preprocess_input(image * 255.0)
        image = np.expand_dims(image, axis=0)
        return image

    def load_model(self):
        """Creates the model used to calculate cost"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        self.model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)
