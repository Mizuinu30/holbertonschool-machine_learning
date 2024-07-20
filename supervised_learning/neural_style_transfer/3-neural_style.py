#!/usr/bin/env python3
"""This module contain the clas NST"""

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
            alpha {float} -- the weight for style cost
            beta {float} -- the weight for content cost
        """
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be "
        error2 += "a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3:
            raise TypeError(error1)
        if style_image.shape[-1] != 3:
            raise TypeError(error1)
        if not isinstance(content_image,
                          np.ndarray) or content_image.ndim != 3:
            raise TypeError(error2)
        if content_image.shape[-1] != 3:
            raise TypeError(error2)
        if not (isinstance(alpha, (float, int)) and alpha >= 0):
            raise TypeError("alpha must be a non-negative number")
        if not (isinstance(beta, (float, int)) and beta >= 0):
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales the image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        Arguments:
            image {np.ndarray} -- the image to be scaled
        Returns:
            np.ndarray -- the scaled image
        """
        error = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TypeError(error)
        if image.shape[-1] != 3:
            raise TypeError(error)
        max_dim = 512
        h, w, _ = image.shape
        scale = max_dim / max(h, w)
        h_new, w_new = int(h * scale), int(w * scale)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(image, [h_new, w_new], method="bicubic")
        image /= 255.0
        image = tf.clip_by_value(image, 0, 1)
        return tf.expand_dims(image, axis=0)

    def load_model(self):
        """Loads the model for Neural Style Transfer"""

        base_vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
        )

        custom_object = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        base_vgg.save("base_vgg")
        vgg = tf.keras.models.load_model("base_vgg",
                                         custom_objects=custom_object)

        for layer in vgg.layers:
            layer.trainable = False

        style_outputs = \
            [vgg.get_layer(name).output for name in self.style_layers]

        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer
        Arguments:
            input_layer {tf.Tensor} -- the layer for which to calculate
            the gram matrix
        Returns:
            tf.Tensor -- the gram matrix
        """
        error = "input_layer must be a tensor of rank 4"
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError(error)
        if len(input_layer.shape) != 4:
            raise TypeError(error)

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def generate_features(self):
        """"Extracts features used to calculate
        Neural Style Transfer cost
        Sets the public instance attributes:
            gram_style_features - a list of gram matrices
                calculated from the style layer outputs
            content_feature - the content later output
                of the content image
        """

        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        style_outputs = self.model(style_image)
        content_output = self.model(content_image)

        style_features = style_outputs[:-1]
        content_feature = content_output[-1]

        self.gram_style_features = [NST.gram_matrix(style_feature)
                                    for style_feature in style_features]
        self.content_feature = content_feature
