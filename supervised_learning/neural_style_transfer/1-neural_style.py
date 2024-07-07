#!/usr/bin/env python3
""" This is the module 1-neural_style.py """


import numpy as np
import tensorflow as tf


class NeuralStyleTransfer:
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializer
        Arguments:
            style_image {np.ndarray} -- the image style
            content_image {np.ndarray} -- the image content
            alpha {float} -- the weight for style cost
            beta {float} -- the weight for content cost
        """
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"

        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError(error1)

        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[-1] != 3:
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

    @staticmethod
    def scale_image(image):
        """
        Rescales the image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        Arguments:
            image {np.ndarray} -- the image to be scaled
        Returns:
            np.ndarray -- the scaled image
        """
        error = "image must be a numpy.ndarray with shape (h, w, 3)"

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
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
        """
        Loads the model for Neural Style Transfer
        """
        base_vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        custom_object = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        base_vgg.save("base_vgg")
        vgg = tf.keras.models.load_model("base_vgg", custom_objects=custom_object)

        for layer in vgg.layers:
            layer.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
