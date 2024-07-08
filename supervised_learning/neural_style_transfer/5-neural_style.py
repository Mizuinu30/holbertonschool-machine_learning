#!/usr/bin/env python3
"""This module contain the class NST"""
import numpy as np
import tensorflow as tf


class NST:
    """This is the class NST"""

    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    content_layer = "block5_conv2"

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
        self.generate_features()

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

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer
        Arguments:
            input_layer {tf.Tensor} -- the layer for which to calculate the gram matrix
        Returns:
            tf.Tensor -- the gram matrix
        """
        error = "input_layer must be a tensor of rank 4"
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(input_layer.shape) != 4:
            raise TypeError(error)

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def generate_features(self):
        """
        Extracts features used to calculate Neural Style Transfer cost
        Sets the public instance attributes:
            gram_style_features - a list of gram matrices calculated from the style layer outputs
            content_feature - the content later output of the content image
        """
        style_image = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(style_image)
        content_output = self.model(content_image)

        style_features = style_outputs[:-1]
        content_feature = content_output[-1]

        self.gram_style_features = [NST.gram_matrix(style_feature) for style_feature in style_features]
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        Arguments:
            style_output {tf.Tensor} -- the layer style output
            gram_target {np.ndarray} -- the gram matrix of the target style
        Returns:
            tf.Tensor -- the layer style cost
        """
        c = style_output.shape[-1]

        err_1 = "style_output must be a tensor of rank 4"
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(style_output.shape) != 4:
            raise TypeError(err_1)

        err_2 = "gram_target must be a tensor of shape [1, {}, {}]".format(c, c)
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or gram_target.shape != (1, c, c):
            raise TypeError(err_2)

        gram_style = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return style_cost

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image
        Arguments:
            style_outputs {list} -- a list of tf.Tensor containing the style outputs for the generated image
        Returns:
            tf.Tensor -- the style cost
        """
        st_len = len(self.style_layers)
        err_list_check = "style_outputs must be a list with a length of {}".format(st_len)
        if not isinstance(style_outputs, list) or len(self.style_layers) != len(style_outputs):
            raise TypeError(err_list_check)

        style_costs = []
        weight = 1 / len(self.style_layers)

        for style_output, gram_target in zip(style_outputs, self.gram_style_features):
            layer_style_cost = self.layer_style_cost(style_output, gram_target)
            weighted_layer_style_cost = weight * layer_style_cost
            style_costs.append(weighted_layer_style_cost)

        style_cost = tf.add_n(style_costs)

        return style_cost
