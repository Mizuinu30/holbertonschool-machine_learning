#!/usr/bin/env python3
""" CNN Module """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5
    architecture using tensorflow
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    activation = tf.nn.relu
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                             activation=activation, kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                             activation=activation,
                             kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
    flatten = tf.layers.Flatten()(pool2)
    FC1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    FC2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(FC1)
    FC3 = tf.layers.Dense(units=10, kernel_initializer=init)(FC2)
    y_pred = FC3
    loss = tf.losses.softmax_cross_entropy(y, FC3)
    train = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_pred = tf.nn.softmax(y_pred)
    return y_pred, train, loss, accuracy
