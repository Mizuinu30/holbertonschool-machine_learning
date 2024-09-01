#!/usr/bin/env python3
""" This module contains the WGAN class that inherits from keras.Model """

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("Environment variable TF_ENABLE_ONEDNN_OPTS set to 0.")

# Import the WGAN_clip class
try:
    WGAN_clip = __import__('1-wgan_clip').WGAN_clip
    print("WGAN_clip class imported successfully.")
except AttributeError as e:
    print(f"Error importing WGAN_clip: {e}")
    exit(1)

# Function to build the generator and discriminator models
def convolutional_GenDiscr():
    """ Builds a Convolutional GAN """

    def get_generator():
        """ Builds the generator model """
        input_layer = keras.layers.Input(shape=(16,), name='input_1')

        x = keras.layers.Dense(2048, name='dense')(input_layer)
        x = keras.layers.Reshape((2, 2, 512), name='reshape')(x)

        x = keras.layers.UpSampling2D(name='up_sampling2d')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same', name='conv2d')(x)
        x = keras.layers.BatchNormalization(name='batch_normalization')(x)
        x = keras.layers.Activation('relu', name='activation_1')(x)

        x = keras.layers.UpSampling2D(name='up_sampling2d_1')(x)
        x = keras.layers.Conv2D(16, (3, 3), padding='same', name='conv2d_1')(x)
        x = keras.layers.BatchNormalization(name='batch_normalization_1')(x)
        x = keras.layers.Activation('relu', name='activation_2')(x)

        x = keras.layers.UpSampling2D(name='up_sampling2d_2')(x)
        x = keras.layers.Conv2D(1, (3, 3), padding='same', name='conv2d_2')(x)
        x = keras.layers.BatchNormalization(name='batch_normalization_2')(x)
        output_layer = keras.layers.Activation('tanh', name='activation_3')(x)

        return keras.models.Model(input_layer, output_layer, name='generator')

    def get_discriminator():
        """ Builds the discriminator model """
        inpt = keras.layers.Input(shape=(16, 16, 1), name='input_2')

        x = keras.layers.Conv2D(32, (3, 3), padding='same', name='conv2d_3')(inpt)
        x = keras.layers.MaxPooling2D(name='max_pooling2d')(x)
        x = keras.layers.Activation('relu', name='activation_4')(x)

        x = keras.layers.Conv2D(64, (3, 3), padding='same', name='conv2d_4')(x)
        x = keras.layers.MaxPooling2D(name='max_pooling2d_1')(x)
        x = keras.layers.Activation('relu', name='activation_5')(x)

        x = keras.layers.Conv2D(128, (3, 3), padding='same', name='conv2d_5')(x)
        x = keras.layers.MaxPooling2D(name='max_pooling2d_2')(x)
        x = keras.layers.Activation('relu', name='activation_6')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding='same', name='conv2d_6')(x)
        x = keras.layers.MaxPooling2D(name='max_pooling2d_3')(x)
        x = keras.layers.Activation('relu', name='activation_7')(x)

        x = keras.layers.Flatten(name='flatten')(x)
        output_layer = keras.layers.Dense(1, name='dense_1')(x)

        return keras.models.Model(inpt, output_layer, name='discriminator')

    return get_generator(), get_discriminator()

# Create the generator and discriminator
print("Creating generator and discriminator models...")
generator, discriminator = convolutional_GenDiscr()
print("Generator and discriminator models created.")

# Create the WGAN_clip model
latent_generator = lambda size: tf.random.normal([size, 100])
real_examples = np.random.rand(1000, 16, 16, 1).astype(np.float32)

try:
    print("Creating WGAN_clip instance...")
    wgan = WGAN_clip(generator, discriminator, latent_generator, real_examples)
    print("WGAN_clip instance created successfully.")
except Exception as e:
    print(f"Error creating WGAN_clip instance: {e}")
    exit(1)

# Compile the model
try:
    print("Compiling WGAN_clip model...")
    wgan.compile()
    print("WGAN_clip model compiled successfully.")
except Exception as e:
    print(f"Error compiling WGAN_clip model: {e}")
    exit(1)

# Train the model
try:
    print("Training WGAN_clip model...")
    wgan.fit(tf.data.Dataset.from_tensor_slices(real_examples).batch(200), epochs=10)
    print("WGAN_clip model trained successfully.")
except Exception as e:
    print(f"Error training WGAN_clip model: {e}")
    exit(1)
