#!/usr/bin/env python3
"""This module contains a function that creates a variational autoencoder"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def sampling(args):
    """This function samples from the mean and log variation"""
    mean, log_variation = args
    epsilon = K.random_normal(shape=K.shape(mean))
    return mean + K.exp(log_variation * 0.5) * epsilon

def vae_loss(input_layer, output_layer, mean, log_variation):
    """This function calculates the VAE loss"""
    reconstruction_loss = keras.losses.binary_crossentropy(input_layer, output_layer)
    reconstruction_loss *= input_layer.shape[1]
    kl_loss = 1 + log_variation - K.square(mean) - K.exp(log_variation)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)

def autoencoder(input_dims, hidden_layers, latent_dims):
    """This function creates a variational autoencoder"""
    encoder_inputs = keras.Input(shape=(input_dims,))
    for idx, units in enumerate(hidden_layers):
        layer = keras.layers.Dense(units=units, activation="relu")
        if idx == 0:
            outputs = layer(encoder_inputs)
        else:
            outputs = layer(outputs)
    layer = keras.layers.Dense(units=latent_dims)
    mean = layer(outputs)
    layer = keras.layers.Dense(units=latent_dims)
    log_variation = layer(outputs)

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [mean, log_variation]
    )
    encoder = keras.models.Model(
        inputs=encoder_inputs, outputs=[z, mean, log_variation]
    )

    decoder_inputs = keras.Input(shape=(latent_dims,))
    for idx, units in enumerate(reversed(hidden_layers)):
        layer = keras.layers.Dense(units=units, activation="relu")
        if idx == 0:
            outputs = layer(decoder_inputs)
        else:
            outputs = layer(outputs)
    layer = keras.layers.Dense(units=input_dims, activation="sigmoid")
    outputs = layer(outputs)
    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)

    outputs = encoder(encoder_inputs)
    outputs = decoder(outputs[0])
    auto = keras.models.Model(inputs=encoder_inputs, outputs=outputs)

    auto.compile(optimizer="adam", loss=lambda y_true, y_pred: vae_loss(encoder_inputs, y_pred, mean, log_variation))

    return encoder, decoder, auto
