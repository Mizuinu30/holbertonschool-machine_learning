#!/usr/bin/env python3
"""This module contains a function that creates a variational autoencoder."""
from tensorflow import keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """This function creates a variational autoencoder.

    Args:
        input_dims (int): The dimensionality of the input data.
        hidden_layers (list of int): The number of units in each hidden layer.
        latent_dims (int): The dimensionality of the latent space.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model.
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation="relu")(x)

    mean = keras.layers.Dense(units=latent_dims)(x)
    log_variance = keras.layers.Dense(units=latent_dims)(x)

    def sampling(args):
        """Samples from the mean and log variance."""
        mean, log_variance = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
        return mean + keras.backend.exp(log_variance * 0.5) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([mean, log_variance])
    encoder = keras.models.Model(inputs=encoder_inputs, outputs=[z, mean, log_variance])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation="relu")(x)

    decoder_outputs = keras.layers.Dense(units=input_dims, activation="sigmoid")(x)
    decoder = keras.models.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # VAE Model
    vae_outputs = decoder(encoder(encoder_inputs)[0])
    auto = keras.models.Model(inputs=encoder_inputs, outputs=vae_outputs)

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
