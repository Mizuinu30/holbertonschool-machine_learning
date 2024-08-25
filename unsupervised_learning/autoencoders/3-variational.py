#!/usr/bin/env python3
"""This modlue contains a function that creates a variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """This function creates a variational autoencoder
    """
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

    def sampling(args):
        """This function samples from the mean and log variation
        """
        mean, log_variation = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
        return mean + keras.backend.exp(log_variation * 0.5) * epsilon

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

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
