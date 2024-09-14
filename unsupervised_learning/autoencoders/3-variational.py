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

    # Latent space
    z_mean = keras.layers.Dense(units=latent_dims, name="z_mean")(x)
    z_log_var = keras.layers.Dense(units=latent_dims, name="z_log_var")(x)

    def sampling(args):
        """Samples from the latent space using the reparameterization trick."""
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(z_mean))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,), name="z")([z_mean, z_log_var])
    encoder = keras.Model(
        inputs=encoder_inputs, outputs=[z, z_mean, z_log_var], name="encoder")

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation="relu")(x)

    decoder_outputs = keras.layers.Dense(
        units=input_dims, activation="sigmoid")(x)
    decoder = keras.Model(
        inputs=decoder_inputs, outputs=decoder_outputs, name="decoder")

    # VAE Model
    z, z_mean, z_log_var = encoder(encoder_inputs)
    reconstructed = decoder(z)
    auto = keras.Model(
        inputs=encoder_inputs, outputs=reconstructed, name="vae")

    # Loss Function
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs, reconstructed)
    reconstruction_loss *= input_dims
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(
            z_mean) - keras.backend.exp(z_log_var), axis=-1)
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    # Compile Model
    auto.compile(optimizer=keras.optimizers.Adam())

    return encoder, decoder, auto


if __name__ == "__main__":
    # Example usage
    input_dims = 784  # For example, MNIST dataset
    hidden_layers = [512, 256]
    latent_dims = 2

    encoder, decoder, auto = autoencoder(
        input_dims, hidden_layers, latent_dims)

    # Verify the model is not None and optimizer is Adam
    print(auto is not None)
    print(auto.optimizer.__class__.__name__)
