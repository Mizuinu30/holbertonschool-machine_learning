#!/usr/bin/env python3
"""
Defines function that creates a variational autoencoder
"""


from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import numpy as np

def sampling(args):
    """ Sampling function for variational autoencoder """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def vae_loss(input_layer, output_layer, z_mean, z_log_var):
    """ Loss function for variational autoencoder """
    reconstruction_loss = binary_crossentropy(input_layer, output_layer)
    reconstruction_loss *= input_layer.shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)

def autoencoder(input_dims, hidden_layers, latent_dims):
    """  Creates a variational autoencoder"""
    # Encoder
    input_layer = Input(shape=(input_dims,))
    x = input_layer
    for nodes in hidden_layers:
        x = Dense(nodes, activation='relu')(x)
    z_mean = Dense(latent_dims)(x)
    z_log_var = Dense(latent_dims)(x)
    z = Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])

    encoder = Model(input_layer, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = Dense(nodes, activation='relu')(x)
    output_layer = Dense(input_dims, activation='sigmoid')(x)

    decoder = Model(latent_inputs, output_layer, name='decoder')

    # VAE
    outputs = decoder(encoder(input_layer)[0])
    auto = Model(input_layer, outputs, name='vae')

    # Compile VAE
    auto.compile(optimizer=Adam(), loss=lambda y_true, y_pred: vae_loss(input_layer, y_pred, z_mean, z_log_var))

    return encoder, decoder, auto
