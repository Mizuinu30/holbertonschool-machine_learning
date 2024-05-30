#!/usr/bin/env python3
""" Train Module """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent """
    early_stopping_callback = []

    def scheduler(epoch):
        return alpha / (1 + decay_rate * epoch)

    if validation_data and early_stopping:
        early_stopping_callback.append(
            K.callbacks.EarlyStopping(patience=patience))

    if validation_data and learning_rate_decay:
        learning_rate_decay_callback = K.callbacks.LearningRateScheduler(
            scheduler, verbose=1)
        early_stopping_callback.append(learning_rate_decay_callback)

    if save_best:
        checkpoint = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        early_stopping_callback.append(checkpoint)

    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=early_stopping_callback,
                       validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle)
