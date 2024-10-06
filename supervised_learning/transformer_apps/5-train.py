#!/usr/bin/env python3
"""
Training script for Transformer-based Portuguese to English translation.
"""

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer

class CustomSchedule(LearningRateSchedule):
    """Learning rate schedule for training."""
    def __init__(self, dm, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.dm = dm
        self.dm = tf.cast(self.dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

def loss_function(y_true, y_pred):
    """Loss function ignoring padding tokens."""
    loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss_ = loss_object(y_true, y_pred)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Function to create and train a transformer model for translation."""
    dataset = Dataset(batch_size=batch_size, max_len=max_len)
    input_vocab_size = dataset.tokenizer_pt.vocab_size + 2
    target_vocab_size = dataset.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden, input_vocab_size, target_vocab_size, max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for (batch, (inputs, target)) in enumerate(dataset.data_train):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, target)
            with tf.GradientTape() as tape:
                predictions = transformer([inputs, target[:, :-1]], True, enc_padding_mask, combined_mask, dec_padding_mask)
                loss = loss_function(target[:, 1:], predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(target[:, 1:], predictions)

            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}, batch {batch}: loss {train_loss.result():.4f}, accuracy {train_accuracy.result():.4f}")

        print(f"Epoch {epoch + 1}: loss {train_loss.result():.4f}, accuracy {train_accuracy.result():.4f}")

    return transformer
