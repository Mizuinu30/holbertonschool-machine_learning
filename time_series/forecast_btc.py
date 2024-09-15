#!/usr/bin/env python3
""" Script to train a RNN model to forecast the BTC close price """
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import joblib

# Load the preprocessed data
data = np.load('preprocessed_data.npy')

# Sequence and prediction parameters
sequence_length = 1440  # Past 24 hours
prediction_offset = 60  # Predicting the close price at t+60

X = []
y = []

# Prepare sequences and targets
for i in range(len(data) - sequence_length - prediction_offset):
    X.append(data[i:i + sequence_length])
    # Close price at t+60
    y.append(data[i + sequence_length + prediction_offset - 1][0])

X = np.array(X)
y = np.array(y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Create tf.data.Dataset
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# Build the RNN model
model = keras.models.Sequential([
    keras.layers.LSTM(64, input_shape=(sequence_length, X.shape[2])),
    keras.layers.Dense(1)
])

# Compile the model with MSE loss
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Save the trained model
model.save('btc_forecast_model.h5')
