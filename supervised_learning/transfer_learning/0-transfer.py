#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras as K

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase

_, (X, Y) = K.datasets.cifar10.load_data()  # !/usr/bin/env python3
"""
Transfer Knowledge using InceptionV3 on CIFAR-10 dataset
"""


def preprocess_data(X, Y):
    """
    Preprocess the data for the model.
    """
    X_p = K.applications.inception_v3.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == "__main__":
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess data for InceptionV3
    X_train, y_train = preprocess_data(x_train, y_train)
    X_test, y_test = preprocess_data(x_test, y_test)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Define input layer and resize images to fit InceptionV3 input
    inputs = K.Input(shape=(32, 32, 3))
    resized_input = K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image, 299//32, 299//32, "channels_last")
    )(inputs)

    # Load pre-trained InceptionV3 model
    base_model = K.applications.InceptionV3(
        weights='imagenet', include_top=False, input_tensor=resized_input
    )
    base_model.trainable = True  # Allow the base model to be trainable

    # Fine-tune from this layer onwards
    for layer in base_model.layers[:249]:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(
        512, activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0),
        kernel_regularizer=K.regularizers.L2(0.01)
    )(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.Dense(
        256, activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0),
        kernel_regularizer=K.regularizers.L2(0.01)
    )(x)
    x = K.layers.Dropout(0.5)(x)
    predictions = K.layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = K.Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks
    early_stopping = K.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    reduce_lr = K.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001
    )

    # Train the model
    model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // 128,
        epochs=100, verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )

    # Save the model
    model.save('cifar10_inceptionv3.h5')

X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
