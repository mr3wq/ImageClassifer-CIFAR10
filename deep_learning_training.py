# Deep Learning Image Classification Project

## Introduction and Project Overview
# This Jupyter Notebook is part of a deep learning project focused on image classification.
# We will use TensorFlow and Keras to build and train a convolutional neural network (CNN) to classify images.

## Importing Necessary Libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import LearningRateScheduler
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

#check if GPU is available:
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    # Add any other transformations you're interested in
)

## Data Loading and Preprocessing
# Here, we will load and preprocess our data. We will use a standard dataset for demonstration purposes.
# ```python
# Example with CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Fit the generator to your data for data augmentation
#datagen.fit(x_train)

#early stopping

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,
    mode='max',  # Stops training when the quantity monitored has stopped increasing
    restore_best_weights=True  # Restores model weights from the epoch with the highest value of the monitored quantity.
)

## Building the CNN Model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    #Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    #Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    #Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


## Training the Model

history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping])

#with data augmentation

# history = model.fit_generator(
#     datagen.flow(x_train, y_train, batch_size=64),
#     epochs=100,
#     validation_data=(x_test, y_test),
#     callbacks=[early_stopping, lr_scheduler]
# )


## Evaluating the Model

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')


## Saving the Model

model.save('image_classification_model.h5')
