import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# Directory setup (replace with your actual directory paths)
base_dir = 'data/dogscats/'
train_dir = os.path.join(base_dir, 'sample/train')
validation_dir = os.path.join(base_dir, 'sample/valid')
test_dir = os.path.join(base_dir, 'sample/bolt')

# Image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,  # Random rotations
        width_shift_range=0.2,  # Random horizontal shifts
        height_shift_range=0.2,  # Random vertical shifts
        shear_range=0.2,  # Shear transformations
        zoom_range=0.2,  # Random zoom
        horizontal_flip=True,  # Horizontal flipping
        fill_mode='nearest'
    )

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')


def create_model():
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size

    if train_generator.samples % batch_size > 0:
        steps_per_epoch += 1
    if validation_generator.samples % batch_size > 0:
        validation_steps += 1

    print("Steps per epoch:", steps_per_epoch)
    print("Validation steps:", validation_steps)

    model = Sequential([
        # First convolutional layer with input shape defined
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        # Second convolutional layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        # Third convolutional layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.5),  # Dropout after the third pooling layer

        # Flattening the output to feed into the dense layer
        Flatten(),

        # Dense layers
        Dense(512, activation='relu'),
        Dropout(0.5),  # Dropout before the final dense layer
        Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Setup ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        'best_model.keras',  # Name of the file to save the best model
        monitor='val_accuracy',  # Metric to monitor
        verbose=1,  # Verbosity mode
        save_best_only=True,  # Only save a model if `val_accuracy` has improved
        mode='max'  # Save the model with max validation accuracy
    )

    # Fit the model with the ModelCheckpoint callback
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=25,  # Adjust epochs if needed
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint]  # Add the checkpoint callback
    )

    # Print training and validation accuracy
    for epoch in range(len(history.history['accuracy'])):
        print(f"Epoch {epoch + 1}/{len(history.history['accuracy'])}")
        print(f"- accuracy: {history.history['accuracy'][epoch]:.4f}")
        print(f"- val_accuracy: {history.history['val_accuracy'][epoch]:.4f}")


def test_model():
    # Load the trained model
    model = load_model("best_model.keras")

    dataframe = pd.DataFrame({
        "filename": os.listdir(test_dir)
    })

    # Data generator for the test set
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=test_dir,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False)

    # Predictions
    predictions = model.predict(test_generator, steps=len(test_generator))

    # Convert predictions to labels
    predicted_labels = ['Dog' if pred >= 0.5 else 'Cat' for pred in predictions]

    # Print predictions with filenames
    for filename, label in zip(test_generator.filenames, predicted_labels):
        print(f"{filename.strip('.jpg')}: {label}")


_input = int(input("Test (1) or Create(2)"))
if _input == 1:
    test_model()
elif _input == 2:
    create_model()
else:
    print("Invalid input")