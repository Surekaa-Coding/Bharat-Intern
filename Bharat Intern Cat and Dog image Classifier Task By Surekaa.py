import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Parameters
img_size = (150, 150)
batch_size = 32
epochs = 30
initial_learning_rate = 1e-4

# Directory paths
train_dir = r'C:\Users\surek\Downloads\Cat and Dog Images\train'
val_dir = r'C:\Users\surek\Downloads\Cat and Dog Images\validation'

# Check if directories exist
print("Train directory exists:", os.path.exists(train_dir))
print("Validation directory exists:", os.path.exists(val_dir))

# Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Model Definition
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Optimizer and Callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # Ensure full epoch usage
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,  # Ensure full validation usage
    callbacks=[early_stopping, reduce_lr]
)

# Model Evaluation
plt.figure(figsize=(14, 7))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save plots
plt.savefig(r'C:\Users\surek\Downloads\training_plots.png')
plt.show()

# Making Predictions
def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path does not exist: {img_path}")
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return "Dog" if prediction[0] > 0.5 else "Cat"

# Define image paths for testing
img_paths = [
    r'C:\Users\surek\Downloads\Model Testing Images\example.jpg',
    r'C:\Users\surek\Downloads\Model Testing Images\example (2).jpg'
]

# Loop through each image path and make predictions
for idx, img_path in enumerate(img_paths):
    try:
        result = predict_image(img_path)
        print(f"Prediction {idx + 1}: {result} for '{img_path}'")
    except FileNotFoundError as e:
        print(f"Error with '{img_path}':", e)
