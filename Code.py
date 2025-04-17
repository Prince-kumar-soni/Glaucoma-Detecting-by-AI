# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Initialize ImageDataGenerator for data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Set the path to your dataset
dataset_path = r'C:\Users\thegr\OneDrive\Desktop\Project\Code\dataset'

# Load data and split into training and validation sets
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),  # Resize images if necessary
    batch_size=32,
    class_mode='binary',  # Binary classification for glaucoma or normal
    subset='training'
)

validation_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary crossentropy for binary classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=validation_data
)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title("Loss")
plt.show()

# Save the model
model.save('glaucoma_detection_model.h5')
print("Model saved as glaucoma_detection_model.h5")

# Example: Predict on a new image
from tensorflow.keras.preprocessing import image

# Replace with the path to an image for testing
img_path = r'C:\Users\thegr\OneDrive\Desktop\Project\Code\new_image.jpeg'

# Ensure the file exists
try:
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        print("Glaucoma detected.")
    else:
        print("Normal eye.")
except FileNotFoundError:
    print(f"Image file not found at path: {img_path}")
