import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt

def build_small_cnn(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    return model

# Compile the model
model = build_small_cnn()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
    
def load_dataset(data_dir="sudoku_dataset"):
    images, labels = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png") and "cell" in filename:
            img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = img.reshape(28, 28, 1).astype('float32') / 255.0
            label = int(filename.split("_")[-1].split(".")[0])
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_dataset()

# Train
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.3
)
model.save('digit_recognition.keras')
# Plot training curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()