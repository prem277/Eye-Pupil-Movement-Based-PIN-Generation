import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Dataset path
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')
print(f"Dataset path: {dataset_path}")


# Function to preprocess images
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    img = cv2.equalizeHist(img)  # Histogram equalization
    img = cv2.resize(img, (64, 64))  # Resize image to 64x64
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# Load and preprocess the data
data = []
labels = []
label_map = {'left': 0, 'right': 1}  # Define the label mapping

for label in label_map:
    dir_path = os.path.join(dataset_path, label)
    if os.path.isdir(dir_path):
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = preprocess_image(img_path)
            data.append(img)
            labels.append(label_map[label])

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)
labels = to_categorical(labels, num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Model Definition
def create_pupil_detection_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Create the model
input_shape = (64, 64, 1)
num_classes = 2
model = create_pupil_detection_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# Save the model
model.save('e:/Projects/PIN Generation/pupil_detection_model.keras')
# Verify if the model file is created
if os.path.isfile('e:/Projects/PIN Generation/pupil_detection_model.keras'):
    print("Model saved successfully.")
else:
    print("Model not saved.")