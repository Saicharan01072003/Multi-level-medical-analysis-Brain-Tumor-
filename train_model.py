import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

# Path to the dataset directory
#TRAIN_IMAGE_DIR = r'C:\Users\saikr\OneDrive\Desktop\MRI_Error_Detection_Project\Brain MRI Images\Train'
#VALIDATION_IMAGE_DIR = r'C:\Users\saikr\OneDrive\Desktop\MRI_Error_Detection_Project\Brain MRI Images\Validation'
TRAIN_IMAGE_DIR = "Brain MRI Images\\Train"
VALIDATION_IMAGE_DIR = "Brain MRI Images\\Validation"

# Load lobe disadvantages from JSON file (optional)
with open('brain_lobes_info.json') as json_file:
    lobe_info = json.load(json_file)

# Preprocessing function for EfficientNet (224x224 RGB)
def preprocess_image_efficientnet(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read the image: {image_path}")
        img_resized = cv2.resize(img, (224, 224))  # Resize for EfficientNet
        img_normalized = img_resized / 255.0
        img_normalized = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
        img_normalized = np.repeat(img_normalized, 3, axis=-1)  # Convert grayscale to RGB
        return img_normalized
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# Preprocessing function for CNN model (256x256 grayscale)
def preprocess_image_cnn(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read the image: {image_path}")
        img_resized = cv2.resize(img, (256, 256))  # Resize for CNN
        img_normalized = img_resized / 255.0
        return img_normalized
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# Load dataset and labels
def load_images_and_labels(image_dir):
    images = []
    labels = []
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg')):
                    images.append(img_path)
                    labels.append(label)
    return images, labels

# Prepare the dataset
train_images, train_labels = load_images_and_labels(TRAIN_IMAGE_DIR)
validation_images, validation_labels = load_images_and_labels(VALIDATION_IMAGE_DIR)

# Encode labels (Normal: 0, Tumor: 1)
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
validation_labels_encoded = label_encoder.transform(validation_labels)

# Preprocess the images for EfficientNet and CNN
train_images_cnn = np.array([preprocess_image_cnn(img) for img in train_images])
train_images_efficientnet = np.array([preprocess_image_efficientnet(img) for img in train_images])

validation_images_cnn = np.array([preprocess_image_cnn(img) for img in validation_images])
validation_images_efficientnet = np.array([preprocess_image_efficientnet(img) for img in validation_images])

# CNN Model (for MRI Error Detection)
def create_cnn_model():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(256, 256, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# EfficientNet Model
def create_efficientnet_model():
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the CNN Model
cnn_model = create_cnn_model()
cnn_model.fit(train_images_cnn, train_labels_encoded, epochs=10, batch_size=32, validation_data=(validation_images_cnn, validation_labels_encoded))

# Save the CNN model
cnn_model.save('models/mri_error_detector.keras')

# Train the EfficientNet Model
efficientnet_model = create_efficientnet_model()
efficientnet_model.fit(train_images_efficientnet, train_labels_encoded, epochs=10, batch_size=32, validation_data=(validation_images_efficientnet, validation_labels_encoded))

# Save the EfficientNet model
efficientnet_model.save('models/efficientnet_b0_model.keras')

# Train the Random Forest model on flattened images (for additional analysis)
train_images_rf = np.array([cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (64, 64)).flatten() for img in train_images])
validation_images_rf = np.array([cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (64, 64)).flatten() for img in validation_images])

# Normalize the image data
scaler = StandardScaler()
train_images_rf = scaler.fit_transform(train_images_rf)
validation_images_rf = scaler.transform(validation_images_rf)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_images_rf, train_labels_encoded)

# Save the Random Forest model and scaler
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Models trained and saved successfully!")
