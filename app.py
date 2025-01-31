import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import os

# Display TensorFlow version for debugging
import tensorflow as tf
st.write(f"TensorFlow version: {tf.__version__}")

# Path to the pre-trained model (ensure the model is included in your app directory)
model_path = 'drowsiness_model.h5'  # or 'models/drowsiness_model.h5' if in a subfolder

# Load the model from the predefined path when the app starts
if os.path.exists(model_path):
    model = load_model(model_path)
    st.write("Model loaded successfully!")
else:
    st.error(f"Model file not found at the specified path: {model_path}")

# Access webcam and display video feed
camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")

if camera_input is not None and 'model' in locals():
    # Convert image from the uploaded image (from camera)
    image = cv2.imdecode(np.frombuffer(camera_input, np.uint8), cv2.IMREAD_COLOR)

    # Resize image to match the model input size (e.g., 224x224)
    image_resized = cv2.resize(image, (224, 224))

    # Preprocess the image (normalization, etc.)
    image_processed = image_resized / 255.0  # Normalize if needed

    # Make a prediction with the model
    prediction = model.predict(np.expand_dims(image_processed, axis=0))

    # Display the prediction (you can modify it based on your model's output)
    st.write(f"Prediction: {prediction}")

    # Optionally, display the captured image from the webcam
    st.image(image, channels="BGR", caption="Captured Image from Webcam")
else:
    st.write("Please check if the model is loaded or the webcam feed is active.")
