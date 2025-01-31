import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import requests
import numpy as np

# Load Hugging Face model and feature extractor (for vision tasks)
model_name = "facebook/dino-vits16"  # Example model for image classification
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Display model details
st.write(f"Loaded model: {model_name}")

# Load an image from webcam
camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")

if camera_input is not None:
    # Open the image and preprocess it
    img = Image.open(camera_input)
    img = img.convert("RGB")

    # Preprocess the image for the model
    inputs = feature_extractor(images=img, return_tensors="pt")

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Get the raw prediction scores
        predicted_class_idx = torch.argmax(logits, dim=-1).item()

    # Display the image and prediction
    st.image(img, caption="Captured Image from Webcam", use_column_width=True)
    st.write(f"Prediction: Class {predicted_class_idx} with score {logits.max().item()}")
