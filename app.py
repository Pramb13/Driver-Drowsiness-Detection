import streamlit as st
import torch
import pandas as pd
import av
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
USER_CREDENTIALS = {"user": "123"}
ADMIN_CREDENTIALS = {"admin": "admin123"}

# Store session predictions
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

@st.cache_resource
def load_model():
    """ Load the image classification model and feature extractor """
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, feature_extractor):
    """ Convert and preprocess image for model input """
    image = image.convert("RGB")
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    """ Get model prediction and confidence score """
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            prediction_score = probabilities[0, predicted_class_idx].item()

        return predicted_class_idx, prediction_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

class VideoProcessor(VideoProcessorBase):
    """ Video processing class for real-time drowsiness detection """
    def __init__(self):
        self.model, self.feature_extractor = load_model()

    def recv(self, frame):
        """ Process each frame and make predictions """
        try:
            img = frame.to_image()
            inputs = preprocess_image(img, self.feature_extractor)
            predicted_class_idx, prediction_score = get_prediction(self.model, inputs)

            # Add prediction to session state
            if predicted_class_idx is not None:
                prediction
