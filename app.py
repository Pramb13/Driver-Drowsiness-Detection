import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load Model
@st.cache_resource
def load_model():
    """Load the pre-trained deep learning model and feature extractor"""
    model = AutoModelForImageClassification.from_pretrained("facebook/dino-vits16")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits16")
    return model, feature_extractor

# Initialize Model
model, feature_extractor = load_model()

# Predict Function
def predict(model, inputs):
    """Run inference on the model and return predicted label & confidence score"""
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    return predicted_class_idx, confidence

# Custom Video Processing Class for Real-Time Detection
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to NumPy array

        # Preprocess Frame
        inputs = feature_extractor(images=Image.fromarray(img), return_tensors="pt")
        predicted_class_idx, confidence = predict(model, inputs)

        # Annotate Frame with Prediction
        label = "Not Drowsy" if predicted_class_idx == 0 else "Drowsy"
        color = (0, 255, 0) if predicted_class_idx == 0 else (0, 0, 255)
        cv2.putText(img, f"{label} ({confidence:.2f})", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return img

# Streamlit App UI
st.title("🚗 Live Drowsiness Detection System")
st.markdown("This app uses a deep learning model to detect drowsiness in real time from webcam video.")

# Start Webcam Stream
webrtc_streamer(key="drowsiness", video_processor_factory=VideoProcessor)
