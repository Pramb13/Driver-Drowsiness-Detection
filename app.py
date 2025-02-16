import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load Model
@st.cache_resource
def load_model():
    """Load the pre-trained deep learning model and feature extractor."""
    model = AutoModelForImageClassification.from_pretrained("facebook/dino-vits16")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits16")
    return model, feature_extractor

# Initialize Model
model, feature_extractor = load_model()

# Prediction Function
def predict(model, image):
    """Run inference and return predicted label & confidence score."""
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    return predicted_class_idx, confidence

# Custom Video Processing Class (No OpenCV)
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = Image.fromarray(frame.to_ndarray(format="rgb24"))  # Convert frame to PIL Image

        # Predict Drowsiness
        predicted_class_idx, confidence = predict(model, img)

        # Annotate Frame with Prediction
        label = "Not Drowsy" if predicted_class_idx == 0 else "Drowsy"
        color = (0, 255, 0) if predicted_class_idx == 0 else (255, 0, 0)

        # Draw Text on Image
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 30)  # Try system font
        except:
            font = None  # Default to basic font if not found
        text = f"{label} ({confidence:.2f})"
        draw.text((50, 50), text, fill=color, font=font)

        return img

# Streamlit App UI
st.title("🚗 Live Drowsiness Detection System")
st.markdown("This app uses a deep learning model to detect drowsiness in real time from webcam video.")

# Start Webcam Stream
webrtc_streamer(key="drowsiness", video_processor_factory=VideoProcessor)
