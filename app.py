import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Load Model
@st.cache_resource
def load_model():
    """Load pre-trained model from Hugging Face."""
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

# Custom Video Processing Class
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
        text = f"{label} ({confidence:.2f})"
        draw.text((10, 10), text, fill=color)

        return av.VideoFrame.from_ndarray(np.array(img), format="rgb24")

# Streamlit App UI
st.title("🚗 Live Drowsiness Detection System")
st.markdown("This app uses a deep learning model to detect drowsiness in real time.")

# Start Webcam Stream
webrtc_streamer(key="drowsiness", video_processor_factory=VideoProcessor)
