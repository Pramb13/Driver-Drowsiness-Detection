import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# --- Page Config ---
st.set_page_config(page_title="Live Drowsiness Detection", layout="wide")

# --- JavaScript for Live Webcam ---
live_webcam_code = """
    <video id="video" autoplay playsinline style="width: 100%; border-radius: 10px;"></video>
    <script>
        async function setupCamera() {
            const video = document.getElementById('video');
            if (navigator.mediaDevices.getUserMedia) {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
            }
        }
        setupCamera();
    </script>
"""

# --- Load Model ---
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("facebook/dino-vits16")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits16")
    return model, feature_extractor

model, feature_extractor = load_model()

# --- Streamlit UI ---
st.title("🚗 Live Drowsiness Detection System")
st.markdown("This app uses a deep learning model to detect drowsiness in real-time.")

# --- Embed Webcam Video ---
st.components.v1.html(live_webcam_code, height=300)

# --- Live Prediction Logic ---
uploaded_image = st.file_uploader("📷 Capture Frame for Prediction", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    
    # Preprocess Image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_idx].item()

    # Show Prediction
    label = "Not Drowsy" if predicted_class_idx == 0 else "Drowsy"
    color = "green" if predicted_class_idx == 0 else "red"

    # Display Result
    st.image(image, caption="Captured Frame", use_column_width=True)
    st.markdown(f"### Prediction: <span style='color:{color};'>{label}</span> ({confidence:.2f})", unsafe_allow_html=True)
