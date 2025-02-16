import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import base64
import io

# --- Page Config ---
st.set_page_config(page_title="Live Drowsiness Detection", layout="wide")

# --- JavaScript for Live Webcam (Continuous Frames) ---
live_webcam_code = """
    <video id="video" autoplay playsinline style="width: 100%; border-radius: 10px;"></video>
    <canvas id="canvas" style="display: none;"></canvas>
    <script>
        async function setupCamera() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            if (navigator.mediaDevices.getUserMedia) {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
            }

            setInterval(() => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                let imgData = canvas.toDataURL('image/jpeg');
                window.parent.postMessage(imgData, '*');
            }, 1000); // Capture every 1 second
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

# --- Capture Frames for Prediction ---
image_data = st.empty()  # Placeholder for incoming images
prediction_text = st.empty()  # Placeholder for prediction text

# --- Function to Predict Drowsiness ---
def predict_drowsiness(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    
    label = "Not Drowsy" if predicted_class_idx == 0 else "Drowsy"
    color = "green" if predicted_class_idx == 0 else "red"
    return label, confidence, color

# --- Receive Live Data ---
st.markdown("### **Live Prediction**")

while True:
    # 🔍 Debug: Check if image data is received
    msg = st.query_params.get("img", [""])[0]  # ✅ Updated query method
    if not msg:
        st.warning("No image received. Please check the webcam feed.")
        continue
    
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(msg.split(",")[1])
        img = Image.open(io.BytesIO(img_bytes))

        # 🔍 Debug: Show received image
        image_data.image(img, caption="Live Frame", use_column_width=True)

        # Predict Drowsiness
        label, confidence, color = predict_drowsiness(img)

        # Show prediction result
        prediction_text.markdown(
            f"### Prediction: <span style='color:{color};'>{label}</span> ({confidence:.2f})",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error processing image: {e}")
