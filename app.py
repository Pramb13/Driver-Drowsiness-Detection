import streamlit as st
import torch
import pandas as pd
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import time
import datetime
import os
import base64

# Disable Streamlit Watchdog Warning
os.environ["STREAMLIT_WATCHDOG"] = "0"

# Load Model
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("facebook/dino-vits16")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits16")
    return model, feature_extractor

def preprocess_image(image, feature_extractor):
    image = image.convert("RGB")
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        prediction_score = probabilities[0, predicted_class_idx].item()
    return predicted_class_idx, prediction_score

# Webcam with Live Stream (HTML5 + JS)
def webcam_stream():
    st.markdown(
        """
        <style>
        .video-container {
            display: flex;
            justify-content: center;
        }
        </style>
        <div class="video-container">
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas" style="display: none;"></canvas>
        </div>
        <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.play();
            } catch (err) {
                console.error("Error accessing the camera: ", err);
            }
        }

        function captureFrame() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/png');
        }

        function sendFrame() {
            const imageData = captureFrame();
            fetch('/upload_image', {
                method: 'POST',
                body: JSON.stringify({ image: imageData }),
                headers: { 'Content-Type': 'application/json' }
            });
        }

        setInterval(sendFrame, 3000); // Capture image every 3 seconds
        startCamera();
        </script>
        """,
        unsafe_allow_html=True
    )

# Admin Panel to View History
def admin_panel():
    st.title("Admin Dashboard")
    st.write("Below are the recorded predictions with timestamps:")
    if "predictions" in st.session_state:
        df = pd.DataFrame(st.session_state["predictions"])
        st.dataframe(df)
    else:
        st.write("No data available.")

# Main App
def main():
    st.title("Live Drowsiness Detection")
    webcam_stream()  # Show live camera feed

    model, feature_extractor = load_model()

    # Handle Image Upload from JS
    if st.experimental_get_query_params().get("image"):
        img_data = st.experimental_get_query_params()["image"]
        img_data = base64.b64decode(img_data.split(",")[1])
        image = Image.open(io.BytesIO(img_data))
        inputs = preprocess_image(image, feature_extractor)
        predicted_class_idx, prediction_score = get_prediction(model, inputs)

        label = ["Not Drowsy", "Drowsy"][predicted_class_idx]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.image(image, caption=f"Prediction: {label} (Score: {prediction_score:.2f})")

        if "predictions" not in st.session_state:
            st.session_state["predictions"] = []
        st.session_state["predictions"].append({
            "Timestamp": timestamp,
            "Prediction": label,
            "Score": f"{prediction_score:.2f}"
        })

if __name__ == "__main__":
    main()
