import torch
import cv2
import numpy as np
import streamlit as st
import tempfile
import urllib.request
import os

# Streamlit Page Config
st.set_page_config(page_title="Real-Time Drowsiness Detection", layout="wide")

# Title
st.title("😴 Real-Time Drowsiness Detection")
st.write("This application detects drowsiness using a deep learning model.")

# Function to download the model if missing
@st.cache_resource
def load_model():
    try:
        model_path = "best.pt"

        # Download model if not available
        model_url = "https://your-cloud-storage.com/best.pt"  # Replace with your actual model URL
        if not os.path.exists(model_path):
            st.info("Downloading model file, please wait...")
            urllib.request.urlretrieve(model_url, model_path)
            st.success("Model downloaded successfully!")

        # Load YOLOv5 model
        model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, source="local")
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()
if model is None:
    st.stop()  # Stop execution if model fails to load

# Function to process frames
def detect_drowsiness(frame):
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = map(int, det)
            label = model.names[cls]
            color = (0, 255, 0) if label == "Awake" else (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        return frame

# Webcam Input
st.subheader("Webcam Feed for Real-Time Drowsiness Detection")
video_placeholder = st.empty()
start = st.button("START", key="start")

if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error accessing webcam")
            break
        
        frame = detect_drowsiness(frame)

        # Convert frame to bytes for Streamlit
        _, img_encoded = cv2.imencode(".jpg", frame)
        img_bytes = img_encoded.tobytes()
        video_placeholder.image(img_bytes, channels="RGB", use_column_width=True)

    cap.release()
