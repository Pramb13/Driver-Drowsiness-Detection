import streamlit as st
import torch
import requests
import cv2
import numpy as np
import tempfile
import os

# ------------------------------#
# **CONFIGURATION & MODEL LOADING**
# ------------------------------#

st.title("🚘 Live Drowsiness Detection System")
st.write("This app uses a deep learning model to detect drowsiness in real-time.")

MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?id=YOUR_DRIVE_FILE_ID"  # Replace with your actual file ID

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading YOLOv5 model..."):
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded successfully!")

# Load YOLOv5 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)

# ------------------------------#
# **LIVE WEBCAM STREAM & PREDICTION**
# ------------------------------#

st.subheader("📷 Live Prediction")

# OpenCV video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Failed to open webcam. Please allow camera permissions.")

FRAME_WINDOW = st.image([])

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("No frame captured. Check webcam connection.")
        break

    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform YOLOv5 detection
    results = model(frame_rgb)

    # Render predictions
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        color = (0, 255, 0) if int(cls) == 0 else (0, 0, 255)  # Green for awake, Red for drowsy
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display frame
    FRAME_WINDOW.image(frame_rgb)

# Release resources
cap.release()
cv2.destroyAllWindows()
