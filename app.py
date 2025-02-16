import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import urllib.request
from PIL import Image

# ✅ Set page title and description
st.set_page_config(page_title="Real-Time Drowsiness Detection", layout="wide")
st.title("💤 Real-Time Drowsiness Detection")
st.write("This application detects drowsiness using a deep learning model.")

# ✅ Function to download the model if missing
@st.cache_resource
def load_model():
    model_path = "best.pt"
    model_url = "https://your-cloud-storage.com/best.pt"  # 🔹 Replace with actual model URL

    # Download model if not found
    if not os.path.exists(model_path):
        st.info("Downloading model file, please wait...")
        urllib.request.urlretrieve(model_url, model_path)
        st.success("Model downloaded successfully!")

    # Load YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, source="local")
    model.eval()
    return model

# ✅ Load the YOLO model
model = load_model()

# ✅ Function to process the image and detect drowsiness
def detect_drowsiness(image):
    img = np.array(image.convert("RGB"))
    results = model(img)  # Run YOLOv5 model
    return results

# ✅ Webcam feed
st.write("### Webcam feed for real-time drowsiness detection")
video = st.camera_input("Click 'Take Photo' to capture an image")

if video:
    # Save uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(video.getvalue())
        temp_path = temp_file.name

    # Read image with OpenCV
    img = Image.open(temp_path)
    
    # Detect drowsiness
    results = detect_drowsiness(img)

    # Display image with detections
    st.image(results.render()[0], caption="Detection Results", use_column_width=True)
