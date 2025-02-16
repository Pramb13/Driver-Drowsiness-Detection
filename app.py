import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
import os
from PIL import Image
from yolov5 import YOLOv5

# Load YOLOv5 Model (Make sure you have the model file in the same directory or provide the path)
MODEL_PATH = "best.pt"  # Change this to your actual model path
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLOv5(MODEL_PATH, device)

st.title("🚗 Live Drowsiness Detection System")
st.write("This app uses a deep learning model to detect drowsiness in real time.")

# Webcam Capture Section
st.subheader("📷 Live Camera Feed")
video_file = st.camera_input("Capture a frame for prediction")

if video_file is not None:
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(video_file.getvalue())
        temp_path = temp_file.name

    # Read Image
    image = Image.open(temp_path)
    image = np.array(image)

    # Convert to BGR for OpenCV Processing
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLOv5 Model for Prediction
    results = model.predict(image)

    # Draw Bounding Boxes
    for result in results.xyxy[0]:  # Extract first frame's results
        x1, y1, x2, y2, conf, cls = result
        label = f"Drowsy ({conf:.2f})" if int(cls) == 1 else "Alert"
        color = (0, 0, 255) if int(cls) == 1 else (0, 255, 0)

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Convert back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display Processed Image with Predictions
    st.image(image, caption="Live Prediction", use_column_width=True)

    # Remove Temp File
    os.remove(temp_path)
