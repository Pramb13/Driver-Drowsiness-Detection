import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import tempfile
import time

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

model = load_model()

st.title("Live Driver Drowsiness Detection")
st.write("Detects drowsiness in real-time using YOLOv5.")

# Live camera input
st.markdown("### Activate Camera")
camera_input = st.camera_input("Start your camera to begin detection.")

if camera_input:
    st.markdown("### Live Detection Output")
    stframe = st.empty()  # Placeholder for video feed

    # Save the camera input temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(camera_input.getvalue())
        temp_video_path = temp_file.name

    # Open the video stream using OpenCV
    cap = cv2.VideoCapture(temp_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV default) to RGB for model compatibility
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLOv5 detection
        results = model(rgb_frame)
        results.render()  # Render detections on the frame

        # Convert rendered image back to BGR for Streamlit display
        rendered_frame = np.array(results.ims[0])
        stframe.image(rendered_frame, caption="Detected Frame", use_column_width=True)

        # Optional: Add a delay for smoother performance
        time.sleep(0.1)

    cap.release()
