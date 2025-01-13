import streamlit as st
import torch
from PIL import Image
import numpy as np
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

st.title("Live Driver Drowsiness Detection")
st.write("Detects drowsiness in real-time using YOLOv5.")

# Live camera input
st.markdown("### Activate Camera")
camera_input = st.camera_input("Start your camera to begin detection.")

if camera_input:
    st.markdown("### Live Detection Output")
    stframe = st.empty()  # Placeholder for video feed

    # Processing live frames
    while True:
        try:
            # Capture frame from camera
            img = Image.open(camera_input)

            # YOLOv5 detection
            results = model(np.array(img))
            results.render()  # Render detections on the frame

            # Convert rendered image back to PIL for Streamlit display
            detected_frame = Image.fromarray(results.ims[0])
            stframe.image(detected_frame, caption="Detected Frame", use_column_width=True)

            # Optional: Add a delay for smoother performance
            time.sleep(0.1)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            break
