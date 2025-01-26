import streamlit as st
import cv2
import numpy as np
from drowsy_detection import process_frame

st.title("Driver Drowsiness Detection System")
uploaded_file = st.file_uploader("Upload a video file...", type=["mp4", "avi"])

if uploaded_file is not None:
    # Create a video capture object
    cap = cv2.VideoCapture(uploaded_file.name)

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for drowsiness detection
        processed_frame = process_frame(frame)

        # Display the processed frame
        st.image(processed_frame, channels="BGR", use_column_width=True)

    cap.release()
