import streamlit as st
import cv2
import numpy as np
from drowsy_detection import process_frame

st.title("Real-Time Driver Drowsiness Detection System")

# Start video capture from webcam
video_capture = cv2.VideoCapture(0)

# Display the video stream
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process the frame for drowsiness detection
    processed_frame = process_frame(frame)

    # Display the processed frame
    st.image(processed_frame, channels="BGR", use_column_width=True)

# Release the video capture object
video_capture.release()
