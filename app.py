import streamlit as st
import cv2
import torch
import dlib
from imutils import face_utils
import numpy as np
import time
from scipy.spatial import distance as dist

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Load dlib face detector and landmark predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Make sure to download this file from dlib
try:
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError as e:
    st.error(f"Error loading shape predictor: {e}")
    st.stop()

detector = dlib.get_frontal_face_detector()

# Define EAR threshold and counters for drowsiness detection
EAR_THRESHOLD = 0.2
blink_counter = 0
drowsiness_counter = 0

# Eye Aspect Ratio (EAR) calculation function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to process webcam input
def process_video(video):
    global blink_counter, drowsiness_counter
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame")
            break

        # Convert frame to grayscale for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using dlib
        faces = detector(gray)

        for face in faces:
            # Get the landmarks for the face
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Get the coordinates for the left and right eyes
            left_eye = shape[42:48]
            right_eye = shape[36:42]

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # If EAR is below the threshold, increase blink_counter
            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                # If eyes are open, reset blink_counter
                if blink_counter > 0:
                    drowsiness_counter += 1  # Increase drowsiness counter
                    blink_counter = 0  # Reset blink counter

            # Display EAR on the frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # If drowsiness counter reaches threshold, display alert
            if drowsiness_counter > 10:
                cv2.putText(frame, "DROWSINESS ALERT!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # Draw bounding boxes around the eyes
            cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)

        # Perform YOLOv5 detection on the frame (for visualization of face and objects)
        results = model(frame)
        results.render()  # Render the detection boxes on the frame

        # Show the processed frame in Streamlit
        st.image(frame, channels="BGR", caption="Drowsiness Detection", use_column_width=True)

        time.sleep(0.1)  # Add a small delay to control frame rate

# Create Streamlit Interface
st.title("Driver Drowsiness Detection")

# Webcam input for video stream
video = st.camera_input("Capture Video")

if video is not None:
    process_video(video)
else:
    st.write("Please allow access to the camera.")
