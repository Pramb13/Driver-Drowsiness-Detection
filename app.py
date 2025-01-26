import cv2
import numpy as np
import streamlit as st
import time
import playsound
from scipy.spatial import distance as dist
from threading import Thread
import mediapipe as mp

# Setup Streamlit page
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("Driver Drowsiness Detection")

# Constants
thresh = 0.27
sound_path = "assets/alarm.wav"
drowsyTime = 1.5  # 1500ms for drowsiness detection

# Initialize mediapipe face mesh detector
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Setup camera
capture = cv2.VideoCapture(0)

# Functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def soundAlert(path):
    playsound.playsound(path)

def detect_face_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        return landmarks
    return None

def check_eye_status(landmarks):
    left_eye = [landmarks.landmark[i] for i in range(33, 133, 2)]
    right_eye = [landmarks.landmark[i] for i in range(362, 463, 2)]
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    
    ear = (left_ear + right_ear) / 2.0
    return ear < thresh

# Streamlit app logic
st.sidebar.header("Settings")
drowsy_time = st.sidebar.slider("Drowsiness Alert Time (seconds)", min_value=1, max_value=5, value=1, step=1)

frame_count = 0
drowsy = False

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(frame)

    if landmarks:
        eye_status = check_eye_status(landmarks)
        if eye_status:
            frame_count = 0
        else:
            frame_count += 1
            if frame_count > drowsy_time * 30:  # Assuming 30 FPS
                drowsy = True
                thread = Thread(target=soundAlert, args=(sound_path,))
                thread.setDaemon(True)
                thread.start()

    # Display result on Streamlit
    st.image(frame, channels="RGB", use_column_width=True)
    if drowsy:
        st.warning("**Drowsiness Alert!** Please take a break.")

# Release camera resources
capture.release()
