import streamlit as st
import torch
import numpy as np
import time
from PIL import Image
import cv2
import mediapipe as mp
import tempfile

# Load YOLOv5 model using the ultralytics package
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO('best.pt')  # Replace with actual model path

model = load_model()

# Mediapipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.25

st.title("Driver Drowsiness Detection")
st.write("Detects drowsiness using YOLOv5 and Mediapipe.")

# Streamlit Camera Input
camera_input = st.camera_input("Start your camera to begin detection.")

if camera_input:
    st.markdown("### Live Detection Output")
    stframe = st.empty()  # Placeholder for live video feed
    
    # Capture and process video stream
    video_bytes = camera_input.getvalue()
    
    # Save the image temporarily to process each frame
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_image_path = temp_file.name

    # Open and process the video stream
    image = cv2.imread(temp_image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLOv5 detection
    results = model(rgb_image)
    results.render()  # Render detections

    # Convert rendered image for display
    rendered_image = Image.fromarray(results.ims[0])

    # Mediapipe facial landmark detection
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    height, width, _ = rgb_image.shape
    landmarks = []

    with face_mesh as mesh:
        face_results = mesh.process(rgb_image)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Collect eye coordinates
                landmarks = [
                    (int(pt.x * width), int(pt.y * height))
                    for pt in face_landmarks.landmark
                ]

                # Define eye regions
                left_eye = landmarks[362:368]
                right_eye = landmarks[133:139]

                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(np.array(left_eye))
                right_ear = eye_aspect_ratio(np.array(right_eye))
                ear = (left_ear + right_ear) / 2.0

                # Annotate the image with drowsiness detection
                if ear < EAR_THRESHOLD:
                    cv2.putText(rgb_image, "Drowsy Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the processed frame
    stframe.image(rendered_image, caption="Processed Frame", use_column_width=True)
