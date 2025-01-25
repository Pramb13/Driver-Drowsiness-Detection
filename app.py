import streamlit as st
import torch
import numpy as np
import time
import imageio
from PIL import Image
import tempfile
import cv2
import mediapipe as mp  # For facial landmark detection

# Load YOLOv5 model
@st.cache_resource
def load_model():
    # Load custom YOLOv5 model (replace 'best.pt' with your model path)
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

model = load_model()

# Initialize MediaPipe Face Detection and Landmark Models
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.5)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks (synergy points)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for drowsiness
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 48
frame_counter = 0

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

    # Open video using imageio
    video_reader = imageio.get_reader(temp_video_path)

    for frame in video_reader:
        # Convert from BGR (imageio default) to RGB for model compatibility
        rgb_frame = np.array(frame)[..., :3]  # Ensure it's RGB and remove alpha channel if any

        # YOLOv5 detection for face detection
        results = model(rgb_frame)
        results.render()  # Render detections on the frame

        # Convert the rendered image back to RGB for Streamlit display
        rendered_frame = np.array(results.ims[0])

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MediaPipe
        results_face_detection = face_detection.process(rgb_frame)
        
        if results_face_detection.detections:
            # Use MediaPipe Face Mesh for landmark detection
            results_face_mesh = face_mesh.process(rgb_frame)
            
            if results_face_mesh.multi_face_landmarks:
                for face_landmarks in results_face_mesh.multi_face_landmarks:
                    # Get the coordinates of the left and right eyes (indices 33-133 for left eye, 362-463 for right eye)
                    left_eye = []
                    right_eye = []

                    for i in range(33, 133):  # Left eye landmarks
                        left_eye.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])
                    for i in range(362, 463):  # Right eye landmarks
                        right_eye.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])

                    left_eye = np.array(left_eye)
                    right_eye = np.array(right_eye)

                    # Calculate the EAR for both eyes
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    # Check if the EAR is below the threshold (indicating drowsiness)
                    if ear < EAR_THRESHOLD:
                        frame_counter += 1
                        if frame_counter >= CONSEC_FRAMES:
                            # Driver is drowsy
                            cv2.putText(rendered_frame, "Drowsy Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        frame_counter = 0

        # Display the rendered frame with drowsiness detection
        stframe.image(rendered_frame, caption="Detected Frame", use_column_width=True)

        # Optional: Add a delay for smoother performance
        time.sleep(0.1)

    video_reader.close()
