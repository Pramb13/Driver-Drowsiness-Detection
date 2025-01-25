import streamlit as st
import torch
import numpy as np
import time
import imageio
from PIL import Image
import tempfile
import cv2
import dlib  # For facial landmarks detection

# Load YOLOv5 model
@st.cache_resource
def load_model():
    # Load custom YOLOv5 model (replace 'best.pt' with your model path)
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

model = load_model()

# Load Dlib facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from Dlib

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

        # Convert the frame to grayscale for face and eye detection
        gray_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2GRAY)

        # Detect faces in the frame
        faces = detector(gray_frame)

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray_frame, face)

            # Get the coordinates of the left and right eyes (indices 36-41 and 42-47)
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

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
