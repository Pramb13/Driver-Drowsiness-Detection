import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.spatial import distance
from PIL import Image

# Constants
ALERT_SOUND = "audio_alert.wav"
EYE_AR_THRESH = 0.25  # Eye Aspect Ratio threshold for drowsiness
EYE_AR_CONSEC_FRAMES = 20  # Number of consecutive frames to trigger an alert

st.title("🚘 Driver Drowsiness Detection System (No Dlib)")

# Initialize MediaPipe face and landmarks detector
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start mediapipe FaceMesh model
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR)."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(image):
    """Detects drowsiness in an image and returns the annotated image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return image
    
    for landmarks in results.multi_face_landmarks:
        # Get eye landmarks (indices 33-133 for left eye, 362-463 for right eye)
        left_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(33, 133)]
        right_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(362, 463)]
        
        # Normalize landmark coordinates
        h, w, _ = image.shape
        left_eye = [(int(p[0] * w), int(p[1] * h)) for p in left_eye]
        right_eye = [(int(p[0] * w), int(p[1] * h)) for p in right_eye]
        
        # Calculate Eye Aspect Ratio (EAR)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Draw eyes
        for (x, y) in left_eye + right_eye:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # Check for drowsiness
        if ear < EYE_AR_THRESH:
            cv2.putText(image, "DROWSINESS ALERT!", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            play_alert()

    return image

def play_alert():
    """Play alert sound."""
    try:
        with open(ALERT_SOUND, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
    except Exception as e:
        st.error(f"Error playing alert: {e}")

# Sidebar controls
st.sidebar.header("📸 Upload an Image or Use Webcam")
image_file = st.sidebar.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

if st.sidebar.button("📷 Toggle Webcam"):
    st.session_state.webcam_active = not st.session_state.webcam_active

# Process Uploaded Image
if image_file is not None:
    image = Image.open(image_file)
    image = np.array(image)  # Convert to OpenCV format
    st.image(image, caption="📌 Uploaded Image", use_container_width=True)

    processed_image = detect_drowsiness(image)
    st.image(processed_image, caption="🔍 Processed Image", use_container_width=True)

# Webcam Processing
if st.session_state.webcam_active:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Failed to access webcam.")
    else:
        drowsy_frame_count = 0

        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = detect_drowsiness(frame_rgb)

            # Display processed frame
            stframe.image(processed_frame, channels="BGR")

        cap.release()
