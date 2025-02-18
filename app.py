import streamlit as st
import cv2
import numpy as np
import dlib
import time
from scipy.spatial import distance
from PIL import Image

# Constants
ALERT_SOUND = "audio_alert.wav"
EYE_AR_THRESH = 0.25  # Eye Aspect Ratio threshold for drowsiness
EYE_AR_CONSEC_FRAMES = 20  # Number of consecutive frames to trigger an alert

st.title("🚘 Driver Drowsiness Detection System (Without YOLOv5)")

# Load Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib website

# Eye landmarks
(left_start, left_end) = (42, 48)
(right_start, right_end) = (36, 42)

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR)."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(image):
    """Detects drowsiness in an image and returns the annotated image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw eyes
        for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Check drowsiness
        if avg_ear < EYE_AR_THRESH:
            cv2.putText(image, "DROWSINESS ALERT!", (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                shape = predictor(gray, face)
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

                left_eye = shape[left_start:left_end]
                right_eye = shape[right_start:right_end]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                # Draw landmarks
                for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Check drowsiness
                if avg_ear < EYE_AR_THRESH:
                    drowsy_frame_count += 1
                    if drowsy_frame_count >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (face.left(), face.top() - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        st.warning("🚨 Drowsiness Detected! Playing alert sound.")
                        play_alert()
                else:
                    drowsy_frame_count = 0

            stframe.image(frame, channels="BGR")

        cap.release()
