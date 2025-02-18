import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp

# Initialize mediapipe face mesh model for detecting facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam capture
st.title("🚘 Driver Drowsiness Detection System")

# Constants
ALERT_SOUND = "audio_alert.wav"  # Make sure this file is in the right directory

def play_alert():
    try:
        with open(ALERT_SOUND, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
    except Exception as e:
        st.error(f"Error playing alert: {e}")

def calculate_eye_aspect_ratio(eye_landmarks):
    # Calculate the Euclidean distance between the two vertical eye landmarks
    # and the two horizontal eye landmarks to calculate EAR
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def detect_drowsiness(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for the eyes
            left_eye = []
            right_eye = []
            for i in [33, 133, 160, 158, 153, 144]:  # Left eye landmarks
                left_eye.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])
            for i in [362, 263, 387, 385, 380, 373]:  # Right eye landmarks
                right_eye.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])
            
            # Convert to numpy arrays for calculation
            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)

            # Calculate Eye Aspect Ratio (EAR)
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0  # Average EAR for both eyes

            # Set threshold for drowsiness detection
            EAR_THRESHOLD = 0.22

            # Check if EAR is below the threshold (indicating drowsiness)
            if ear < EAR_THRESHOLD:
                return True
    return False

def main():
    st.sidebar.header("Upload an Image or Use Webcam")
    image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    use_webcam = st.sidebar.checkbox("Use Webcam")
    
    if image_file is not None:
        image = st.image(image_file, caption="Uploaded Image", use_column_width=True)
        frame = np.array(image)
        if detect_drowsiness(frame):
            st.warning("🚨 Drowsiness Detected! Playing alert sound.")
            play_alert()
    
    if use_webcam:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)  # Accessing the webcam
        
        if not cap.isOpened():
            st.error("❌ Failed to access webcam")
            return
        
        start_detection = st.sidebar.button("Start Detection")
        
        while start_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame")
                break
            
            # Convert frame to RGB for Mediapipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = []
                    right_eye = []
                    for i in [33, 133, 160, 158, 153, 144]:
                        left_eye.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])
                    for i in [362, 263, 387, 385, 380, 373]:
                        right_eye.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])
                    
                    left_eye = np.array(left_eye)
                    right_eye = np.array(right_eye)

                    left_ear = calculate_eye_aspect_ratio(left_eye)
                    right_ear = calculate_eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    EAR_THRESHOLD = 0.22
                    if ear < EAR_THRESHOLD:
                        st.warning("🚨 Drowsiness Detected! Playing alert sound.")
                        play_alert()
            
            # Display frame with detection
            stframe.image(frame, channels="BGR")
        
        cap.release()

if __name__ == "__main__":
    main()
