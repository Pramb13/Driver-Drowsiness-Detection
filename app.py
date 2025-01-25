import cv2
import streamlit as st
import mediapipe as mp
import time

# Streamlit page configuration
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("Driver Drowsiness Detection")
st.write("This app detects the driver's drowsiness by checking their eye status in real time.")

# Initialize Mediapipe and utility functions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR) for eye closure detection
def is_eye_closed(eye_points, landmarks):
    eye_aspect_ratio = (
        abs(landmarks[eye_points[1]].y - landmarks[eye_points[5]].y) +
        abs(landmarks[eye_points[2]].y - landmarks[eye_points[4]].y)
    ) / (2.0 * abs(landmarks[eye_points[0]].x - landmarks[eye_points[3]].x))
    return eye_aspect_ratio < 0.2  # Adjust this threshold as needed

# Streamlit Cloud environment may have restrictions; check for a working camera
def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return False
    cap.release()
    return True

# Attempt to access the camera
camera_available = check_camera()

if not camera_available:
    st.error("Error: Could not access the camera. Please check if the camera is connected or in use.")
else:
    stframe = st.empty()  # Placeholder for video frames
    video_stream = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = video_stream.read()
            if not ret:
                st.error("Failed to capture frame. Please refresh the page and try again.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = cv2.flip(frame, 1)  # Mirror the frame

            # Process frame for face landmarks
            result = face_mesh.process(frame)
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # Draw landmarks on the face
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION
                    )

                    # Define eye landmarks for left and right eyes
                    left_eye = [33, 160, 158, 133, 153, 144]
                    right_eye = [362, 385, 387, 263, 373, 380]

                    # Detect eye closure
                    left_eye_closed = is_eye_closed(left_eye, face_landmarks.landmark)
                    right_eye_closed = is_eye_closed(right_eye, face_landmarks.landmark)

                    # Display drowsiness status
                    if left_eye_closed and right_eye_closed:
                        cv2.putText(frame, "DROWSY!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "AWAKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame in Streamlit
            stframe.image(frame, channels="RGB", use_column_width=True)

            # Limit FPS to reduce CPU usage
            time.sleep(0.1)

    video_stream.release()
