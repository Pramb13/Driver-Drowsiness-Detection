import cv2
import streamlit as st
import mediapipe as mp
import time

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to check if eyes are closed
def is_eye_closed(eye_points, landmarks):
    try:
        # Calculate Eye Aspect Ratio (EAR)
        eye_aspect_ratio = (abs(landmarks[eye_points[1]].y - landmarks[eye_points[5]].y) + 
                            abs(landmarks[eye_points[2]].y - landmarks[eye_points[4]].y)) / (2.0 * abs(landmarks[eye_points[0]].x - landmarks[eye_points[3]].x))
        return eye_aspect_ratio < 0.2  # Threshold for closed eyes (adjust as needed)
    except:
        return False  # Return False if landmarks are missing

# Streamlit app interface
st.title("Driver Drowsiness Detection")
st.write("This app detects the driver's drowsiness by checking their eye status in real time.")

# Initialize webcam
stframe = st.empty()
video_stream = cv2.VideoCapture(0)

if not video_stream.isOpened():
    st.error("Error: Could not access the camera. Please check if the camera is connected or in use.")
else:
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = video_stream.read()

            # Handle the case where the frame is None
            if not ret or frame is None:
                st.warning("No frame captured. Retrying...")
                time.sleep(0.1)
                continue

            # Convert to RGB and flip horizontally for a mirror view
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)

            # Process the frame for face landmarks
            result = face_mesh.process(frame)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # Draw face landmarks
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION
                    )

                    # Define eye points
                    left_eye = [33, 160, 158, 133, 153, 144]
                    right_eye = [362, 385, 387, 263, 373, 380]

                    # Check if eyes are closed
                    left_eye_closed = is_eye_closed(left_eye, face_landmarks.landmark)
                    right_eye_closed = is_eye_closed(right_eye, face_landmarks.landmark)

                    # Display status
                    status = "DROWSY!" if left_eye_closed and right_eye_closed else "AWAKE"
                    color = (0, 0, 255) if status == "DROWSY!" else (0, 255, 0)
                    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Show the frame in Streamlit
            stframe.image(frame, channels="RGB", use_column_width=True)

            # Add delay to reduce CPU usage
            time.sleep(0.1)

# Release resources
video_stream.release()
