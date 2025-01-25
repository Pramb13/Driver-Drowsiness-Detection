import cv2
import streamlit as st
import mediapipe as mp
import time
from imutils.video import VideoStream
import imutils

# Initialize mediapipe face detection and landmarks model
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to check if eyes are closed
def is_eye_closed(eye_points, landmarks):
    # Calculate Eye Aspect Ratio (EAR)
    eye_aspect_ratio = (abs(landmarks[eye_points[1]].y - landmarks[eye_points[5]].y) + 
                        abs(landmarks[eye_points[2]].y - landmarks[eye_points[4]].y)) / (2.0 * abs(landmarks[eye_points[0]].x - landmarks[eye_points[3]].x))
    return eye_aspect_ratio < 0.2  # Threshold for closed eyes (you can tune this value)

# Streamlit app interface
st.title("Driver Drowsiness Detection")
st.write("This app detects the driver's drowsiness by checking their eye status in real time.")

# Capture video
stframe = st.empty()
video_stream = VideoStream(src=0).start()

# Initialize mediapipe face mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        # Capture the frame from the video stream
        frame = video_stream.read()
        
        # Ensure the frame is valid
        if frame is None:
            st.write("Failed to capture frame. Please check your camera.")
            break
        
        frame = imutils.resize(frame, width=640)  # Resize the frame to 640px width
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (required by mediapipe)
        
        # Process the image and find face landmarks
        result = face_mesh.process(rgb_frame)
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Draw face landmarks on the frame
                mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION)
                
                # Eye region points (for left and right eyes)
                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]
                
                # Check if eyes are closed using the eye aspect ratio
                left_eye_closed = is_eye_closed(left_eye, face_landmarks.landmark)
                right_eye_closed = is_eye_closed(right_eye, face_landmarks.landmark)
                
                # Show alert if eyes are closed (drowsy), otherwise show awake
                if left_eye_closed and right_eye_closed:
                    cv2.putText(frame, "DROWSY!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "AWAKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Add a small delay to prevent high CPU usage
        time.sleep(0.1)
