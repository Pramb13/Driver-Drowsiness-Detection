import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import numpy as np

# Initialize MediaPipe for face detection and face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to check if eyes are closed using Eye Aspect Ratio (EAR)
def is_eye_closed(eye_points, landmarks):
    ear = (abs(landmarks[eye_points[1]].y - landmarks[eye_points[5]].y) + 
           abs(landmarks[eye_points[2]].y - landmarks[eye_points[4]].y)) / (2.0 * abs(landmarks[eye_points[0]].x - landmarks[eye_points[3]].x))
    return ear < 0.2

# Define the video transformer for processing frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        # Convert frame to RGB (required by MediaPipe)
        img = frame.to_rgb()
        
        # Process the frame for face landmarks
        results = self.face_mesh.process(img)
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(image=img, landmark_list=landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION)
                
                # Check if eyes are closed
                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]
                left_eye_closed = is_eye_closed(left_eye, landmarks.landmark)
                right_eye_closed = is_eye_closed(right_eye, landmarks.landmark)
                
                if left_eye_closed and right_eye_closed:
                    cv2.putText(img, "DROWSY!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(img, "AWAKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# Set up the WebRTC streamer
webrtc_streamer(key="driver-drowsiness", video_transformer_factory=VideoTransformer)
