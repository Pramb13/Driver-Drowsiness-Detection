import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh()

def calculate_ear(landmarks):
    # Calculate the Eye Aspect Ratio (EAR)
    left_eye = landmarks[mp_face_mesh.FACEMESH_LEFT_EYE]
    right_eye = landmarks[mp_face_mesh.FACEMESH_RIGHT_EYE]

    # Calculate distances
    left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])
    right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])
    left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
    right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])

    ear = (left_eye_height + right_eye_height) / (2.0 * (left_eye_width + right_eye_width))
    return ear

def process_frame(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Calculate EAR
            ear = calculate_ear(face_landmarks.landmark)
            if ear < 0.2:  # Threshold for drowsiness
                cv2.putText(frame, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "AWAKE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame
