import cv2
import dlib
import streamlit as st
from imutils import face_utils
from scipy.spatial import distance

# Load dlib's pre-trained face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye Aspect Ratio (EAR) calculation
def ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Setup for Streamlit
st.title("Driver Drowsiness Detection System")

# Set up video capture
cap = cv2.VideoCapture(0)

stframe = st.empty()  # This will hold our live stream

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Get the coordinates of the left and right eyes
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]

        # Calculate EAR for both eyes
        left_ear = ear(left_eye)
        right_ear = ear(right_eye)

        # Average EAR to determine drowsiness
        ear_avg = (left_ear + right_ear) / 2.0

        if ear_avg < 0.25:
            cv2.putText(frame, "Drowsy!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame in the Streamlit app
    stframe.image(frame, channels="BGR", use_column_width=True)

cap.release()
