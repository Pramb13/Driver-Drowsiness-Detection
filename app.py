import streamlit as st
import cv2
import numpy as np
import time
import pinecone
import dlib
from imutils import face_utils
import os

# Initialize Pinecone API
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east1-gcp")  # Initialize Pinecone client

# Drowsiness Detection Parameters
EAR_THRESHOLD = 0.3  # Eye aspect ratio threshold for drowsiness detection
EYE_AR_CONSEC_FRAMES = 48  # Number of frames to indicate drowsiness

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib model

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Create Pinecone index
index_name = "driver-drowsiness"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Dimension of embeddings, can be modified
        metric='cosine'
    )

# Create an index instance
index = pinecone.Index(index_name)

class DrowsinessDetection:
    def __init__(self):
        self.drowsy_frame_count = 0
        self.drowsy = False

    def detect_drowsiness(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Get the coordinates of the left and right eyes
            left_eye = shape[42:48]
            right_eye = shape[36:42]

            # Calculate the EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Compute the average EAR
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below threshold
            if ear < EAR_THRESHOLD:
                self.drowsy_frame_count += 1
                if self.drowsy_frame_count >= EYE_AR_CONSEC_FRAMES:
                    self.drowsy = True
            else:
                self.drowsy_frame_count = 0
                self.drowsy = False

        return self.drowsy

    def store_in_pinecone(self, drowsy_state, frame):
        # Generate embeddings (mock embeddings for now, you can use actual face embeddings)
        embedding = np.random.rand(384).tolist()  # Mock embeddings
        metadata = {"drowsy_state": drowsy_state, "timestamp": time.time()}

        upsert_data = [
            ("drowsy_state", embedding, metadata)
        ]

        try:
            response = index.upsert(vectors=upsert_data)
            st.write(f"Upsert response: {response}")
        except Exception as e:
            st.error(f"Error during Pinecone upsert: {str(e)}")

# Main function
def main():
    st.title("Driver Drowsiness Detection System")
    st.write("Detect whether the driver is drowsy based on their eye movements.")

    # Start webcam feed
    cap = cv2.VideoCapture(0)
    drowsiness_detector = DrowsinessDetection()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect drowsiness in the current frame
        is_drowsy = drowsiness_detector.detect_drowsiness(frame)

        # Show the result on the frame
        if is_drowsy:
            cv2.putText(frame, "Drowsy!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame using OpenCV
        cv2.imshow('Frame', frame)

        # If drowsiness detected, store the result in Pinecone
        if is_drowsy:
            drowsiness_detector.store_in_pinecone("Drowsy", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the Streamlit app
if __name__ == "__main__":
    main()
