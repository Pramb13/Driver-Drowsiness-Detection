import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

# Load the pre-trained model (make sure to specify the correct path if necessary)
model = load_model('drowsiness_model.h5')

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect drowsiness
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.reshape(roi_gray, (1, 64, 64, 1))
        
        # Prediction using the loaded model
        prediction = model.predict(roi_gray)
        
        # Determine the drowsiness state
        if prediction[0][0] > 0.5:
            label = "Drowsy"
        else:
            label = "Alert"
        
        # Draw bounding box and label on face
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame

# Convert OpenCV frame to PIL Image for Streamlit
def convert_frame_to_pil(frame):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return pil_img

# Streamlit app interface
st.title("Driver Drowsiness Detection System")
st.write("This application detects drowsiness in drivers using live camera feed.")

# Start video capture
video_capture = cv2.VideoCapture(0)

# Button to start detection
if st.button("Start Detection"):
    video_container = st.empty()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame = detect_drowsiness(frame)
        pil_frame = convert_frame_to_pil(frame)
        
        # Display the frame in Streamlit
        video_container.image(pil_frame)
        
        # Stop detection button
        if st.button("Stop Detection"):
            break

# Release video capture and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
