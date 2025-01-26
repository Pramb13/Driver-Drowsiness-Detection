# Driver Drowsiness Detection System

import cv2
import numpy as np
import streamlit as st
from keras.models import load_model

# Load pre-trained model
model = load_model('drowsiness_model.h5')

# Function to detect drowsiness
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.reshape(roi_gray, (1, 64, 64, 1))
        prediction = model.predict(roi_gray)
        if prediction[0][0] > 0.5:
            label = "Drowsy"
        else:
            label = "Alert"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

# Streamlit app
st.title("Driver Drowsiness Detection System")
st.write("This application detects drowsiness in drivers using live camera feed.")

# Start video capture
video_capture = cv2.VideoCapture(0)

if st.button("Start Detection"):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = detect_drowsiness(frame)
        st.image(frame, channels="BGR")
        if st.button("Stop Detection"):
            break

video_capture.release()
cv2.destroyAllWindows()
