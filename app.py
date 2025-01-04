import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import io

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to make detections
def detect_object(img):
    results = model(img)
    return results

# Streamlit UI
st.title("YOLOv5 Object Detection")
st.write("Upload an image or use webcam for real-time object detection.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Convert image to numpy array and make predictions
    img = np.array(img)
    results = detect_object(img)
    
    # Display results
    results.print()  # Print results in the Streamlit console
    st.write("Detection Results")
    st.image(np.squeeze(results.render()))  # Render detections on the image
else:
    st.write("Use webcam for real-time object detection.")

# Real-time webcam detection
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make detections
        results = detect_object(frame)
        
        # Display frame with detection
        cv2.imshow('YOLOv5', np.squeeze(results.render()))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
