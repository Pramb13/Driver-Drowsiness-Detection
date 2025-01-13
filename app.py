import streamlit as st
import cv2
import torch
from yolov5 import detect  # Ensure this matches your YOLOv5 setup
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

st.title("Driver Drowsiness Detection System")
st.write("This system detects driver drowsiness in real-time.")

# Upload or use webcam
option = st.selectbox("Select Input Source", ["Webcam", "Upload Video"])
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
else:
    st.write("Ensure your webcam is connected.")

# Process video
if st.button("Start Detection"):
    if option == "Upload Video" and uploaded_file is not None:
        cap = cv2.VideoCapture(uploaded_file.name)
    else:
        cap = cv2.VideoCapture(0)  # Use webcam

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLOv5 Detection
        results = model(frame_rgb)
        detections = results.pandas().xyxy[0]  # Bounding boxes and classes

        # Visualize detections
        for _, row in detections.iterrows():
            x1, y1, x2, y2, confidence, cls = (
                int(row['xmin']),
                int(row['ymin']),
                int(row['xmax']),
                int(row['ymax']),
                row['confidence'],
                row['class'],
            )
            label = results.names[int(cls)]
            if label in ["Closed_Eyes", "Face"]:
                color = (0, 0, 255) if label == "Closed_Eyes" else (0, 255, 0)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_rgb, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display frame in Streamlit
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

st.write("End of the Detection.")
