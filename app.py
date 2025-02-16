import streamlit as st
import torch
import cv2
import numpy as np
import os
import time
from PIL import Image
from io import BytesIO
import pygame  # For playing audio alerts

# Initialize pygame mixer for audio alerts
pygame.mixer.init()
alert_sound = "audio_alert.wav"

# Load YOLOv5 model
MODEL_PATH = "best.pt"
st.title("🚘 Driver Drowsiness Detection System")

@st.cache_resource
def load_model():
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def detect_drowsiness(image):
    if model is None:
        return None, "Model not loaded"
    
    # Convert image to format required by YOLOv5
    img_array = np.array(image)
    results = model(img_array)
    detections = results.pandas().xyxy[0]  # Get detected objects
    return detections, None

def play_alert():
    pygame.mixer.music.load(alert_sound)
    pygame.mixer.music.play()
    time.sleep(1)  # Allow alert to play

def main():
    st.sidebar.header("Upload an Image or Use Webcam")
    image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    use_webcam = st.sidebar.checkbox("Use Webcam")
    start_detection = st.sidebar.button("Start Detection")
    
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        detections, error = detect_drowsiness(image)
        if error:
            st.error(error)
        else:
            st.write(detections)
            if not detections.empty and 'drowsy' in detections['name'].values:
                st.warning("🚨 Drowsiness Detected! Playing alert sound.")
                play_alert()
    
    if use_webcam:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        while start_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, error = detect_drowsiness(frame_rgb)
            if error:
                st.error(error)
                break
            for _, row in detections.iterrows():
                x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                label = row["name"]
                conf = row["confidence"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if label == 'drowsy':
                    st.warning("🚨 Drowsiness Detected! Playing alert sound.")
                    play_alert()
            stframe.image(frame, channels="BGR")
        cap.release()

if __name__ == "__main__":
    main()
