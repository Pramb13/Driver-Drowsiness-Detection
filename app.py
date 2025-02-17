import streamlit as st
import torch
import cv2
import numpy as np
import os
from PIL import Image

# Constants
MODEL_PATH = "best.pt"
ALERT_SOUND = "audio_alert.wav"

# Title
st.title("🚘 Driver Drowsiness Detection System")

# Function to load the YOLOv5 model
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True, trust_repo=True
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function for drowsiness detection
def detect_drowsiness(image):
    if model is None:
        return None, "Model not loaded"
    
    img_array = np.array(image)
    results = model(img_array)
    
    if results is None:
        return None, "No detections found"
    
    detections = results.pandas().xyxy[0]  # Get detected objects
    return detections, None

# Function to play alert sound
def play_alert():
    try:
        with open(ALERT_SOUND, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
    except Exception as e:
        st.error(f"Error playing alert: {e}")

# Main function
def main():
    st.sidebar.header("Upload an Image or Use Webcam")
    
    # Upload Image
    image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Webcam Toggle
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False

    if st.sidebar.button("Toggle Webcam"):
        st.session_state.webcam_active = not st.session_state.webcam_active

    # Process Uploaded Image
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        detections, error = detect_drowsiness(image)
        if error:
            st.error(error)
        else:
            st.write(detections)
            if not detections.empty and "drowsy" in detections["name"].values:
                st.warning("🚨 Drowsiness Detected! Playing alert sound.")
                play_alert()

    # Process Webcam Feed
    if st.session_state.webcam_active:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Failed to access webcam")
            return

        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, error = detect_drowsiness(frame_rgb)
            
            if error:
                st.error(error)
                break
            
            # Draw bounding boxes
            for _, row in detections.iterrows():
                x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                label = row["name"]
                conf = row["confidence"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if label == "drowsy":
                    st.warning("🚨 Drowsiness Detected! Playing alert sound.")
                    play_alert()

            stframe.image(frame, channels="BGR")

        cap.release()

if __name__ == "__main__":
    main()
