import os
import torch
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import cv2

# 🔹 Load Model Function
@st.cache_resource
def load_model():
    MODEL_PATH = "best.pt"

    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Model file not found! Please upload `best.pt`.")
        return None
    
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, source="local", force_reload=True)
        return model
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None

# 🔹 Title & App Description
st.title("💤 Real-Time Drowsiness Detection")
st.write("This application detects drowsiness using a deep learning model.")

# 🔹 Load Model
model = load_model()
if model:
    st.success("✅ Model loaded successfully!")

# 🔹 Live Camera Feed
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Convert to RGB (YOLO requires RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLOv5 Inference
    results = model(img_rgb)

    # Draw bounding boxes & labels
    for *xyxy, conf, cls in results.xyxy[0]:  
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 🔹 Start WebRTC Streaming
webrtc_streamer(key="drowsiness-detection", video_frame_callback=video_frame_callback)
