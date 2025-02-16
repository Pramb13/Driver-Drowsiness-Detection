import os
import torch
import streamlit as st
import subprocess

# ✅ Install YOLOv5 if not installed
YOLO_DIR = "yolov5"
if not os.path.exists(YOLO_DIR):
    st.warning("Installing YOLOv5... Please wait ⏳")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"], check=True)

# ✅ Function to Load YOLOv5 Model
@st.cache_resource
def load_model():
    MODEL_PATH = "best.pt"
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please upload `best.pt` to your project directory.")
        return None
    try:
        model = torch.hub.load("yolov5", "custom", path=MODEL_PATH, source="local")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

st.title("💤 Real-Time Drowsiness Detection")

# ✅ Load Model
model = load_model()
if model:
    st.success("Model loaded successfully! 🚀")
else:
    st.error("Failed to load model. Please check logs.")
