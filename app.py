import torch
import os
import urllib.request
import streamlit as st

# ✅ Set model path
MODEL_PATH = "best.pt"
MODEL_URL = "https://your-cloud-storage.com/best.pt"  # Replace with actual model link

# ✅ Function to Load Model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model... Please wait.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, source="local")
    return model

st.title("💤 Real-Time Drowsiness Detection")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
