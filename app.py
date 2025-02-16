import streamlit as st
import torch
import os
import subprocess
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Install YOLOv5 (if not available)
subprocess.run(["pip", "install", "ultralytics"], check=True)

# Load YOLOv5 Model
@st.cache_resource
def load_model():
    MODEL_PATH = "best.pt"

    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Model file not found! Please upload `best.pt` below.")
        return None

    try:
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=MODEL_PATH,
            source="github",
            force_reload=True,
        )
        st.success("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None


# Webcam Processing for Drowsiness Detection
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.model is not None:
            results = self.model(img)
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                img = cv2.putText(img, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit UI
st.title("🛑 Driver Drowsiness Detection")

# Upload model if missing
uploaded_file = st.file_uploader("Upload `best.pt` Model", type=["pt"])
if uploaded_file:
    with open("best.pt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Model uploaded successfully! Refresh to use it.")

# Load model
model = load_model()

# Start webcam
if model is not None:
    webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
else:
    st.warning("⚠️ Please upload a valid YOLOv5 model to proceed.")
