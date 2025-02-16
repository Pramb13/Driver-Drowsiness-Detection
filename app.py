import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# Load your YOLOv5 model (make sure it's trained and exported correctly)
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Change path to your model
    model.eval()
    return model

model = load_model()

# Define a Video Processor to handle video frames
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to NumPy array
        img_resized = cv2.resize(img, (640, 640))  # Resize for YOLOv5

        # Convert frame to PyTorch tensor
        img_tensor = self.transform(img_resized).unsqueeze(0)

        # Perform inference
        results = self.model(img_tensor)
        detections = results.pandas().xyxy[0]  # Extract bounding boxes

        # Draw detections
        for _, row in detections.iterrows():
            if row['confidence'] > 0.5:  # Adjust threshold as needed
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{row['name']} {row['confidence']:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("🚗 Real-Time Drowsiness Detection")
st.write("This application detects drowsiness using a deep learning model.")

# Start WebRTC Stream
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
)
