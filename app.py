import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Load Model
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("facebook/dino-vits16")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits16")
    return model, feature_extractor

# Preprocess Frame
def preprocess_frame(frame, feature_extractor):
    image = Image.fromarray(frame)  # Convert OpenCV frame (numpy array) to PIL Image
    return feature_extractor(images=image, return_tensors="pt")

# Predict from Frame
def predict(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    return predicted_class_idx, confidence

# Live Video Feed
def video_feed():
    st.title("Live Drowsiness Detection")
    model, feature_extractor = load_model()

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return
    
    # Stream Video
    stframe = st.empty()  # Placeholder for video feed
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process Frame
        inputs = preprocess_frame(frame, feature_extractor)
        predicted_class_idx, confidence = predict(model, inputs)

        # Annotate Frame with Prediction
        label = "Not Drowsy" if predicted_class_idx == 0 else "Drowsy"
        color = (0, 255, 0) if predicted_class_idx == 0 else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Convert BGR to RGB and Display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()

# Run Video Feed
if __name__ == "__main__":
    video_feed()
