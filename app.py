import streamlit as st
import torch
import pandas as pd
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import time
import cv2
import numpy as np

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
USER_CREDENTIALS = {"user": "123"}
ADMIN_CREDENTIALS = {"admin": "admin123"}

if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

def authenticate(username, password, role):
    if role == "User" and username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True
    elif role == "Admin" and username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        return True
    return False

@st.cache_resource
def load_model():
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, feature_extractor):
    image = image.convert("RGB")
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()
        return predicted_class_idx, prediction_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def detect_eye_closure(image):
    # Convert to grayscale for facial landmarks detection
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV pre-trained face and eye detector models
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # If eyes are not detected, assume drowsy (eyes closed)
        if len(eyes) == 0:
            return True  # Eyes closed
    return False  # Eyes open

def display_result(image, predicted_class_idx, prediction_score):
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    if predicted_class_idx is not None:
        prediction_label = LABELS[predicted_class_idx]
        st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")
        st.session_state["predictions"].append({"Prediction": prediction_label, "Confidence Score": prediction_score})
    else:
        st.write("Error: Could not make a prediction.")

def sidebar():
    st.sidebar.title("Drowsiness Detection System")
    role = st.sidebar.radio("Select Role", ("User", "Admin"))
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if authenticate(username, password, role):
            st.session_state["authenticated"] = True
            st.session_state["role"] = role
            st.sidebar.success(f"Logged in as {role}")
        else:
            st.sidebar.error("Invalid credentials. Please try again.")

def main():
    st.title("Real-Time Drowsiness Detection")
    st.markdown("This application detects drowsiness using a deep learning model.")
    sidebar()
    
    if "authenticated" not in st.session_state:
        return
    
    role = st.session_state.get("role", "User")
    if role == "User":
        model, feature_extractor = load_model()
        if model is None or feature_extractor is None:
            st.error("Failed to load the model. Please check your internet connection or try again later.")
            return

        camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
        if camera_input is not None:
            with st.spinner("Processing..."):
                time.sleep(1)
                img = Image.open(camera_input)
                
                # Check for eye closure before model prediction
                if detect_eye_closure(img):
                    st.write("Eyes are closed! You may be drowsy.")
                    prediction_label = "Drowsy"
                    prediction_score = 0.85  # Assume high confidence for eye closure
                    display_result(img, 1, prediction_score)
                else:
                    inputs = preprocess_image(img, feature_extractor)
                    predicted_class_idx, prediction_score = get_prediction(model, inputs)
                    display_result(img, predicted_class_idx, prediction_score)
    else:
        st.title("Admin Dashboard")
        st.write("Below are the recorded predictions: ")
        if st.session_state["predictions"]:
            df = pd.DataFrame(st.session_state["predictions"])
            st.dataframe(df)
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
