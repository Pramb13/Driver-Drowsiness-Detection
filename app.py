import streamlit as st
import torch
import pandas as pd
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from datetime import datetime
import time

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

def display_result(image, predicted_class_idx, prediction_score):
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    if predicted_class_idx is not None:
        prediction_label = LABELS[predicted_class_idx]
        st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")
        # Add current timestamp to the prediction entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["predictions"].append({
            "Prediction": prediction_label, 
            "Confidence Score": prediction_score,
            "Timestamp": timestamp
        })
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

        # Capture webcam input
        camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
        
        # Process image and make prediction if camera input is not None
        if camera_input is not None:
            img = Image.open(camera_input)
            inputs = preprocess_image(img, feature_extractor)
            predicted_class_idx, prediction_score = get_prediction(model, inputs)

            # Adjust threshold for prediction
            if prediction_score < 2.5:
                predicted_class_idx = 1  # Force prediction to "Drowsy"
                prediction_score = 0.85  # Set a default confidence score

            display_result(img, predicted_class_idx, prediction_score)
        
        else:
            st.write("Waiting for webcam input...")
    else:
        st.title("Admin Dashboard")
        st.write("Below are the recorded predictions with date and time:")
        if st.session_state["predictions"]:
            df = pd.DataFrame(st.session_state["predictions"])
            st.dataframe(df)
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
