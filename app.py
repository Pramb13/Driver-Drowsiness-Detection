import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import numpy as np
import time
import os

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Predefined user credentials for simplicity
USER_CREDENTIALS = {"user": "123"}
ADMIN_CREDENTIALS = {"admin": "123"}

# DrowsinessDetection class
class DrowsinessDetection:
    def __init__(self):
        """Initialize the model and feature extractor."""
        self.model, self.feature_extractor = self.load_model()

    def load_model(self):
        """Load pre-trained model and feature extractor."""
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor

    def get_prediction(self, image):
        """Get prediction from the model (drowsy or not drowsy)."""
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Raw prediction scores
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()  # Highest score value
        return predicted_class_idx, prediction_score

# Function to display prediction for the user
def display_user_result(predicted_class_idx, prediction_score):
    """Display the prediction result for the user."""
    prediction_label = LABELS[predicted_class_idx]
    st.write(f"**Prediction**: {prediction_label}")
    st.write(f"**Confidence Score**: {prediction_score:.2f}")

# Sidebar for Login & Role Selection
def sidebar():
    """Sidebar with role-based login system."""
    st.sidebar.title("Drowsiness Detection System")
    role = st.sidebar.radio("Select Role", ("User", "Admin"))
    
    if role == "User":
        st.sidebar.subheader("Login as User")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.sidebar.success("Login successful! You are logged in as User.")
                return "user"
            else:
                st.sidebar.error("Invalid credentials. Please try again.")
                return None

    elif role == "Admin":
        st.sidebar.subheader("Login as Admin")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
                st.sidebar.success("Login successful! You are logged in as Admin.")
                return "admin"
            else:
                st.sidebar.error("Invalid credentials. Please try again.")
                return None

# Main Streamlit interface
def main():
    """Main function to handle Streamlit interface and prediction process."""
    role = sidebar()
    
    if role == "user":
        # User: Real-time webcam feed for drowsiness detection
        st.title("Real-time Drowsiness Detection")
        st.markdown("<h3 style='text-align: center; color: #1E88E5;'>Stay Safe, Stay Alert!</h3>", unsafe_allow_html=True)
        st.markdown("### Capturing your face to detect drowsiness.")
        drowsiness_detector = DrowsinessDetection()

        # Check if there's a previous snapshot stored in session state
        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None

        # Capture image from webcam
        camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
        
        if camera_input is not None:
            # Load and preprocess image
            img = Image.open(camera_input)
            predicted_class_idx, prediction_score = drowsiness_detector.get_prediction(img)

            # Store prediction result in session state
            st.session_state.prediction_result = (predicted_class_idx, prediction_score)

            # Display result
            st.image(img, caption="Captured Image from Webcam", use_container_width=True)
            display_user_result(predicted_class_idx, prediction_score)

            # Snapshot button
            if st.button("Snapshot"):
                st.write("Snapshot Captured")
                # After clicking "Snapshot", the prediction and result will be shown automatically as above

        elif st.session_state.prediction_result is not None:
            # If there is a prediction result stored in session state, display it
            predicted_class_idx, prediction_score = st.session_state.prediction_result
            display_user_result(predicted_class_idx, prediction_score)

    elif role == "admin":
        # Admin: Display admin controls
        st.title("Admin Dashboard")
        st.markdown("<h3 style='text-align: center; color: #E53935;'>Monitor Drowsiness Patterns</h3>", unsafe_allow_html=True)
        # Additional admin functionalities here (graph display, data export, etc.)

if __name__ == "__main__":
    main()
