import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import numpy as np
import os

# Set Hugging Face API key from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["huggingface"]["api_key"]

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Predefined credentials for user and admin
# Predefined usernames and passwords (for demonstration purposes)
USER_CREDENTIALS = {"user": "user_password"}
ADMIN_CREDENTIALS = {"admin": "admin_password"}

# Function for User login
def login(username, password):
    # Check credentials for user and admin
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return "user"
    elif username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        return "admin"
    else:
        return None

# Drowsiness detection class
class DrowsinessDetection:
    def __init__(self):
        """Initialize the model and the feature extractor"""
        self.model, self.feature_extractor = self.load_model()

    def load_model(self):
        """Load pre-trained model and feature extractor."""
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor

    def extract_image_features(self, image):
        """Extract features from the image using the model's feature extractor."""
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            feature_vector = outputs.logits  # Raw features from the model (logits)
            feature_vector = feature_vector.squeeze().cpu().numpy()  # Ensure it's a 1D numpy array
        return feature_vector  # Return the numpy array

    def get_prediction(self, inputs):
        """Make a prediction using the model and return the predicted class and confidence."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Get the raw prediction scores
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()  # Highest score value
        return predicted_class_idx, prediction_score

# Preprocess image for the model
def preprocess_image(image, feature_extractor):
    """Preprocess the image for model prediction."""
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Display the image and prediction result
def display_result(image, predicted_class_idx, prediction_score):
    """Display the image along with the prediction result."""
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    prediction_label = LABELS[predicted_class_idx]
    st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")

# Main Streamlit interface
def main():
    """Main function to handle Streamlit interface and prediction process."""
    
    # Login screen for user and admin
    st.sidebar.title("Login")
    login_type = st.sidebar.selectbox("Login as", ["Select role", "User", "Admin"])
    
    if login_type != "Select role":
        if login_type == "User":
            # User login form
            st.sidebar.text_input("Username", key="user_username")
            st.sidebar.text_input("Password", type="password", key="user_password")
            if st.sidebar.button("Login as User"):
                username = st.session_state.user_username
                password = st.session_state.user_password
                if login(username, password) == "user":
                    st.sidebar.success("Logged in as User")
                    # User can see webcam input and drowsiness prediction
                    capture_and_predict()
                else:
                    st.sidebar.error("Invalid credentials, please try again.")
                    
        elif login_type == "Admin":
            # Admin login form
            st.sidebar.text_input("Username", key="admin_username")
            st.sidebar.text_input("Password", type="password", key="admin_password")
            if st.sidebar.button("Login as Admin"):
                username = st.session_state.admin_username
                password = st.session_state.admin_password
                if login(username, password) == "admin":
                    st.sidebar.success("Logged in as Admin")
                    # Admin features can be added here
                    st.write("Welcome Admin!")
                    # You could allow Admin to access logs or perform actions here
                else:
                    st.sidebar.error("Invalid credentials, please try again.")
    else:
        st.write("Please select a role to log in.")

# Function to capture and predict drowsiness using webcam
def capture_and_predict():
    """Capture image from webcam and make a drowsiness prediction."""
    # Initialize DrowsinessDetection
    detector = DrowsinessDetection()

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)
        inputs = preprocess_image(img, detector.feature_extractor)

        # Get prediction
        predicted_class_idx, prediction_score = detector.get_prediction(inputs)

        # Display the image and prediction result
        display_result(img, predicted_class_idx, prediction_score)

if __name__ == "__main__":
    main()
