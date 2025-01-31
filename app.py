import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import numpy as np
import seaborn as sns
import pandas as pd
import os

# Predefined user credentials
USER_CREDENTIALS = {"user": "user_password"}
ADMIN_CREDENTIALS = {"admin": "admin_password"}

# Placeholder for drowsiness data for demo purposes
# In a production setting, you can store this data in a database
drowsiness_data = []

# Set Hugging Face API key
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["huggingface"]["api_key"]

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Login function
def login(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return "user"
    elif username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        return "admin"
    else:
        return None

# DrowsinessDetection class
class DrowsinessDetection:
    def __init__(self):
        """Initialize the model and feature extractor"""
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
            logits = outputs.logits  # Get the raw prediction scores
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()  # Highest score value
        return predicted_class_idx, prediction_score

# Seaborn visualization for Admin
def plot_drowsiness_data():
    """Generate a seaborn plot based on the drowsiness data collected."""
    if drowsiness_data:
        # Convert data to a DataFrame
        df = pd.DataFrame(drowsiness_data)
        
        # Plotting the data
        st.subheader("Drowsiness Prediction Data")
        fig = sns.countplot(x='prediction', data=df, palette='viridis')
        fig.set_title('Drowsiness Predictions Over Time')
        st.pyplot(fig)
    else:
        st.write("No data available to display.")

# Sidebar Login
def login_sidebar():
    """Login interface for users to enter their credentials."""
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        user_type = login(username, password)
        return user_type
    return None

# Main function to handle different user roles
def main():
    """Main function to handle the interface and prediction process."""
    st.title("Drowsiness Detection System")
    
    # Login sidebar for authentication
    user_type = login_sidebar()
    
    if user_type == "user":
        # User's interface
        st.subheader("Welcome User! Capture an image to predict drowsiness.")
        drowsiness_detector = DrowsinessDetection()
        
        # Capture image from webcam
        camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
        
        if camera_input is not None:
            # Load and process the captured image
            img = Image.open(camera_input)
            predicted_class_idx, prediction_score = drowsiness_detector.get_prediction(img)
            
            # Store prediction data (for demonstration, we'll append it)
            result = LABELS[predicted_class_idx]
            st.image(img, caption="Captured Image", use_container_width=True)
            st.write(f"Prediction: {result} with confidence {prediction_score:.2f}")
            
            # No record stored for the user
            # You could optionally store user-specific data here for further analysis.
            
    elif user_type == "admin":
        # Admin's interface
        st.subheader("Welcome Admin! View historical prediction data.")
        plot_drowsiness_data()
        
    elif user_type is None:
        st.write("Please enter valid credentials.")
    
if __name__ == "__main__":
    main()
