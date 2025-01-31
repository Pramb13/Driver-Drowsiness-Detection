import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import numpy as np
import seaborn as sns
import pandas as pd
import os
import time

# Predefined user credentials for simplicity
USER_CREDENTIALS = {"user": "123"}
ADMIN_CREDENTIALS = {"admin": "123"}

# Placeholder for drowsiness data (this would be stored in a database in real life)
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

    def store_in_drowsiness_data(self, predicted_class_idx, prediction_score):
        """Store the drowsiness detection result for admin view."""
        metadata = {
            "class": LABELS[predicted_class_idx],
            "score": prediction_score,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        drowsiness_data.append(metadata)
        self.dump_data_to_csv(metadata)  # Save data to CSV

    def dump_data_to_csv(self, metadata):
        """Append data to a CSV file for future use."""
        df = pd.DataFrame([metadata])
        file_path = "drowsiness_data.csv"

        # If the file does not exist, create it with headers
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False, header=True)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)

# Function to display graphs for Admin
def display_graph():
    """Display prediction statistics using Seaborn."""
    if len(drowsiness_data) == 0:
        st.write("No records available for display.")
        return

    # Create a DataFrame from the collected data
    df = pd.DataFrame(drowsiness_data)
    
    # Seaborn bar plot
    st.subheader("Prediction Statistics")
    prediction_counts = df['class'].value_counts()
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette="Blues_d")
    ax.set(xlabel="Prediction", ylabel="Count")
    st.pyplot(ax.figure)

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
            user_role = login(username, password)
            if user_role == "user":
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
            admin_role = login(username, password)
            if admin_role == "admin":
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

        # Capture image from webcam
        camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
        
        if camera_input is not None:
            # Load and preprocess image
            img = Image.open(camera_input)
            predicted_class_idx, prediction_score = drowsiness_detector.get_prediction(img)
            
            # Store result in drowsiness data (only for admin view)
            drowsiness_detector.store_in_drowsiness_data(predicted_class_idx, prediction_score)
            
            # Display result
            st.image(img, caption="Captured Image from Webcam", use_container_width=True)
            prediction_label = LABELS[predicted_class_idx]
            st.write(f"**Prediction**: {prediction_label}")
            st.write(f"**Confidence Score**: {prediction_score:.2f}")
            
            # Add snapshot button to capture and process image
            if st.button("Snapshot"):
                # Saving the snapshot to session state
                st.session_state.img = img
                st.session_state.predicted_class_idx = predicted_class_idx
                st.session_state.prediction_score = prediction_score

                # Display snapshot after capturing
                st.image(st.session_state.img, caption="Snapshot", use_container_width=True)
                prediction_label = LABELS[st.session_state.predicted_class_idx]
                st.write(f"**Prediction**: {prediction_label}")
                st.write(f"**Confidence Score**: {st.session_state.prediction_score:.2f}")
    
    elif role == "admin":
        # Admin: Display statistics of drowsiness predictions
        st.title("Admin Dashboard")
        st.markdown("<h3 style='text-align: center; color: #E53935;'>Monitor Drowsiness Patterns</h3>", unsafe_allow_html=True)
        display_graph()
        
        st.markdown("### Download the Drowsiness Report")
        if len(drowsiness_data) > 0:
            df = pd.DataFrame(drowsiness_data)
            st.write("### Drowsiness Data (Last 10 records)")
            st.dataframe(df.tail(10))  # Display last 10 records
            st.download_button(
                label="Download as CSV",
                data=df.to_csv(index=False),
                file_name="drowsiness_data.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
