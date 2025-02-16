import streamlit as st
import torch
import pandas as pd
import av
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
USER_CREDENTIALS = {"user": "123"}
ADMIN_CREDENTIALS = {"admin": "admin123"}

# Store session predictions
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

@st.cache_resource
def load_model():
    """ Load the image classification model and feature extractor """
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, feature_extractor):
    """ Convert and preprocess image for model input """
    image = image.convert("RGB")
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    """ Get model prediction and confidence score """
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            prediction_score = probabilities[0, predicted_class_idx].item()

        return predicted_class_idx, prediction_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

class VideoProcessor(VideoProcessorBase):
    """ Video processing class for real-time drowsiness detection """
    def __init__(self):
        self.model, self.feature_extractor = load_model()

    def recv(self, frame):
        """ Process each frame and make predictions """
        try:
            img = frame.to_image()
            inputs = preprocess_image(img, self.feature_extractor)
            predicted_class_idx, prediction_score = get_prediction(self.model, inputs)

            if predicted_class_idx is not None:
                prediction_label = LABELS[predicted_class_idx]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Store predictions
                st.session_state["predictions"].append({
                    "Prediction": prediction_label,
                    "Confidence Score": f"{prediction_score:.2f}",
                    "Timestamp": timestamp
                })

                return av.VideoFrame.from_image(img)

        except Exception as e:
            st.error(f"Error processing frame: {e}")
            return frame  # Return the original frame if an error occurs

def sidebar():
    """ Sidebar authentication for users and admins """
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

def authenticate(username, password, role):
    """ Authentication function for users and admins """
    if role == "User" and username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True
    elif role == "Admin" and username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        return True
    return False

def main():
    """ Main app logic """
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

        # Live video streaming
        webrtc_streamer(key="drowsiness-detection", video_processor_factory=VideoProcessor)

    else:  # Admin Panel
        st.title("Admin Dashboard")
        st.write("Below are the recorded predictions with date and time:")
        
        if st.session_state["predictions"]:
            df = pd.DataFrame(st.session_state["predictions"])
            st.dataframe(df)
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
