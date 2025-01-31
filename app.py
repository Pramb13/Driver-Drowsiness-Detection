import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import time

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]

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
    else:
        st.write("Error: Could not make a prediction.")

def sidebar():
    st.sidebar.title("Drowsiness Detection System")
    role = st.sidebar.radio("Select Role", ("User", "Admin"))
    return role

def main():
    st.title("Real-Time Drowsiness Detection")
    st.markdown("This application detects drowsiness using a deep learning model.")
    
    role = sidebar()
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
                inputs = preprocess_image(img, feature_extractor)
                predicted_class_idx, prediction_score = get_prediction(model, inputs)
                display_result(img, predicted_class_idx, prediction_score)
    else:
        st.sidebar.write("Admin functionalities will be added here.")

if __name__ == "__main__":
    main()
