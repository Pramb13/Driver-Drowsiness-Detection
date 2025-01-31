import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from datetime import datetime

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Initialize the model and feature extractor
def load_model():
    """Load pre-trained model and feature extractor."""
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor

# Preprocess image for the model
def preprocess_image(image, feature_extractor):
    """Preprocess the image for model prediction."""
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Make prediction using the model
def get_prediction(model, inputs):
    """Make a prediction using the model and return the predicted class and confidence."""
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Get the raw prediction scores
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        prediction_score = logits.max().item()  # Highest score value
    return predicted_class_idx, prediction_score

# Get the current time in a human-readable format
def get_current_time():
    """Return the current time in a readable format."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time

# Display the image and prediction result
def display_result(image, predicted_class_idx, prediction_score, current_time):
    """Display the image along with the prediction result and time."""
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    prediction_label = LABELS[predicted_class_idx]
    st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")
    st.write(f"Prediction made at: {current_time}")

# Main Streamlit interface
def main():
    """Main function to handle Streamlit interface and prediction process."""
    # Load model and feature extractor
    model, feature_extractor = load_model()

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)
        inputs = preprocess_image(img, feature_extractor)

        # Get prediction
        predicted_class_idx, prediction_score = get_prediction(model, inputs)

        # Get current time
        current_time = get_current_time()

        # Display the image, prediction result, and time
        display_result(img, predicted_class_idx, prediction_score, current_time)

if __name__ == "__main__":
    main()
