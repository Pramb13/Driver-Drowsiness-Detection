import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import time

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

# Display the image and prediction result
def display_result(image, predicted_class_idx, prediction_score):
    """Display the image along with the prediction result."""
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    prediction_label = LABELS[predicted_class_idx]
    st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")

# Main Streamlit interface
def main():
    """Main function to handle Streamlit interface and prediction process."""
    # Load model and feature extractor
    model, feature_extractor = load_model()

    # Capture image from webcam (loop)
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    while True:
        if camera_input is not None:
            # Load and preprocess image
            img = Image.open(camera_input)
            inputs = preprocess_image(img, feature_extractor)

            # Get prediction
            predicted_class_idx, prediction_score = get_prediction(model, inputs)

            # Display the image and prediction result
            display_result(img, predicted_class_idx, prediction_score)

            # Pause between frames to control frame rate
            time.sleep(0.1)

if __name__ == "__main__":
    main()
