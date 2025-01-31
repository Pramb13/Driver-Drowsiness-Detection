import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from datetime import datetime
import os
import pandas as pd
import pygame  # Use pygame for audio playback

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)
HISTORY_FILE = "drowsiness_history.csv"  # File to track drowsiness history
ALERT_SOUND_PATH = "audio_alert.wav"  # Path to your alert sound file (replace with actual file)

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

# Play an alert sound when drowsiness is detected using pygame
def play_alert():
    """Play an alert sound when drowsiness is detected."""
    if os.path.exists(ALERT_SOUND_PATH):
        try:
            pygame.mixer.init()  # Initialize pygame mixer for sound playback
            pygame.mixer.music.load(ALERT_SOUND_PATH)  # Load the audio file
            pygame.mixer.music.play()  # Play the sound
            print("Alert sound is playing...")
        except Exception as e:
            print(f"Error while playing the sound: {e}")
    else:
        print("Alert sound file not found!")

# Save drowsiness event to history file
def save_to_history(prediction, confidence, current_time):
    """Save drowsiness prediction to history CSV file."""
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=["Time", "Prediction", "Confidence"])
        df.to_csv(HISTORY_FILE, index=False)
    
    new_data = pd.DataFrame([[current_time, prediction, confidence]], columns=["Time", "Prediction", "Confidence"])
    new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

# Display the image and prediction result
def display_result(image, predicted_class_idx, prediction_score, current_time):
    """Display the image along with the prediction result and time."""
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    prediction_label = LABELS[predicted_class_idx]
    st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")
    st.write(f"Prediction made at: {current_time}")

# Display history of drowsiness events
def display_history():
    """Display history of drowsiness events."""
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.write("Drowsiness History:")
        st.dataframe(df)

# Main Streamlit interface
def main():
    """Main function to handle Streamlit interface and prediction process."""
    # Load model and feature extractor
    model, feature_extractor = load_model()

    # Display title and description
    st.title("Real-Time Drowsiness Detection")
    st.markdown("""
        This app uses your webcam feed to detect signs of drowsiness. If drowsiness is detected, an alert will be triggered.
        The history of your drowsiness events will be tracked.
    """)

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

        # Display the result
        display_result(img, predicted_class_idx, prediction_score, current_time)

        # If drowsy, play sound alert and save to history
        if predicted_class_idx == 1:
            play_alert()

        # Save the event to the history file
        save_to_history(LABELS[predicted_class_idx], prediction_score, current_time)

    # Display drowsiness history
    if st.button("Show Drowsiness History"):
        display_history()

if __name__ == "__main__":
    main()
