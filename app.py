import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import pymongo
from bson import Binary
import io

# Access MongoDB URI from Streamlit secrets
MONGO_URI = st.secrets["mongo"]["uri"]

# MongoDB Connection
client = pymongo.MongoClient(MONGO_URI)
db = client.get_database()  # Get the default database
collection = db.predictions  # Replace with your desired collection name

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

# Save the result to MongoDB
def save_to_mongo(image, predicted_class_idx, prediction_score):
    """Save the image and prediction results to MongoDB."""
    # Convert image to binary for storage
    byte_io = io.BytesIO()
    image.save(byte_io, format="PNG")
    image_binary = byte_io.getvalue()

    # Document to insert
    document = {
        "image": Binary(image_binary),
        "prediction_label": LABELS[predicted_class_idx],
        "confidence": prediction_score,
        "timestamp": st.timestamp()  # Add a timestamp for the record
    }

    # Insert into MongoDB
    collection.insert_one(document)

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

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)
        inputs = preprocess_image(img, feature_extractor)

        # Get prediction
        predicted_class_idx, prediction_score = get_prediction(model, inputs)

        # Save the result to MongoDB
        save_to_mongo(img, predicted_class_idx, prediction_score)

        # Display the image and prediction result
        display_result(img, predicted_class_idx, prediction_score)

if __name__ == "__main__":
    main()
