import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import numpy as np
import os
import pinecone
import time

# Set environment variables for Hugging Face and Pinecone API keys
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["huggingface"]["api_key"]
os.environ['PINECONE_API_KEY'] = st.secrets["pinecone"]["api_key"]

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = st.secrets["pinecone"]["index_name"]  # Pinecone index name
pinecone_environment = st.secrets["pinecone"]["environment"]


index = pinecone.Index(INDEX_NAME)

# Load model and feature extractor
def load_model():
    """Load pre-trained model and feature extractor."""
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor

def extract_image_features(image):
    """Extract features from the image using the model's feature extractor."""
    model, feature_extractor = load_model()
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # Get the output from the model
        outputs = model(**inputs)
        
        # The embeddings are typically in `outputs.logits`
        feature_vector = outputs.logits  # Extract the logits (features)
        
        # Ensure the feature vector has the correct dimension, e.g., 1024 for 'dino-vits16'
        feature_vector = feature_vector.squeeze().cpu().numpy()  # Remove extra dimensions and convert to numpy
        
        # Check if the dimension is correct (1024)
        if feature_vector.shape[0] != 1024:
            st.error(f"Embedding dimension mismatch: Expected 1024, got {feature_vector.shape[0]}")
            return None
    return feature_vector  # Return the numpy array

# Store image prediction data in Pinecone
def store_in_pinecone(image, predicted_class_idx, prediction_score):
    """Store image prediction data in Pinecone."""
    feature_vector = extract_image_features(image)  # Extract image features
    
    if feature_vector is None:
        st.error("Failed to extract valid feature vector.")
        return
    
    # Prepare metadata
    metadata = {
        "class": LABELS[predicted_class_idx],
        "score": prediction_score,
    }
    
    # Generate unique vector ID (you can use a unique identifier, e.g., time-based or random)
    vector_id = str(np.random.randint(0, 1000000))  # Generate a random ID

    # Prepare the vector for upsert
    vector = {
        "id": vector_id,
        "values": feature_vector.tolist(),  # Convert numpy array to list
        "metadata": metadata
    }

    # Upsert the vector into Pinecone
    try:
        upsert_response = index.upsert(
            vectors=[vector],
            namespace="ns1"  # Using "ns1" as the namespace
        )
        st.write(f"Upserted data with ID: {vector_id}")
        st.write(upsert_response)
    except Exception as e:
        st.error(f"Error during Pinecone upsert: {str(e)}")

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

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)
        inputs = preprocess_image(img, feature_extractor)

        # Get prediction
        predicted_class_idx, prediction_score = get_prediction(model, inputs)

        # Store the result in Pinecone
        store_in_pinecone(img, predicted_class_idx, prediction_score)

        # Display the image and prediction result
        display_result(img, predicted_class_idx, prediction_score)

if __name__ == "__main__":
    main()
