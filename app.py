import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import pinecone
import numpy as np

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Fetch Pinecone API key and index name securely from Streamlit secrets
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]  # Secure access to the Pinecone API key
INDEX_NAME = st.secrets["pinecone"]["index_name"]  # Secure access to the Pinecone index name
index = INDEX_NAME

# Initialize the model and feature extractor
def load_model():
    """Load pre-trained model and feature extractor."""
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor

# Store data in Pinecone
def store_in_pinecone(index, image, predicted_class_idx, prediction_score):
    """Store image prediction data in Pinecone."""
    # Convert image to a vector (using feature extractor or model)
    feature_vector = extract_image_features(image)  # Extract image features from model

    # Prepare the metadata for the image
    metadata = {
        "class": LABELS[predicted_class_idx],
        "score": prediction_score,
    }

    # Generate a unique ID for the image
    vector_id = str(np.random.randint(0, 1000000))

    # Upsert the vector and metadata into Pinecone
    index.upsert([(vector_id, feature_vector.tolist(), metadata)])

def extract_image_features(image):
    """Extract features from the image using the model's feature extractor."""
    model, feature_extractor = load_model()
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        feature_vector = outputs.logits  # Raw features from the model
    return feature_vector.squeeze().numpy()  # Convert to numpy array

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
        store_in_pinecone(index, img, predicted_class_idx, prediction_score)

        # Display the image and prediction result
        display_result(img, predicted_class_idx, prediction_score)

if __name__ == "__main__":
    main()
