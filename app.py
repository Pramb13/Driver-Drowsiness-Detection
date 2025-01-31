import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from pinecone import Pinecone as PineconeClient
import numpy as np
import os

# Set Hugging Face API key and Pinecone API key from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["huggingface"]["api_key"]
os.environ['PINECONE_API_KEY'] = st.secrets["pinecone"]["api_key"]

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Fetch Pinecone API key, index name, and environment securely from Streamlit secrets
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = st.secrets["pinecone"]["index_name"]  # Secure access to the Pinecone index name
pinecone_environment = st.secrets["pinecone"]["environment"]

# Pinecone Setup for text embedding
def setup_pinecone_index():
    """Initialize Pinecone index for text embeddings."""
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index_name = "textembedding"
    
    # Check if the index exists, if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
        )
        st.write(f"Index '{index_name}' created.")
    else:
        st.write(f"Index '{index_name}' already exists.")

    return pc.Index(index_name)

# Load model and feature extractor for image classification
def load_model():
    """Load pre-trained model and feature extractor."""
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor

# Store data in Pinecone (for image embeddings)
def store_in_pinecone(index, image, predicted_class_idx, prediction_score):
    """Store image prediction data in Pinecone."""
    feature_vector = extract_image_features(image)  # Extract image features

    # Prepare metadata and generate unique vector ID
    metadata = {
        "class": LABELS[predicted_class_idx],
        "score": prediction_score,
    }
    vector_id = str(np.random.randint(0, 1000000))  # Generate a random ID

    # Create the vector with the ID, feature vector, and metadata
    vector = {
        "id": vector_id,
        "values": feature_vector.tolist(),  # Ensure it's a list for Pinecone
        "metadata": metadata
    }

    # Upsert the vector into the Pinecone index
    upsert_response = index.upsert(
        vectors=[vector],
        namespace="ns1"  # Using "ns1" as the namespace
    )
    st.write(f"Upserted data with ID: {vector_id}")
    return upsert_response

# Extract image features for embedding
def extract_image_features(image):
    """Extract features from the image using the model's feature extractor."""
    model, feature_extractor = load_model()
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        feature_vector = outputs.logits  # Raw features from the model (logits)
        feature_vector = feature_vector.squeeze().cpu().numpy()  # Ensure it's a 1D numpy array
    return feature_vector  # Return the numpy array

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
    # Setup Pinecone index for text embedding
    text_index = setup_pinecone_index()

    # Load model and feature extractor for image classification
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
        store_in_pinecone(text_index, img, predicted_class_idx, prediction_score)

        # Display the image and prediction result
        display_result(img, predicted_class_idx, prediction_score)

if __name__ == "__main__":
    main()
