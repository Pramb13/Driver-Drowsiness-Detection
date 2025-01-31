import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import time
import pinecone
import numpy as np

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Pinecone Initialization
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")  # Replace with your Pinecone API key and environment
INDEX_NAME = "drowsiness-detection"

# Initialize Pinecone index
def create_pinecone_index():
    """Create Pinecone index if it doesn't exist."""
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            INDEX_NAME,
            dimension=768,  # This depends on your embeddings' dimension
            metric="cosine"
        )
    return pinecone.Index(INDEX_NAME)

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

# Store predictions in Pinecone
def store_in_pinecone(index, predicted_class_idx, prediction_score, image):
    """Store predictions in Pinecone."""
    # Convert image to a vector/embedding (or you can use the model output as a vector)
    # For simplicity, we'll use prediction score as the embedding (normally, you'd generate embeddings using a feature extractor)
    embedding = np.array([prediction_score])  # Convert score to numpy array (you can adjust this logic)
    
    # Create metadata (store more info as needed)
    metadata = {"prediction": LABELS[predicted_class_idx], "score": prediction_score}
    
    # Generate unique ID for this prediction (e.g., based on timestamp)
    id = f"prediction-{int(time.time())}"
    
    # Upsert to Pinecone
    upsert_response = index.upsert(vectors=[(id, embedding.tolist(), metadata)])
    return upsert_response

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

    # Create Pinecone index
    index = create_pinecone_index()

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)
        inputs = preprocess_image(img, feature_extractor)

        # Get prediction
        predicted_class_idx, prediction_score = get_prediction(model, inputs)

        # Display the image and prediction result
        display_result(img, predicted_class_idx, prediction_score)

        # Store prediction result in Pinecone
        upsert_response = store_in_pinecone(index, predicted_class_idx, prediction_score, img)
        st.write(f"Pinecone upsert response: {upsert_response}")

if __name__ == "__main__":
    main()
