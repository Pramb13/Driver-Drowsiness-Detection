import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import pinecone
import numpy as np
from sklearn.preprocessing import normalize

# Access secrets securely from Streamlit secrets
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENVIRONMENT = st.secrets["pinecone"]["environment"]
INDEX_NAME = st.secrets["pinecone"]["index_name"]


# Function to create the Pinecone index
def create_pinecone_index():
    """Check if the index exists and create it if it doesn't."""
    # Replace deprecated list_indexes() with indexes()
    if INDEX_NAME not in pinecone.indexes():  # Corrected line
        print(f"Index '{INDEX_NAME}' not found. Creating index...")
        # Dimension must match the size of the image embeddings (e.g., 512 for many transformer models)
        pinecone.create_index(INDEX_NAME, dimension=1024)  # Adjust dimension based on your model's output size
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    # Return the index connection
    return pinecone.Index(INDEX_NAME)

# Initialize the HuggingFace model
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Load the pre-trained model and feature extractor
def load_model():
    """Load pre-trained model and feature extractor."""
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor

# Preprocess the image and extract embeddings
def preprocess_image(image, feature_extractor):
    """Preprocess the image for model prediction."""
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Get the image embeddings
def get_image_embedding(model, inputs):
    """Generate image embedding using the model."""
    with torch.no_grad():
        outputs = model.get_input_embeddings()(inputs['pixel_values'])
        embeddings = outputs.mean(dim=1)  # Get the average of the embeddings
    return embeddings.squeeze().cpu().numpy()

# Add image embedding to Pinecone
def add_embedding_to_pinecone(embedding, image_id):
    """Normalize the embedding and add it to Pinecone."""
    embedding = normalize([embedding])[0]  # Normalize the embedding for Pinecone
    index.upsert(vectors=[(image_id, embedding)])

# Search Pinecone for similar image embeddings
def search_similar_embeddings(query_embedding, top_k=3):
    """Normalize the query embedding and search for similar vectors."""
    query_embedding = normalize([query_embedding])[0]  # Normalize query vector
    result = index.query([query_embedding], top_k=top_k)
    return result

# Display the image and prediction result
def display_result(image, predicted_class_idx, prediction_score):
    """Display the image along with the prediction result."""
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    prediction_label = LABELS[predicted_class_idx]
    st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")

# Main Streamlit interface
def main():
    """Main function to handle Streamlit interface and prediction process."""
    # Create Pinecone index (or use an existing one)
    global index
    index = create_pinecone_index()

    # Load model and feature extractor
    model, feature_extractor = load_model()

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)
        inputs = preprocess_image(img, feature_extractor)

        # Get image embedding
        embedding = get_image_embedding(model, inputs)

        # Add image embedding to Pinecone
        image_id = "unique_image_id"  # Generate a unique image ID (could use timestamp or other unique identifiers)
        add_embedding_to_pinecone(embedding, image_id)

        # Get prediction (optional)
        predicted_class_idx, prediction_score = get_prediction(model, inputs)

        # Display the image and prediction result
        display_result(img, predicted_class_idx, prediction_score)

        # Optionally search for similar images in Pinecone (for demonstration)
        similar_results = search_similar_embeddings(embedding)
        st.write("Similar Images Found in Pinecone:")
        for result in similar_results['matches']:
            st.write(f"ID: {result['id']}, Score: {result['score']}")

if __name__ == "__main__":
    main()
