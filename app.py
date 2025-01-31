import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import numpy as np
import os
from pinecone import Pinecone as PineconeClient
import time

# Set Hugging Face API key and Pinecone API key from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["huggingface"]["api_key"]
os.environ['PINECONE_API_KEY'] = st.secrets["pinecone"]["api_key"]

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)
INDEX_NAME = st.secrets["pinecone"]["index_name"]  # Secure access to Pinecone index

# Initialize Pinecone client
class PineconeHandler:
    def __init__(self):
        self.index_name = "imageembedding"  # Update this if you want a different name for the image index
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))

        # Check if the Pinecone index exists, otherwise create one
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,  # Set the same dimension as the model output (ensure this matches the model)
                metric='cosine',  # Metric is cosine for similarity-based search
            )
            st.write(f"Index '{self.index_name}' created.")
        else:
            st.write(f"Index '{self.index_name}' already exists.")
        
        self.index = self.pc.Index(self.index_name)
    
    def store_embeddings(self, embeddings, metadata, retries=3, delay=2):
        """Store embeddings in Pinecone."""
        upsert_data = []
        for i, embedding in enumerate(embeddings):
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # Generate unique ID for each image embedding
            id = f"image-{i}"
            metadata_dict = metadata[i] if isinstance(metadata[i], dict) else {}

            # Prepare the data for upsert
            upsert_data.append((id, embedding, metadata_dict))

        # Attempt to upsert embeddings into Pinecone
        for attempt in range(retries):
            try:
                response = self.index.upsert(vectors=upsert_data)
                st.write(f"Upsert response: {response}")

                if response.get("upserted", 0) > 0:
                    st.write(f"Successfully upserted {response['upserted']} vectors.")
                else:
                    st.error("No vectors were upserted.")
                break
            except Exception as e:
                st.error(f"Error during Pinecone upsert: {str(e)}")
                if attempt < retries - 1:
                    st.write(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    st.error("Max retries reached. Please check the service status.")


# Load Hugging Face model and feature extractor
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
        outputs = model(**inputs)
        feature_vector = outputs.logits  # Raw features from the model (logits)
        feature_vector = feature_vector.squeeze().cpu().numpy()  # Ensure it's a 1D numpy array
    return feature_vector  # Return the numpy array


def main():
    """Main function to handle Streamlit interface and prediction process."""
    # Initialize Pinecone handler
    pinecone_handler = PineconeHandler()

    # Load the model and feature extractor
    model, feature_extractor = load_model()

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)

        # Extract features from the image
        image_embedding = extract_image_features(img)

        # Prediction (Optional: Here, you can implement prediction logic)
        # For example, classify the image and get the class label (drowsy or not)
        inputs = feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()  # Highest score value

        # Prepare metadata for Pinecone storage
        metadata = [{"class": LABELS[predicted_class_idx], "score": prediction_score}]
        
        # Store the image embedding and metadata in Pinecone
        pinecone_handler.store_embeddings([image_embedding], metadata)

        # Display the image and prediction result
        st.image(img, caption="Captured Image from Webcam", use_container_width=True)
        prediction_label = LABELS[predicted_class_idx]
        st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")

if __name__ == "__main__":
    main()
