import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from pinecone import pinecone as PineconeClient
import numpy as np
import os
import time

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

# Initialize Pinecone Client
class DrowsinessDetection:
    def __init__(self):
        try:
            # Initialize Pinecone connection
            self.index_name = INDEX_NAME  # Index name from Streamlit secrets
            self.pc = PineconeClient.Client(api_key=PINECONE_API_KEY, environment=pinecone_environment)

            # Check if the index exists, create if it doesn't
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # The dimension of the embeddings (match this with model output)
                    metric='cosine',
                )
                st.write(f"Index '{self.index_name}' created.")
            else:
                st.write(f"Index '{self.index_name}' already exists.")
            
            # Assign the Pinecone index
            self.index = self.pc.Index(self.index_name)
            st.write("Pinecone index is set.")
        
        except Exception as e:
            st.error(f"Error initializing Pinecone: {e}")
            raise

    def load_model(self):
        """Load pre-trained model and feature extractor."""
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor

    def extract_image_features(self, image):
        """Extract features from the image using the model's feature extractor."""
        model, feature_extractor = self.load_model()
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            feature_vector = outputs.logits  # Raw features from the model (logits)
            feature_vector = feature_vector.squeeze().cpu().numpy()  # Ensure it's a 1D numpy array
        return feature_vector

    def store_in_pinecone(self, image, predicted_class_idx, prediction_score):
        """Store image prediction data in Pinecone."""
        if not hasattr(self, 'index'):
            st.error("Pinecone index is not initialized properly.")
            return
        
        feature_vector = self.extract_image_features(image)
        
        # Ensure the feature vector dimension matches the index's dimension (1024)
        if feature_vector.shape[0] != 1024:
            st.error(f"Embedding dimension mismatch: Expected 1024, got {feature_vector.shape[0]}")
            return

        # Prepare metadata
        metadata = {
            "class": LABELS[predicted_class_idx],
            "score": prediction_score,
        }

        # Generate unique vector ID
        vector_id = str(np.random.randint(0, 1000000))  # Generate a random ID

        # Create the vector with the ID, feature vector, and metadata
        vector = {
            "id": vector_id,
            "values": feature_vector.tolist(),  # Ensure it's a list for Pinecone
            "metadata": metadata
        }

        try:
            # Upsert the vector into the Pinecone index
            upsert_response = self.index.upsert(
                vectors=[vector],
                namespace="ns1"  # Using "ns1" as the namespace
            )
            st.write(f"Upserted data with ID: {vector_id}")
            st.write(f"Upsert response: {upsert_response}")
        except Exception as e:
            st.error(f"Error during Pinecone upsert: {e}")

    def preprocess_image(self, image, feature_extractor):
        """Preprocess the image for model prediction."""
        image = image.convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs

    def get_prediction(self, model, inputs):
        """Make a prediction using the model and return the predicted class and confidence."""
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Get the raw prediction scores
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()  # Highest score value
        return predicted_class_idx, prediction_score

    def display_result(self, image, predicted_class_idx, prediction_score):
        """Display the image along with the prediction result."""
        st.image(image, caption="Captured Image from Webcam", use_container_width=True)
        prediction_label = LABELS[predicted_class_idx]
        st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")

# Main Streamlit interface
def main():
    """Main function to handle Streamlit interface and prediction process."""
    # Initialize DrowsinessDetection object
    drowsiness_detector = DrowsinessDetection()

    # Load model and feature extractor
    model, feature_extractor = drowsiness_detector.load_model()

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)
        inputs = drowsiness_detector.preprocess_image(img, feature_extractor)

        # Get prediction
        predicted_class_idx, prediction_score = drowsiness_detector.get_prediction(model, inputs)

        # Store the result in Pinecone
        drowsiness_detector.store_in_pinecone(img, predicted_class_idx, prediction_score)

        # Display the image and prediction result
        drowsiness_detector.display_result(img, predicted_class_idx, prediction_score)

if __name__ == "__main__":
    main()
