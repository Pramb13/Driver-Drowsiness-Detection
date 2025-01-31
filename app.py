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
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = st.secrets["pinecone"]["index_name"]  # Pinecone index name
pinecone_environment = st.secrets["pinecone"]["environment"]

# Import Pinecone with alias
import pinecone

class DrowsinessDetection:
    def __init__(self):
        # Initialize Pinecone client (Pinecone SDK 2.x)
        pinecone.init(api_key=PINECONE_API_KEY, environment=pinecone_environment)

        # Ensure the index exists, create if necessary
        if INDEX_NAME not in pinecone.list_indexes():
            st.write(f"Index '{INDEX_NAME}' does not exist. Creating it...")
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=1024,  # Embedding dimension (for DINO-ViT, it's 1024)
                metric='cosine'  # Cosine similarity metric
            )
            st.write(f"Index '{INDEX_NAME}' created.")
        else:
            st.write(f"Index '{INDEX_NAME}' already exists.")

        # Set the Pinecone index
        self.index = pinecone.Index(INDEX_NAME)

        # Load the Hugging Face model and feature extractor
        self.model, self.feature_extractor = self.load_model()

    def load_model(self):
        """Load the pre-trained model and feature extractor."""
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor

    def extract_image_features(self, image):
        """Extract features from the image using the model's feature extractor."""
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            feature_vector = outputs.logits  # Extract the logits as the feature vector

            # Ensure the feature vector has the correct dimension
            feature_vector = feature_vector.squeeze().cpu().numpy()
            if feature_vector.shape[0] != 1024:
                st.error(f"Embedding dimension mismatch: Expected 1024, got {feature_vector.shape[0]}")
                return None
        return feature_vector

    def store_in_pinecone(self, image, predicted_class_idx, prediction_score):
        """Store image prediction data in Pinecone."""
        feature_vector = self.extract_image_features(image)
        
        if feature_vector is None:
            st.error("Failed to extract valid feature vector.")
            return

        # Prepare metadata
        metadata = {
            "class": LABELS[predicted_class_idx],
            "score": prediction_score,
        }

        # Generate a unique vector ID
        vector_id = str(np.random.randint(0, 1000000))  # Generate a random ID

        # Prepare the vector for upsert
        vector = {
            "id": vector_id,
            "values": feature_vector.tolist(),  # Convert numpy array to list
            "metadata": metadata
        }

        # Upsert the vector into Pinecone
        try:
            upsert_response = self.index.upsert(
                vectors=[vector],
                namespace="ns1"  # Namespace (you can adjust this based on your needs)
            )
            st.write(f"Upserted data with ID: {vector_id}")
            st.write(upsert_response)
        except Exception as e:
            st.error(f"Error during Pinecone upsert: {str(e)}")

    def preprocess_image(self, image):
        """Preprocess the image for model prediction."""
        image = image.convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs

    def get_prediction(self, inputs):
        """Make a prediction using the model and return the predicted class and confidence."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Get the raw prediction scores
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()  # Highest score value
        return predicted_class_idx, prediction_score

    def display_result(self, image, predicted_class_idx, prediction_score):
        """Display the image along with the prediction result."""
        st.image(image, caption="Captured Image from Webcam", use_container_width=True)
        prediction_label = LABELS[predicted_class_idx]
        st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")

def main():
    """Main function to handle Streamlit interface and prediction process."""
    # Instantiate the DrowsinessDetection class
    drowsiness_detector = DrowsinessDetection()

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input is not None:
        # Load and preprocess image
        img = Image.open(camera_input)
        inputs = drowsiness_detector.preprocess_image(img)

        # Get prediction
        predicted_class_idx, prediction_score = drowsiness_detector.get_prediction(inputs)

        # Store the result in Pinecone
        drowsiness_detector.store_in_pinecone(img, predicted_class_idx, prediction_score)

        # Display the image and prediction result
        drowsiness_detector.display_result(img, predicted_class_idx, prediction_score)

if __name__ == "__main__":
    main()
