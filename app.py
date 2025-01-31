import streamlit as st
import cv2
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from pinecone import Pinecone, ServerlessSpec
import torch
from PIL import Image
import os
import time

# Set environment variables for Pinecone and Hugging Face
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["huggingface"]["api_key"]
os.environ['PINECONE_API_KEY'] = st.secrets["pinecone"]["api_key"]

# Constants
MODEL_NAME = "facebook/dino-vits16"  # You can use another model suited for drowsiness detection
LABELS = ["Not Drowsy", "Drowsy"]  # Drowsiness detection labels

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "drowsiness_detection"  # You can change this name
pinecone_environment = "us-west1-gcp"

# Initialize Pinecone client
class DrowsinessDetection:
    def __init__(self):
        try:
            st.write("Initializing Drowsiness Detection...")

            # Create an instance of the Pinecone client
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if the index exists, create if it doesn't
            if INDEX_NAME not in self.pc.list_indexes().names():
                st.write(f"Creating Pinecone index: {INDEX_NAME}")
                self.pc.create_index(
                    name=INDEX_NAME,
                    dimension=1024,  # Adjust to your model's output dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                st.write(f"Index '{INDEX_NAME}' created.")
            else:
                st.write(f"Index '{INDEX_NAME}' already exists.")
            
            # Access the index
            self.index = self.pc.Index(INDEX_NAME)  # Access Pinecone index
            
            st.write("Pinecone index is set.")
        
        except Exception as e:
            st.error(f"Error initializing Pinecone: {e}")
            raise  # Raise the exception if initialization fails

    def load_model(self):
        """Load the pre-trained HuggingFace model and feature extractor."""
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor

    def extract_image_features(self, image):
        """Extract features from the image using the model's feature extractor."""
        model, feature_extractor = self.load_model()
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            feature_vector = outputs.logits.squeeze().cpu().numpy()
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
        vector_id = str(np.random.randint(0, 1000000))

        # Create the vector with the ID, feature vector, and metadata
        vector = {
            "id": vector_id,
            "values": feature_vector.tolist(),
            "metadata": metadata
        }

        try:
            # Upsert the vector into the Pinecone index
            upsert_response = self.index.upsert(
                vectors=[vector],
                namespace="ns1"  # Example namespace
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
            logits = outputs.logits  # Get raw prediction scores
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()  # Highest score
        return predicted_class_idx, prediction_score

    def detect_drowsiness(self, image):
        """Detect if the driver is drowsy."""
        model, feature_extractor = self.load_model()
        inputs = self.preprocess_image(image, feature_extractor)
        predicted_class_idx, prediction_score = self.get_prediction(model, inputs)
        return predicted_class_idx, prediction_score

# Streamlit interface
st.title("Driver Drowsiness Detection System")

# Initialize the detection system
drowsiness_detector = DrowsinessDetection()

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame from webcam")
        break

    # Convert frame to PIL image for processing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect drowsiness
    predicted_class_idx, prediction_score = drowsiness_detector.detect_drowsiness(image)

    # Display the result
    st.image(image, caption="Driver's face", use_column_width=True)
    st.write(f"Prediction: {LABELS[predicted_class_idx]}")
    st.write(f"Confidence score: {prediction_score:.2f}")

    # Store the result in Pinecone
    drowsiness_detector.store_in_pinecone(image, predicted_class_idx, prediction_score)

    # Break the loop if 'q' is pressed (for testing purposes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam when done
cap.release()
cv2.destroyAllWindows()
