import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import time
from datetime import datetime

# Constants
MODEL_NAME = "facebook/dino-vits16"  # Example model for image classification
LABELS = ["Not Drowsy", "Drowsy"]  # Example labels (adjust as per your model)

# Apply custom CSS for better UI styling
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            color: #333;
        }
        .title {
            font-family: 'Helvetica', sans-serif;
            color: #0077b6;
            text-align: center;
            margin-top: 50px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #0077b6;
            margin-bottom: 30px;
        }
        .prediction-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .prediction-label {
            font-size: 24px;
            font-weight: bold;
            color: #ff6347;
        }
        .confidence-score {
            font-size: 18px;
            color: #2e8b57;
        }
        .result-container {
            margin-top: 30px;
            display: flex;
            justify-content: center;
        }
        .image-container {
            width: 70%;
            margin-top: 20px;
        }
        .camera-feed {
            margin-bottom: 10px;
        }
        .history-container {
            margin-top: 50px;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .history-title {
            font-size: 22px;
            font-weight: bold;
            color: #0077b6;
            margin-bottom: 15px;
        }
        .history-item {
            margin-bottom: 15px;
        }
        .history-item p {
            margin: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize the model and feature extractor
@st.cache_resource  # Cache the model loading to avoid reloading it each time
def load_model():
    """Load pre-trained model and feature extractor."""
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Preprocess image for the model
def preprocess_image(image, feature_extractor):
    """Preprocess the image for model prediction."""
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Make prediction using the model
def get_prediction(model, inputs):
    """Make a prediction using the model and return the predicted class and confidence."""
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Get the raw prediction scores
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()  # Highest score value
        return predicted_class_idx, prediction_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Store the prediction result in session state
def store_prediction(predicted_class_idx, prediction_score):
    """Store the prediction result along with timestamp in session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []  # Initialize history if it doesn't exist

    # Add new prediction with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "timestamp": timestamp,
        "label": LABELS[predicted_class_idx],
        "confidence": prediction_score
    }
    st.session_state.history.append(result)

# Display the image and prediction result
def display_result(image, predicted_class_idx, prediction_score):
    """Display the image along with the prediction result."""
    # Display the image and results inside a container
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predicted_class_idx is not None:
        prediction_label = LABELS[predicted_class_idx]
        st.markdown(f'<div class="prediction-container">', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction-label">{prediction_label}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence-score">Confidence: {prediction_score:.2f}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Display prediction history
def display_history():
    """Display the prediction history."""
    if 'history' in st.session_state and len(st.session_state.history) > 0:
        st.markdown('<div class="history-container">', unsafe_allow_html=True)
        st.markdown('<p class="history-title">Prediction History</p>', unsafe_allow_html=True)
        for item in st.session_state.history:
            st.markdown(f'<div class="history-item">', unsafe_allow_html=True)
            st.markdown(f'<p><strong>Timestamp:</strong> {item["timestamp"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p><strong>Prediction:</strong> {item["label"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p><strong>Confidence:</strong> {item["confidence"]:.2f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Main Streamlit interface
def main():
    """Main function to handle Streamlit interface and prediction process."""
    st.markdown('<h1 class="title">Real-Time Drowsiness Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">This application uses a deep learning model to detect drowsiness based on images captured from your webcam. Please allow camera access and make sure you are well-lit for accurate results.</p>', unsafe_allow_html=True)

    # Load model and feature extractor
    model, feature_extractor = load_model()

    if model is None or feature_extractor is None:
        st.error("Failed to load the model. Please check your internet connection or try again later.")
        return

    # Capture image from webcam
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection", key="camera_input")

    if camera_input is not None:
        # Show loading indicator while processing
        with st.spinner("Processing..."):
            time.sleep(1)  # Simulate a small delay for better UX
            img = Image.open(camera_input)

            # Preprocess image
            inputs = preprocess_image(img, feature_extractor)

            # Get prediction
            predicted_class_idx, prediction_score = get_prediction(model, inputs)

            # Store the result in history
            store_prediction(predicted_class_idx, prediction_score)

            # Display the result
            display_result(img, predicted_class_idx, prediction_score)

    # Display prediction history
    display_history()

if __name__ == "__main__":
    main()

