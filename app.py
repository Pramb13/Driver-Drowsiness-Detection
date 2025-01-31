import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from datetime import datetime
import pandas as pd

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
    </style>
""", unsafe_allow_html=True)

# Initialize the model and feature extractor
@st.cache_resource  # Cache the model loading to avoid reloading it each time
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

# Store user prediction history
def store_prediction(username, label, confidence):
    """Store prediction history in user records"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if username not in user_records:
        user_records[username] = []
    user_records[username].append({"timestamp": timestamp, "label": label, "confidence": confidence})

# Capture and display webcam feed
def webcam_feed(model, feature_extractor, username):
    """Capture webcam feed and make predictions in real-time."""
    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    
    if camera_input:
        img = Image.open(camera_input)
        inputs = preprocess_image(img, feature_extractor)
        
        # Get prediction
        predicted_class_idx, prediction_score = get_prediction(model, inputs)
        
        # Display prediction result
        st.image(img, caption="Captured Image from Webcam", use_container_width=True)
        prediction_label = LABELS[predicted_class_idx]
        st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")
        
        # Store prediction in user history
        store_prediction(username, prediction_label, prediction_score)

# Admin page to view all users' prediction records
def admin_page():
    st.title("Admin Dashboard")
    
    # Display all user records
    st.subheader("All User Prediction Records")
    
    # Create a table for records
    if user_records:
        all_records = []
        for username, records in user_records.items():
            for record in records:
                all_records.append({"username": username, "timestamp": record["timestamp"], "label": record["label"], "confidence": record["confidence"]})
        
        df = pd.DataFrame(all_records)
        st.dataframe(df)  # Display the records as a table
    else:
        st.write("No records available yet.")

# User page to view personal prediction records
def user_page(username):
    st.title(f"{username}'s Prediction History")
    
    # Display user-specific records
    if username in user_records and user_records[username]:
        records = user_records[username]
        df = pd.DataFrame(records)
        st.dataframe(df)  # Display the user's records as a table
    else:
        st.write("No predictions available for this user.")

# Main function for the app
def main():
    # Display login form
    st.title("Drowsiness Detection App")

    # Create a login form
    st.subheader("Please log in to continue")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        role = authenticate(username, password)
        
        if role is None:
            st.error("Invalid credentials! Please try again.")
        else:
            # Login successful, redirect user based on role
            st.session_state.username = username  # Store username in session
            if role == "admin":
                st.session_state.role = "admin"
                admin_page()  # Show admin dashboard
            elif role == "user":
                st.session_state.role = "user"
                st.success(f"Welcome {username}!")
                webcam_feed(load_model()[0], load_model()[1], username)  # Real-time prediction for the user

    # If the user is already logged in, show the user or admin page
    if "username" in st.session_state:
        if st.session_state.role == "admin":
            admin_page()
        elif st.session_state.role == "user":
            user_page(st.session_state.username)

if __name__ == "__main__":
    main()
