import streamlit as st
from keras.models import load_model
import io
import os
import tensorflow as tf
import h5py

# Display versions for debugging
st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"h5py version: {h5py.__version__}")

# Check if model file exists in the current directory (for static files)
model_path = 'drowsiness_model.h5'

if os.path.exists(model_path):
    st.write(f"Model file found at: {os.path.abspath(model_path)}")
else:
    st.write("Model file not found in the specified path.")

# File uploader widget for dynamically uploading the model
uploaded_file = st.file_uploader("Choose a model file (h5)", type=["h5"])

if uploaded_file is not None:
    # If file is uploaded, attempt to load it
    try:
        # Load the model from the uploaded file
        model = load_model(io.BytesIO(uploaded_file.read()))
        st.write("Model loaded successfully!")
    except Exception as e:
        # If there is an error, display the error message
        st.error(f"Error loading model: {str(e)}")

elif os.path.exists(model_path):
    # If no file is uploaded, attempt to load the static model file
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully from the local path!")
    except Exception as e:
        # If there is an error, display the error message
        st.error(f"Error loading model from local path: {str(e)}")

# Optionally, add code to use the loaded model for inference
# if model is loaded successfully, you can use it here
