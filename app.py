import streamlit as st
from keras.models import load_model
import io
import os
import tensorflow as tf
import h5py

# Display TensorFlow and h5py versions for debugging
st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"h5py version: {h5py.__version__}")

# Define the model path (you could place this in your repository)
model_path = 'drowsiness_model.h5'

# Check if model file exists in the current directory (for static files)
if os.path.exists(model_path):
    st.write(f"Model file found at: {os.path.abspath(model_path)}")
else:
    st.write("Model file not found in the specified path. Please upload a model.")

# File uploader widget for dynamically uploading the model
uploaded_file = st.file_uploader("Upload your model (h5)", type=["h5"])

# Handle file upload or loading from the local path
if uploaded_file is not None:
    try:
        # If a file is uploaded, attempt to load the model from the uploaded file
        model = load_model(io.BytesIO(uploaded_file.read()))
        st.write("Model loaded successfully from uploaded file!")
    except Exception as e:
        # If there is an error loading the model, show the error message
        st.error(f"Error loading model from uploaded file: {str(e)}")
        
elif os.path.exists(model_path):
    # If no file is uploaded, attempt to load the static model file
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully from local path!")
    except Exception as e:
        # If there is an error loading the model, show the error message
        st.error(f"Error loading model from local path: {str(e)}")

# If the model is loaded successfully, you can add inference code here
if 'model' in locals():
    # Example: You can perform inference with your model here
    st.write("Model is ready for inference. Add your code to perform predictions.")
else:
    st.write("Please upload the model or ensure the path is correct.")
