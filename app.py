import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import imutils

# Load YOLOv5 Model
@st.cache_resource(show_spinner=True)
def load_model(path=None):
    try:
        if path:
            return torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
        return torch.hub.load('ultralytics/yolov5', 'yolov5s')
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None

# Function for Image Detection
def detect_image(model, image):
    results = model(image)
    results.render()  # Render results on the image
    return np.squeeze(results.ims[0])

# Function for Real-Time Detection
def real_time_detection(model):
    st.warning("Press the Stop button to exit real-time detection.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access the camera. Please check your camera settings.")
        return

    stop_button = st.button("Stop Detection")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from the camera.")
            break

        results = model(frame)
        frame = np.squeeze(results.render())

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame in Streamlit
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        if stop_button:
            break

    cap.release()

# Function for Android Camera Feed Detection
def android_feed_detection(model, url):
    st.warning("Press the Stop button to exit the Android camera feed detection.")
    stop_button = st.button("Stop Detection")

    while True:
        try:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=500)
            results = model(img)
            img = np.squeeze(results.render())

            # Convert image to RGB for Streamlit display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display image in Streamlit
            st.image(img_rgb, channels="RGB", use_column_width=True)

            if stop_button:
                break
        except requests.exceptions.RequestException as req_e:
            st.error(f"Network error while fetching Android feed: {req_e}")
            break
        except Exception as e:
            st.error(f"Error processing Android feed: {e}")
            break

# Streamlit App
st.title("YOLOv5 Object Detection App")
st.sidebar.title("Options")

# Sidebar Options
mode = st.sidebar.radio(
    "Select Mode",
    ("Image Detection", "Real-Time Detection", "Android Camera Feed"),
)

# Load YOLOv5 Model
model_path = st.sidebar.text_input("Enter Custom Model Path (Optional)", value="")
model = load_model(model_path if model_path else None)

if not model:
    st.stop()

# Image Detection
if mode == "Image Detection":
    st.subheader("Image Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Detecting..."):
            result_img = detect_image(model, np.array(image))
        st.image(result_img, caption="Detection Result", use_column_width=True)

# Real-Time Detection
elif mode == "Real-Time Detection":
    st.subheader("Real-Time Detection")
    if st.button("Start Detection"):
        real_time_detection(model)

# Android Camera Feed
elif mode == "Android Camera Feed":
    st.subheader("Android Camera Feed")
    android_url = st.text_input("Enter Android Camera Feed URL (with /shot.jpg):", value="http://26.2.215.215:8080/shot.jpg")
    if android_url and st.button("Start Detection"):
        android_feed_detection(model, android_url)
