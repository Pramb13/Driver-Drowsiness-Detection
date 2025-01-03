import streamlit as st
import pytorch
import cv2
import numpy as np
from PIL import Image
import requests
import imutils

# Load YOLOv5 Model
@st.cache_resource
def load_model(path=None):
    if path:
        return torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function for Image Detection
def detect_image(model, image):
    results = model(image)
    results.render()  # Render results on the image
    return np.squeeze(results.ims[0])

# Function for Real-Time Detection
def real_time_detection(model):
    st.warning("Press 'q' to exit the live detection.")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access the camera.")
            break

        results = model(frame)
        frame = np.squeeze(results.render())

        cv2.imshow("Real-Time YOLOv5 Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function for Android Camera Feed Detection
def android_feed_detection(model, url):
    st.warning("Press 'Esc' to exit the Android camera feed detection.")
    while True:
        try:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=500)
            results = model(img)
            img = np.squeeze(results.render())

            cv2.imshow("Android Camera Feed Detection", img)
            if cv2.waitKey(1) == 27:  # Esc key to exit
                break
        except Exception as e:
            st.error(f"Error fetching Android feed: {e}")
            break
    cv2.destroyAllWindows()

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
