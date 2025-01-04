import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import imutils
import onnxruntime as ort

# Load YOLOv5 Model using ONNX Runtime
@st.cache_resource
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

# Preprocess Input Image
def preprocess_image(image, input_shape=(640, 640)):
    resized_image = cv2.resize(image, input_shape)
    img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # Convert HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Post-process YOLO Outputs
def postprocess_output(outputs, original_image, conf_threshold=0.5):
    boxes, scores, labels = [], [], []
    for output in outputs[0]:
        conf = output[4]  # Confidence score
        if conf > conf_threshold:
            x, y, w, h = output[:4]  # Bounding box coordinates
            label = int(output[5])  # Class label
            boxes.append([int(x), int(y), int(w), int(h)])
            scores.append(conf)
            labels.append(label)
    
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return original_image

# Streamlit App
st.title("YOLOv5 Object Detection App (ONNX Runtime)")
st.sidebar.title("Options")

# Sidebar Options
mode = st.sidebar.radio(
    "Select Mode",
    ("Image Detection", "Real-Time Detection", "Android Camera Feed"),
)

# Load ONNX Model
onnx_model_path = st.sidebar.text_input("Enter ONNX Model Path", value="yolov5s.onnx")
model = load_onnx_model(onnx_model_path)

# Image Detection
if mode == "Image Detection":
    st.subheader("Image Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detection
        with st.spinner("Detecting..."):
            input_image = preprocess_image(image_np)
            outputs = model.run(None, {"images": input_image})
            result_img = postprocess_output(outputs, image_np)

        st.image(result_img, caption="Detection Result", use_column_width=True)
