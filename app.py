import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import time
import pandas as pd
import pdfkit

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]

@st.cache_resource
def load_model():
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, feature_extractor):
    image = image.convert("RGB")
    return feature_extractor(images=image, return_tensors="pt")

def get_prediction(model, inputs):
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()
        return predicted_class_idx, prediction_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def generate_pdf(prediction_label, prediction_score):
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; }}
            h2 {{ color: #1E88E5; }}
            .result {{ font-size: 20px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h2>Drowsiness Detection Report</h2>
        <p class='result'><strong>Prediction:</strong> {prediction_label}</p>
        <p class='result'><strong>Confidence Score:</strong> {prediction_score:.2f}</p>
    </body>
    </html>
    """
    pdf_path = "drowsiness_report.pdf"
    pdfkit.from_string(html_content, pdf_path)
    return pdf_path

def generate_csv(prediction_label, prediction_score):
    df = pd.DataFrame([[prediction_label, prediction_score]], columns=["Prediction", "Confidence Score"])
    csv_path = "drowsiness_report.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def display_result(image, predicted_class_idx, prediction_score):
    st.image(image, caption="Captured Image from Webcam", use_container_width=True)
    if predicted_class_idx is not None:
        prediction_label = LABELS[predicted_class_idx]
        st.write(f"Prediction: {prediction_label} with confidence {prediction_score:.2f}")
        
        if st.button("Download PDF Report"):
            pdf_path = generate_pdf(prediction_label, prediction_score)
            with open(pdf_path, "rb") as file:
                st.download_button("Download PDF Report", data=file, file_name="drowsiness_report.pdf", mime="application/pdf")
        
        if st.button("Download CSV Report"):
            csv_path = generate_csv(prediction_label, prediction_score)
            with open(csv_path, "rb") as file:
                st.download_button("Download CSV Report", data=file, file_name="drowsiness_report.csv", mime="text/csv")
    else:
        st.write("Error: Could not make a prediction.")

def main():
    st.title("Real-Time Drowsiness Detection")
    st.markdown("This application detects drowsiness using a deep learning model.")
    
    model, feature_extractor = load_model()
    if model is None or feature_extractor is None:
        st.error("Failed to load the model. Please check your internet connection or try again later.")
        return

    camera_input = st.camera_input("Webcam feed for real-time drowsiness detection")
    if camera_input is not None:
        with st.spinner("Processing..."):
            time.sleep(1)
            img = Image.open(camera_input)
            inputs = preprocess_image(img, feature_extractor)
            predicted_class_idx, prediction_score = get_prediction(model, inputs)
            display_result(img, predicted_class_idx, prediction_score)

if __name__ == "__main__":
    main()
