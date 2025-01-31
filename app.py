import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import pdfkit
import os

# Constants
MODEL_NAME = "facebook/dino-vits16"
LABELS = ["Not Drowsy", "Drowsy"]
USER_CREDENTIALS = {"user": "123"}
ADMIN_CREDENTIALS = {"admin": "123"}

class DrowsinessDetection:
    def __init__(self):
        self.model, self.feature_extractor = self.load_model()

    def load_model(self):
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        return model, feature_extractor

    def get_prediction(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            prediction_score = logits.max().item()
        return predicted_class_idx, prediction_score

# Generate PDF output
def generate_pdf(prediction_label, prediction_score, image):
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

# Sidebar authentication
def sidebar():
    st.sidebar.title("Drowsiness Detection System")
    role = st.sidebar.radio("Select Role", ("User", "Admin"))
    
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if role == "User" and username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.sidebar.success("Login successful! You are logged in as User.")
            return "user"
        elif role == "Admin" and username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
            st.sidebar.success("Login successful! You are logged in as Admin.")
            return "admin"
        else:
            st.sidebar.error("Invalid credentials. Please try again.")
            return None

# Main function
def main():
    role = sidebar()
    
    if role == "user":
        st.title("Real-time Drowsiness Detection")
        drowsiness_detector = DrowsinessDetection()
        camera_input = st.camera_input("Capture an image for drowsiness detection")
        
        if camera_input is not None:
            img = Image.open(camera_input)
            predicted_class_idx, prediction_score = drowsiness_detector.get_prediction(img)
            prediction_label = LABELS[predicted_class_idx]
            
            st.image(img, caption="Captured Image", use_container_width=True)
            st.write(f"**Prediction**: {prediction_label}")
            st.write(f"**Confidence Score**: {prediction_score:.2f}")
            
            if st.button("Download Report"):
                pdf_path = generate_pdf(prediction_label, prediction_score, img)
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download PDF Report", 
                        data=file, 
                        file_name="drowsiness_report.pdf", 
                        mime="application/pdf"
                    )

    elif role == "admin":
        st.title("Admin Dashboard")
        st.write("Admin functionalities will be added here.")

if __name__ == "__main__":
    main()
