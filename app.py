mport streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np

# Load YOLOv5 model (pre-trained or custom)
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Replace 'yolov5s' with custom weights if available

model = load_model()

def detect_objects(frame):
    """
    Perform object detection on a video frame using YOLOv5.
    """
    results = model(frame)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cords

def draw_boxes(frame, labels, cords, confidence_threshold=0.3):
    """
    Draw bounding boxes on the frame for detected objects.
    """
    for label, cord in zip(labels, cords):
        if cord[4] < confidence_threshold:  # Filter out low-confidence detections
            continue
        x1, y1, x2, y2 = int(cord[0] * frame.shape[1]), int(cord[1] * frame.shape[0]), \
                         int(cord[2] * frame.shape[1]), int(cord[3] * frame.shape[0])
        conf = cord[4]
        class_name = model.names[int(label)]
        label_text = f"{class_name} {conf:.2f}"
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def main():
    st.title("Driver Drowsiness Detection System")
    st.markdown(
        """
        This application uses YOLOv5 to detect driver drowsiness and related conditions.
        - Start the webcam feed to analyze video frames in real-time.
        - Close the detection checkbox to stop the webcam.
        """
    )

    # Checkbox to start/stop detection
    run_detection = st.checkbox("Start Detection")

    if run_detection:
        # Start webcam feed
        cap = cv2.VideoCapture(0)  # Open webcam
        stframe = st.empty()  # Placeholder for video frames

        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access webcam. Please check your camera.")
                break

            # YOLOv5 detection
            labels, cords = detect_objects(frame)
            frame = draw_boxes(frame, labels, cords)

            # Display the frame in Streamlit
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()

if __name__ == "__main__":
    main() 
