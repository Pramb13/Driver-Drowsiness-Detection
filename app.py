import streamlit as st
import cv2
import torch
import numpy as np
from scipy.spatial import distance
import dlib

# Load YOLOv5 model (pre-trained or custom)
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Replace 'yolov5s' with custom weights if available

model = load_model()

# Load dlib's face detector and shape predictor for facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye blink threshold
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 20  # Number of frames to count eyes closed

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
        counter = 0  # Counter for consecutive frames with eyes closed

        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access webcam. Please check your camera.")
                break

            # Convert frame to grayscale for dlib face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                # Get facial landmarks
                landmarks = predictor(gray, face)

                # Get the coordinates for the left and right eye
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

                # Calculate EAR for both eyes
                left_eye_ear = eye_aspect_ratio(left_eye)
                right_eye_ear = eye_aspect_ratio(right_eye)
                ear = (left_eye_ear + right_eye_ear) / 2.0

                # Check if the eyes are closed
                if ear < EAR_THRESHOLD:
                    counter += 1
                else:
                    counter = 0

                # If eyes are closed for a certain number of frames, trigger drowsiness
                if counter >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # YOLOv5 detection
            labels, cords = detect_objects(frame)
            frame = draw_boxes(frame, labels, cords)

            # Display the frame in Streamlit
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()

if __name__ == "__main__":
    main()
