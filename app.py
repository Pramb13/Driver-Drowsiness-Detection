import streamlit as st
import torch
import numpy as np
import time
import cv2
import mediapipe as mp  # Replace dlib with mediapipe for facial landmarks detection

# Load YOLOv5 model with error handling
@st.cache_resource
def load_model():
    model_path = './best.pt'  # Replace with your custom model path
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLOv5 model: {e}")
        raise

# Load YOLOv5 model
model = load_model()

# Initialize Mediapipe for face and eye landmarks
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Initialize drowsiness detection parameters
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 48
frame_counter = 0

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Streamlit UI elements
st.title("Live Driver Drowsiness Detection")
st.write("Detects drowsiness in real-time using YOLOv5 and Mediapipe.")

# Live camera input
st.markdown("### Activate Camera")
camera_input = st.camera_input("Start your camera to begin detection.")

if camera_input:
    st.markdown("### Live Detection Output")
    stframe = st.empty()  # Placeholder for video feed

    # Open the camera feed directly instead of saving it to a temporary file
    frame_counter = 0
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while True:
                # Capture frame-by-frame
                frame = camera_input.getvalue()  # Get the camera frame
                rgb_frame = np.array(frame)[..., :3]  # Ensure it's RGB

                # YOLOv5 detection for face detection
                results = model(rgb_frame)
                results.render()  # Render detections on the frame

                # Convert the rendered image back to RGB for Streamlit display
                rendered_frame = np.array(results.ims[0])

                # Convert the frame to grayscale for face and eye detection
                gray_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2GRAY)

                # Detect faces in the frame using mediapipe
                faces = face_detection.process(rendered_frame)

                if faces.detections:
                    for detection in faces.detections:
                        # Get the bounding box for the face
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = rendered_frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                      int(bboxC.width * iw), int(bboxC.height * ih)

                        # Crop the face from the frame
                        face_roi = rendered_frame[y:y+h, x:x+w]
                        
                        # Detect facial landmarks
                        results = face_mesh.process(face_roi)
                        if results.multi_face_landmarks:
                            for landmarks in results.multi_face_landmarks:
                                # Get the eye landmarks (indices 33-133 for left eye, 362-463 for right eye)
                                left_eye = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(33, 133)])
                                right_eye = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(362, 463)])

                                # Calculate EAR for both eyes
                                left_ear = eye_aspect_ratio(left_eye)
                                right_ear = eye_aspect_ratio(right_eye)
                                ear = (left_ear + right_ear) / 2.0

                                # Check if the EAR is below the threshold (indicating drowsiness)
                                if ear < EAR_THRESHOLD:
                                    frame_counter += 1
                                    if frame_counter >= CONSEC_FRAMES:
                                        # Driver is drowsy
                                        cv2.putText(rendered_frame, "Drowsy Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                else:
                                    frame_counter = 0

                # Display the rendered frame with drowsiness detection
                stframe.image(rendered_frame, caption="Detected Frame", use_column_width=True)

                # Optional: Add a delay for smoother performance
                time.sleep(0.1)
