import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def get_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        return landmarks
    return None

def draw_landmarks(image, landmarks):
    mp_drawing.draw_landmarks(image, landmarks, mp_face_mesh.FACEMESH_CONTOURS)

# Start video capture
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Process the frame to get face landmarks
    landmarks = get_landmarks(frame)
    if landmarks:
        draw_landmarks(frame, landmarks)

    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
