import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Initialize mixer
mixer.init()
mixer.music.load("music.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the dlib model and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Landmark indices for the eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Thresholds
THRESHOLD = 0.25
FRAME_CHECK = 20

class DrowsinessDetection(VideoProcessorBase):
    def __init__(self):
        self.flag = 0

    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        subjects = detector(gray, 0)
        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < THRESHOLD:
                self.flag += 1
                if self.flag >= FRAME_CHECK:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                self.flag = 0

        return frame

# Streamlit app UI
st.title("Drowsiness Detection System")
st.write("This application uses a webcam feed to detect drowsiness using eye aspect ratio.")

# Start video stream
webrtc_streamer(key="drowsiness-detection", video_processor_factory=DrowsinessDetection)
