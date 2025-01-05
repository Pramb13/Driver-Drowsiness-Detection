import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Initialize mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load models and cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

model = load_model('dl_model/drowsiness_cnn.h5')

class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.score = 0
        self.thicc = 2

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye_cascade.detectMultiScale(gray)
        right_eye = reye_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        lbl = ''
        rpred = [99]
        lpred = [99]

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
            r_eye = cv2.resize(r_eye, (32, 32))
            r_eye = r_eye.reshape((-1, 32, 32, 3))
            rpred = np.argmax(model.predict(r_eye), axis=-1)
            lbl = 'Open' if rpred[0] == 1 else 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
            l_eye = cv2.resize(l_eye, (32, 32))
            l_eye = l_eye.reshape((-1, 32, 32, 3))
            lpred = np.argmax(model.predict(l_eye), axis=-1)
            lbl = 'Open' if lpred[0] == 1 else 'Closed'
            break

        if rpred[0] == 0 and lpred[0] == 0:
            self.score += 1
            cv2.putText(frame, "Closed", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            self.score -= 1
            cv2.putText(frame, "Open", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if self.score < 0:
            self.score = 0

        cv2.putText(frame, f'Score: {self.score}', (100, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if self.score > 12:
            try:
                sound.play()
            except:
                pass
            self.thicc = min(16, self.thicc + 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), self.thicc)
        else:
            self.thicc = max(2, self.thicc - 2)

        return frame

# Streamlit interface
st.title("Drowsiness Detection System")
st.write("This application uses your webcam to detect drowsiness and alerts you if you seem sleepy.")

webrtc_streamer(key="drowsiness", video_transformer_factory=DrowsinessDetector, rtc_configuration={
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})
