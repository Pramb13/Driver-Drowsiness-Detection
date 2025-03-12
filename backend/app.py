import os
import cv2
import numpy as np
import pinecone
import uuid
import mediapipe as mp
from flask import Flask, Response, jsonify

app = Flask(__name__)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "drowsiness-logs"

# Connect to Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Function to log drowsiness in Pinecone
def log_drowsiness(status):
    unique_id = str(uuid.uuid4())  # Generate unique ID
    index.upsert(vectors=[(unique_id, [1 if status == "Drowsy" else 0])])

# Function to detect drowsiness
def detect_drowsiness(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        log_drowsiness("Drowsy")
    return frame

# Video streaming generator
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect_drowsiness(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# API Endpoint: Video Stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API Endpoint: Fetch Drowsiness Logs
@app.route('/logs', methods=['GET'])
def get_logs():
    logs = index.fetch([x['id'] for x in index.describe_index_stats()['namespaces'][''].values()])
    return jsonify(logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
