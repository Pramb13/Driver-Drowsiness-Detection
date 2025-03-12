from flask import Flask, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import sqlite3

app = Flask(__name__)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

# Initialize database
def init_db():
    conn = sqlite3.connect("drowsiness.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, status TEXT)")
    conn.commit()
    conn.close()

init_db()

# Function to log drowsiness
def log_drowsiness(status):
    conn = sqlite3.connect("drowsiness.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO logs (status) VALUES (?)", (status,))
    conn.commit()
    conn.close()

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
    conn = sqlite3.connect("drowsiness.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs")
    logs = cursor.fetchall()
    conn.close()
    return jsonify(logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
