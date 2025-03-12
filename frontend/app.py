import streamlit as st
import requests

# Backend API URL (Replace with your Render URL)
FLASK_BACKEND_URL = "https://your-backend.onrender.com"

st.title("🚗 Driver Drowsiness Detection System")

# Show Video Stream
st.image(f"{FLASK_BACKEND_URL}/video_feed")

# Fetch Logs
if st.button("Show Drowsiness Logs"):
    response = requests.get(f"{FLASK_BACKEND_URL}/logs")
    if response.status_code == 200:
        logs = response.json()
        for log in logs:
            st.write(f"🕒 Log ID: {log[0]} - Status: {log[1]}")
    else:
        st.error("Failed to fetch logs")
