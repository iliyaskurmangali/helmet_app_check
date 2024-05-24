import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO("yolo_v8_25.pt")  # Path to the pre-trained YOLOv5 nano model

# Set up the Streamlit app
st.title("Real-Time Object Detection with YOLOv8")

# Use st.file_uploader to upload a video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    # Read the uploaded video file
    video_bytes = uploaded_file.read()
    cap = cv2.VideoCapture(video_bytes)

    # Function to run inference and display video frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform inference on the frame
        results = model(frame)

        # Display the processed frame
        st.image(frame, channels='BGR')

    cap.release()
