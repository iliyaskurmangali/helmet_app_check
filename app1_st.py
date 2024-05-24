import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from io import BytesIO
import tempfile
import os

# Load the YOLO model
model = YOLO("yolo_v8_25.pt")  # Path to the pre-trained YOLOv5 nano model

# Set up the Streamlit app
st.title("Real-Time Object Detection with YOLOv8")

# Use st.file_uploader to upload a video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    # Read the uploaded video file as a byte stream
    video_bytes = uploaded_file.read()

    # Save the video bytes to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(video_bytes)
    temp_file.close()

    # Function to run inference and save processed video
    def run_inference_on_video(video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Perform inference on the frame
            results = model(frame)

            # Draw bounding boxes and labels on the frame
            for result in results:
                for detection in result.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = map(int, detection[:6])
                    label = f"{model.names[class_id]} {conf:.2f}"
                    color = (0, 255, 0) if model.names[class_id] == "helmet" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Write the processed frame to the output video
            out.write(frame)

        cap.release()
        out.release()

    # Run inference on the uploaded video and save the processed video
    output_file_path = "processed_video.mp4"
    run_inference_on_video(temp_file.name, output_file_path)

    # Provide a link for the user to download the processed video
    st.markdown(f"Download processed video [here](/{output_file_path})")
