import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("yolo_v8_25.pt")  # Path to the pre-trained YOLO model

# Define a video processor class
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform inference on the frame
        results = self.model(img)

        # Draw the bounding boxes and labels on the frame
        for result in results:
            for detection in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = map(int, detection[:6])
                label = f"{self.model.names[class_id]} {conf:.2f}"
                color = (0, 255, 0)  # Bounding box color (green)
                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # Draw the label
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Set up the Streamlit app
st.title("Real-Time Object Detection with YOLOv8")

# Add a WebRTC streamer component
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
