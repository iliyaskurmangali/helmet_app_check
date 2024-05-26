import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO("yolo_v8_25.pt")  # Path to the pre-trained YOLOv5 nano model

# Set up the Streamlit app
st.title("Real-Time Object Detection with YOLOv8")

# Sidebar for additional settings and information
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

# Initialize session state for counts
if 'helmet_count' not in st.session_state:
    st.session_state.helmet_count = 0
if 'no_helmet_count' not in st.session_state:
    st.session_state.no_helmet_count = 0

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Perform inference on the frame
    results = model(img, conf=confidence_threshold)

    # Reset counts for each frame
    helmet_count = 0
    no_helmet_count = 0

    # Draw bounding boxes and labels on the frame
    for result in results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = map(int, detection[:6])
            label = f"{model.names[class_id]} {conf:.2f}"

            if model.names[class_id] == "helmet":
                color = (0, 255, 0)  # Green for helmet
                helmet_count += 1
            else:
                color = (0, 0, 255)  # Red for no helmet
                no_helmet_count += 1

            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # Draw the label
            # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update the counts in session state
    st.session_state.helmet_count = helmet_count
    st.session_state.no_helmet_count = no_helmet_count

    # Display the counts on the frame
    cv2.putText(img, f"Helmet: {helmet_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"No Helmet: {no_helmet_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

# Display helmet count and no helmet count
st.sidebar.markdown(f"**Helmet Count:** {st.session_state.helmet_count}")
st.sidebar.markdown(f"**No Helmet Count:** {st.session_state.no_helmet_count}")
