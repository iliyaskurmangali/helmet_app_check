import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time

# Load the YOLO model
model = YOLO("best_v8_50.pt")  # Path to the pre-trained YOLOv5 model

# Set up the Streamlit app
st.set_page_config(page_title="Safety Helmet Detection", page_icon=":construction_worker:", layout="wide")
st.title(":construction_worker: Safety Helmet Detection")
st.markdown("""
This application performs real-time object detection using the YOLOv8 model. Adjust the settings in the sidebar to customize the detection.
""")

# Sidebar for additional settings and information
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, help="Adjust the confidence threshold for detections.")
draw_boxes = st.sidebar.checkbox("Draw Bounding Boxes", True, help="Toggle to enable/disable drawing bounding boxes.")
draw_circles = st.sidebar.checkbox("Draw Bounding Circles", False, help="Toggle to enable/disable drawing bounding circles.")

# Color pickers for bounding box colors
helmet_color = st.sidebar.color_picker("Helmet Color", "#00FF00", help="Choose color for helmet bounding boxes.")
no_helmet_color = st.sidebar.color_picker("No Helmet Color", "#FF0000", help="Choose color for no helmet bounding boxes.")

# Function to convert hex to BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# Convert hex colors to BGR for OpenCV
helmet_color_bgr = hex_to_bgr(helmet_color)
no_helmet_color_bgr = hex_to_bgr(no_helmet_color)

# Initialize session state for counts
if 'helmet_count' not in st.session_state:
    st.session_state.helmet_count = 0
if 'no_helmet_count' not in st.session_state:
    st.session_state.no_helmet_count = 0

# Function to process video frames
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Perform inference on the frame
    results = model(img, conf=confidence_threshold)

    # Reset counts for each frame
    helmet_count = 0
    no_helmet_count = 0

    # Draw bounding boxes or circles and labels on the frame
    for detection in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = map(int, detection[:6])

        if model.names[class_id] == "helmet":
            color = helmet_color_bgr  # Use the selected helmet color
            helmet_count += 1
        else:
            color = no_helmet_color_bgr  # Use the selected no helmet color
            no_helmet_count += 1

        # Draw the bounding box or circle
        if draw_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        elif draw_circles:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = (x2 - x1) // 2
            cv2.circle(img, center, radius, color, 2)

    # Display the counts on the frame
    cv2.putText(img, f"Helmet Count: {helmet_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, helmet_color_bgr, 2)
    cv2.putText(img, f"No Helmet Count: {no_helmet_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, no_helmet_color_bgr, 2)

    # Update the counts in session state
    st.session_state.helmet_count = helmet_count
    st.session_state.no_helmet_count = no_helmet_count

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streaming
ctx = webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},  # Disable audio
)

# Sidebar for displaying detection counts
st.sidebar.markdown("### 📊 Detection Counts")
placeholder_helmet = st.sidebar.empty()
placeholder_no_helmet = st.sidebar.empty()

# Function to update sidebar counts
def update_sidebar():
    while True:
        if ctx.state.playing:
            with placeholder_helmet:
                placeholder_helmet.markdown(f"**Helmet Count:** <span style='color: {helmet_color};'>{st.session_state.helmet_count}</span>", unsafe_allow_html=True)
            with placeholder_no_helmet:
                placeholder_no_helmet.markdown(f"**No Helmet Count:** <span style='color: {no_helmet_color};'>{st.session_state.no_helmet_count}</span>", unsafe_allow_html=True)
        time.sleep(1)  # Control the update frequency to avoid overwhelming the UI

# Start the sidebar update function in a separate thread
thread = threading.Thread(target=update_sidebar)
thread.start()

# Information and instructions
st.sidebar.markdown("### ℹ️ Instructions")
st.sidebar.markdown("""
1. Adjust the confidence threshold to filter detections.
2. Toggle drawing bounding boxes or circles on/off.
3. Choose colors for the bounding boxes.
4. Monitor the detection counts in real-time.
5. View the video stream with detection annotations.
""")

# Footer with more information
st.sidebar.markdown("### 📄 More Information")
st.sidebar.markdown("""
- **Model:** YOLOv8
- **Developer:** Nick, Karush, Sayaka, Iliyas
- **Source Code:** [GitHub](https://github.com/iliyaskurmangali/helmet_app_check)
""")
