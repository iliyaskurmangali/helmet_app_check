import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO

model = YOLO("yolo_v8_25.pt")  # Path to the pre-trained YOLOv5 nano model
st.title("Are you wearing a helmet?")
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    result = model(img)
    return av.VideoFrame.from_ndarray(result[0].plot(), format="bgr24")
webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    async_processing=True,
)
