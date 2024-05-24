import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("yolo_v8_25.pt")  # Path to the pre-trained YOLO model

# Set up the Streamlit app
st.title("Are you wearing a helmet?")

# Get the camera input
camera_input = st.camera_input("Press the button to start the camera")

if camera_input:
    # Convert the camera input to an OpenCV image
    frame = np.frombuffer(camera_input.getvalue(), np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Perform inference on the frame
    results = model(frame)

    # Initialize counters
    helmet_count = 0
    no_helmet_count = 0

    # Draw the bounding boxes and labels on the frame
    for result in results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = map(int, detection[:6])
            label = f"{model.names[class_id]} {conf:.2f}"
            if model.names[class_id] == "helmet":  # Assuming "helmet" class ID represents people wearing helmets
                color = (0, 255, 0)  # Green for helmet
                helmet_count += 1
            else:
                color = (0, 0, 255)  # Red for no helmet
                no_helmet_count += 1

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw the label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the counts on the frame
    cv2.putText(frame, f"Helmet: {helmet_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"No Helmet: {no_helmet_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert the frame to RGB format for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels='RGB')
