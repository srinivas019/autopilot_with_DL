import cv2
import numpy as np
import streamlit as st
import random
from ultralytics import YOLO

# Streamlit App Title
st.title("🎥 YOLOv8 Object Detection - Upload a Video")

# Upload video through Streamlit
uploaded_file = st.file_uploader("📂 Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# Load class labels
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Load YOLOv8 Model
model = YOLO("weights/yolov8n.pt", "v8")

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("⚠️ Error: Could not open video.")
        return

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Predict on frame
        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        boxes = detect_params[0].boxes

        # Draw detections
        for box in boxes:
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw bounding box
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[clsID], 3)

            # Display class and confidence
            label = f"{class_list[clsID]}: {conf:.2%}"
            cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display output in Streamlit
        frame_placeholder.image(frame, channels="RGB", caption="Processing Video...", use_column_width=True)

    cap.release()
    st.success("✅ Video Processing Completed!")

# Process video if uploaded
if uploaded_file:
    # Save the uploaded file temporarily
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)  # Show the uploaded video
    process_video(video_path)  # Process the video
