import random
import cv2
import numpy as np
import streamlit as st
import pyttsx3  # Text-to-Speech
from ultralytics import YOLO

# Initialize Streamlit App
st.title("Autopilot-X: Revolutionizing Transportation with AI-Based Smart Driving Systems")
st.sidebar.header("Settings")


# Load YOLOv8 Model
model = YOLO("weights/yolov8n.pt", "v8")

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speed

# Load class labels
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for detection
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Distance Estimation Constants
KNOWN_WIDTH = 0.45  # Example width of an object (meters)
FOCAL_LENGTH = 500  # Pre-calibrated focal length

# Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    st.stop()

# Streamlit video display
frame_placeholder = st.empty()
frame_width = int(cap.get(3))
last_spoken = {}

# Start Webcam Stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Error: Couldn't read frame from webcam.")
        break

    # Predict on frame
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    boxes = detect_params[0].boxes

    for box in boxes:
        clsID = int(box.cls.numpy()[0])
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]
        perceived_width = bb[2] - bb[0]
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width if perceived_width > 0 else 0

        # Draw bounding box
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[clsID], 3)
        label = f"{class_list[clsID]}: {conf:.2%}, {distance:.2f}m"
        cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        object_center_x = (bb[0] + bb[2]) / 2
        if object_center_x < frame_width / 3:
            direction = "Turn Right"
        elif object_center_x > (2 * frame_width) / 3:
            direction = "Turn Left"
        else:
            direction = "Move Forward"

    warning_placeholder = st.empty()  # Create a placeholder for warnings

    if distance < 100:  # If object detected within 100 meters
         warning_text = f"⚠️ {class_list[clsID]} detected {distance:.2f}m ahead! Slow down the car!"
         warning_placeholder.warning(warning_text)  # Update warning message
    
    if clsID not in last_spoken or last_spoken[clsID] != "Slow down":
        speech_text = f"Warning! {class_list[clsID]} detected {distance:.2f} meters ahead. Slow down the car!"
        engine.say(speech_text)
        engine.runAndWait()
        last_spoken[clsID] = "Slow down"
    else:
         warning_placeholder.empty()  # Clear previous warnings if no object is detected



    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB", caption="Live Object Detection", use_container_width=True)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

cap.release()
cv2.destroyAllWindows()


