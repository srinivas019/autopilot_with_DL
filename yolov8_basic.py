import streamlit as st
from ultralytics import YOLO


# Load the model
model = YOLO("yolov8n.pt")

# Predict on an image
detection_output = model.predict(source="inference/images/lo.jpg", conf=0.25, save=True)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())