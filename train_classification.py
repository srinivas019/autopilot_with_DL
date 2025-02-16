# Load YOLOv8n-cls, train it on mnist160 for 3 epochs and predict an image with it
from ultralytics import YOLO
import streamlit as st
model = YOLO('weights/yolov8n-cls.pt')  # load a pretrained YOLOv8n classification model
model.train(data='/Users/mamid/OneDrive/Desktop/edunet/datasets/animals', epochs=100)  # train the model
model('inference/images/bird.jpeg')  # predict on an image
print(model)