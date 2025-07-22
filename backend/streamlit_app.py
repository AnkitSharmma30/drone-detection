import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="Drone & Object Detection", layout="centered")
st.title("Live Drone & Object Detection")

model = YOLO('yolov8n.pt')

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

if run:
    camera = cv2.VideoCapture(0)
    st.write("Camera started. Showing live detections...")
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to grab frame from camera.")
            break
        # Convert to RGB for PIL/YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, imgsz=640, conf=0.3)
        boxes = results[0].boxes
        labels = []
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            labels.append(f"{label} ({conf*100:.2f}%)")
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        FRAME_WINDOW.image(frame, channels='BGR')
        st.write("Detected labels:", ', '.join(labels) if labels else "None")
        if not st.session_state.get('run', True):
            break
    camera.release()
else:
    st.write("Enable the checkbox to start camera.")
