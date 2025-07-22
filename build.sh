#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download YOLOv8 model if not exists
python -c "
import os
from ultralytics import YOLO
if not os.path.exists('yolov8n.pt'):
    print('Downloading YOLOv8 model...')
    model = YOLO('yolov8n.pt')
    print('Model downloaded successfully!')
else:
    print('YOLOv8 model already exists.')
"
