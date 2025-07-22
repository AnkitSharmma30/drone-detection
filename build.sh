#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download YOLOv8 model if not exists
python3 -c "
import os
from ultralytics import YOLO
if not os.path.exists('yolov8n.pt'):
    print('Downloading YOLOv8 model...')
    try:
        model = YOLO('yolov8n.pt')
        print('Model downloaded successfully!')
    except Exception as e:
        print(f'Error downloading model: {e}')
        # Create a dummy file to prevent re-download attempts
        with open('yolov8n.pt', 'w') as f:
            f.write('dummy')
else:
    print('YOLOv8 model already exists.')
"
