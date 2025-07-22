# Drone Detection Web App

This project is a simple web application for live drone detection using YOLOv8 and your webcam. It uses Flask for the backend and serves a modern HTML frontend for live camera preview and detection.

## Features
- Live webcam preview in browser
- Detects drones (and birds) using YOLOv8
- Shows detection confidence and all detected labels
- Modern dark-themed UI

## Requirements
- Python 3.8+
- `yolov8n.pt` YOLOv8 model file (place in the same folder as `app.py`)
- All dependencies listed in `requirements.txt`

## Installation
1. Clone this repository or download the code.
2. Open a terminal in the project folder.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make sure `yolov8n.pt` is present in the folder.

## Running the App
1. Start the Flask server:
   ```
   python app.py
   ```
2. Open your browser and go to:
   ```
   http://localhost:5000
   ```
3. Click "Open Camera" to start your webcam, then "Detect Drone" to begin live detection.

## Notes
- The app works best in Chrome or Edge browsers.
- If you want to access from another device on your network, use your computer's IP address:
  ```
  http://<your-ip>:5000
  ```
- For HTTPS, you can use tools like `mkcert` to generate a local SSL certificate.

## Troubleshooting
- If you see errors about the YOLO model, make sure `yolov8n.pt` is present and compatible.
- If the webcam doesn't open, check browser permissions and try a different browser.
- For model errors like `'Conv' object has no attribute 'bn'`, update your `ultralytics` package:
  ```
  pip install --upgrade ultralytics
  ```

## License
This project is for educational and demonstration purposes.
