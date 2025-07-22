from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

model = YOLO('yolov8n.pt')


# Serve the frontend UI
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
  <title>Drone Detection Live</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
      background: #181A20;
      color: #e0e0e0;
      transition: background 0.5s;
    }
    h2 {
      color: #00c3ff;
      font-size: 2.2rem;
      margin-bottom: 20px;
      letter-spacing: 1px;
      text-shadow: 0 2px 8px #00c3ff44;
      animation: fadeInDown 1s;
    }
    @keyframes fadeInDown {
      from { opacity: 0; transform: translateY(-30px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .container {
      background: #23272f;
      border-radius: 16px;
      box-shadow: 0 8px 32px #000a;
      padding: 32px 24px 24px 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
      animation: fadeIn 1.2s;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    video {
      width: 420px;
      height: 320px;
      background: #111;
      border-radius: 12px;
      box-shadow: 0 4px 16px #00c3ff22;
      margin-bottom: 18px;
      border: 2px solid #00c3ff44;
      animation: popIn 1.2s;
    }
    @keyframes popIn {
      0% { transform: scale(0.9); opacity: 0; }
      100% { transform: scale(1); opacity: 1; }
    }
    button {
      margin: 12px 0 10px 0;
      padding: 12px 28px;
      font-size: 18px;
      border: none;
      border-radius: 8px;
      background: linear-gradient(90deg, #00c3ff 0%, #007bff 100%);
      color: #fff;
      cursor: pointer;
      font-weight: 600;
      box-shadow: 0 2px 8px #00c3ff44;
      transition: background 0.3s, transform 0.2s;
      outline: none;
    }
    button:hover {
      background: linear-gradient(90deg, #007bff 0%, #00c3ff 100%);
      transform: scale(1.05);
    }
    #status {
      font-size: 22px;
      margin-top: 18px;
      min-height: 32px;
      font-weight: bold;
      letter-spacing: 1px;
      text-align: center;
      transition: color 0.3s;
    }
    .detected {
      color: #ff3860;
      text-shadow: 0 2px 8px #ff386044;
      animation: pulse 1s infinite alternate;
    }
    .not-detected {
      color: #28a745;
      text-shadow: 0 2px 8px #28a74544;
    }
    @keyframes pulse {
      from { text-shadow: 0 2px 8px #ff386044; }
      to { text-shadow: 0 2px 24px #ff3860cc; }
    }
    #labels {
      margin-top: 18px;
      font-size: 18px;
      color: #00c3ff;
      background: #23272f;
      border-radius: 8px;
      padding: 10px 18px;
      box-shadow: 0 2px 8px #00c3ff22;
      min-width: 320px;
      text-align: center;
      animation: fadeIn 1.2s;
    }
    ::selection {
      background: #00c3ff44;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Live Drone Detection</h2>
    <video id="camera" width="420" height="320" autoplay playsinline></video>
    <button id="openBtn">Open Camera</button>
    <button id="detectBtn" disabled>Detect Drone</button>
    <p id="status">Status: Waiting...</p>
    <div id="labels"></div>
  </div>

  <script>
    const video = document.getElementById('camera');
    const openBtn = document.getElementById('openBtn');
    const detectBtn = document.getElementById('detectBtn');
    const statusEl = document.getElementById('status');
    let streamStarted = false;

    openBtn.onclick = function() {
      if (streamStarted) return;
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(function(stream) {
            video.srcObject = stream;
            streamStarted = true;
            detectBtn.disabled = false;
            statusEl.innerText = 'Camera opened. Ready to detect.';
          })
          .catch(function(error) {
            statusEl.innerText = 'Error accessing camera: ' + error.message;
            statusEl.className = '';
          });
      } else {
        statusEl.innerText = 'Webcam not supported in this browser.';
        statusEl.className = '';
      }
    };

    let detectionInterval = null;
    detectBtn.onclick = function() {
      if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
        detectBtn.innerText = 'Start Live Detection';
        statusEl.innerText = 'Live detection stopped.';
        return;
      }
      detectBtn.innerText = 'Stop Live Detection';
      statusEl.innerText = 'Starting live detection...';
      detectionInterval = setInterval(async () => {
        // Create canvas from video
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 400;
        canvas.height = video.videoHeight || 300;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const base64Image = canvas.toDataURL('image/jpeg');

        try {
          const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
          });
          if (!response.ok) {
            throw new Error('HTTP error! status: ' + response.status);
          }
          const result = await response.json();
          const labelsDiv = document.getElementById('labels');
          if (result.drone_detected) {
            statusEl.innerText = `🚨 Drone Detected! Confidence: ${result.confidence}%`;
            statusEl.className = 'detected';
          } else {
            statusEl.innerText = `✅ No drone detected. Highest confidence: ${result.confidence}%`;
            statusEl.className = 'not-detected';
          }
          // Show all detected labels
          if (result.detected_labels && result.detected_labels.length > 0) {
            labelsDiv.innerHTML = '<b>Detected labels:</b> ' + result.detected_labels.map(l => `${l.label} (${l.confidence}%)`).join(', ');
          } else {
            labelsDiv.innerHTML = '<b>Detected labels:</b> None';
          }
        } catch (error) {
          statusEl.innerText = 'Detection error: ' + error.message;
          statusEl.className = '';
        }
      }, 1500); // Detect every 1.5 seconds
    };
  </script>
</body>
</html>
    ''')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    print('Received data:', data)
    if not data or 'image' not in data:
        return jsonify({
            "error": "No image data received."
        }), 400

    image_b64 = data['image']
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]

    try:
        img_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        frame = np.array(image)
    except Exception as e:
        print('Image decode error:', str(e))
        return jsonify({
            "error": "Invalid image data. Could not decode image.",
            "details": str(e)
        }), 400

    try:
        results = model.predict(frame, imgsz=640, conf=0.3)
        detections = results[0].boxes
    except Exception as e:
        print('Model prediction error:', str(e))
        return jsonify({
            "error": "Model prediction failed.",
            "details": str(e)
        }), 500

    drone_detected = False
    confidence = 0.0
    max_confidence = 0.0
    detected_labels = []

    for box in detections:
        cls = int(box.cls[0])
        label = model.names[cls].lower().strip()
        conf = float(box.conf[0])
        detected_labels.append({"label": label, "confidence": round(conf * 100, 2)})
        print('Detected label:', label, 'Confidence:', conf)
        if conf > max_confidence:
            max_confidence = conf
        if 'drone' in label:
            drone_detected = True
            confidence = conf
            break
        elif 'bird' in label:
            drone_detected = True
            confidence = conf
            break

    if not drone_detected:
        confidence = max_confidence

    return jsonify({
        "drone_detected": drone_detected,
        "confidence": round(confidence * 100, 2),
        "detected_labels": detected_labels
    })

if __name__ == '__main__':
    print("🌐 Starting Flask app with HTTP...")
    app.run(debug=True, port=5000, host='0.0.0.0')