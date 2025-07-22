from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# You need to download the yolov8n.pt model file for this to work.
# Or replace it with a model trained specifically on drones for better accuracy.
model = YOLO('yolov8n.pt') 

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    # Debug: log incoming data
    print('Received data:', data)
    if not data or 'image' not in data:
        return jsonify({
            "error": "No image data received."
        }), 400

    image_b64 = data['image']
    # Accept both 'data:image/jpeg;base64,...' and raw base64
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

    # Accept both 'drone' and 'bird' as possible labels
    for box in detections:
        cls = int(box.cls[0])
        label = model.names[cls].lower().strip()
        conf = float(box.conf[0])
        detected_labels.append({"label": label, "confidence": round(conf * 100, 2)})
        print('Detected label:', label, 'Confidence:', conf)
        if conf > max_confidence:
            max_confidence = conf
        # Match 'drone' exactly or as substring
        if 'drone' in label:
            drone_detected = True
            confidence = conf
            break
        # Optionally, match 'bird' as a fallback
        elif 'bird' in label:
            drone_detected = True
            confidence = conf
            break

    # If no drone detected, show max confidence for any object
    if not drone_detected:
        confidence = max_confidence

    return jsonify({
        "drone_detected": drone_detected,
        "confidence": round(confidence * 100, 2),
        "detected_labels": detected_labels
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
