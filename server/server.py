from flask import Flask, send_from_directory, request, jsonify
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from ultralytics import YOLO
import threading
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model metadata
IMG_SIZE = 640

app = Flask(__name__, static_folder='static', static_url_path='')

# Load YOLO model
try:
    model = YOLO("best.pt")
    logger.info("YOLO model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Lock for thread-safe model access
model_lock = threading.Lock()

def preprocess_image(image_data):
    if 'data:image' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    if image.size != (IMG_SIZE, IMG_SIZE):
        image = image.resize((IMG_SIZE, IMG_SIZE))
    return image

@app.route('/')
def serve_html():
    return send_from_directory(app.static_folder, 'main.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data['image']
        if not isinstance(image_data, str):
            return jsonify({"error": "Invalid image data"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        image = preprocess_image(image_data)

        with model_lock:
            results = model.predict(source=image, imgsz=IMG_SIZE, conf=0.5, verbose=False)

        detections = []
        result = results[0]
        orig_width, orig_height = image.size

        if result.boxes:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                label = result.names[int(cls)]
                detections.append({
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "confidence": float(conf),
                    "label": label
                })

        logger.info(f"Found {len(detections)} detections")
        return jsonify({"detections": detections})

    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting server on port 5000...")
    # ssl_context = ('192.168.15.14.pem', '192.168.15.14-key.pem')
    app.run(host="0.0.0.0", port=5000, debug=False)
