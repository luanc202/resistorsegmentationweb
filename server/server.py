import cv2
from flask import Flask, send_from_directory, request, Response
from ultralytics import YOLO
import numpy as np
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
model = YOLO("best.pt")

# Lock for thread-safe model access
model_lock = threading.Lock()

@app.route('/')
def serve_html():
    return send_from_directory(app.static_folder, 'main.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return Response("No image provided", status=400)

    file = request.files['image']
    try:
        # Read and decode the image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")

        # Process with YOLO and get annotated image
        with model_lock:
            results = model.predict(source=img, imgsz=IMG_SIZE)
            annotated_image = results[0].plot()

        # Encode to JPEG and send back
        _, encoded_image = cv2.imencode('.jpg', annotated_image)
        return Response(encoded_image.tobytes(), content_type='image/jpeg')
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return Response(str(e), status=500)

if __name__ == "__main__":
    print("Starting server on port 5000...")
    ssl_context = ('192.168.15.14.pem', '192.168.15.14-key.pem')
    app.run(host="0.0.0.0", port=5000, debug=False, ssl_context=ssl_context)
