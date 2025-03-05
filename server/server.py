import base64
import json
import numpy as np
import cv2
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from tflite_runtime.interpreter import Interpreter
import threading
import os

# Model metadata
LABELS = [
    "black_belt", "blue_belt", "brown_belt", "gold_belt", "gray_belt",
    "green_belt", "orange_belt", "purple_belt", "red_belt", "resistor",
    "white_belt", "yellow_belt"
]
IMG_SIZE = 640

# Initialize Flask and SocketIO
app = Flask(__name__, static_folder='static', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*")

# Load TF Lite model
interpreter = Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a lock for thread-safe access to the interpreter
interpreter_lock = threading.Lock()

# Serve the HTML file at the root URL
@app.route('/')
def serve_html():
    return send_from_directory(app.static_folder, 'main.html')

def preprocess_image(base64_data):
    base64_string = base64_data.split(",")[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to decode image")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def post_process(output_data, threshold=0.5):
    detections = []
    output = output_data[0]
    for detection in output:
        confidence = detection[4]
        if confidence > threshold:
            x, y, w, h = detection[0], detection[1], detection[2], detection[3]
            class_id = int(detection[5])
            x_center = float(x) * IMG_SIZE
            y_center = float(y) * IMG_SIZE
            width = float(w) * IMG_SIZE
            height = float(h) * IMG_SIZE
            box = [x_center - width / 2, y_center - height / 2, width, height]
            label = LABELS[class_id]
            detections.append({"box": box, "label": label})
    return detections

@socketio.on("connect")
def handle_connect():
    print("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

@socketio.on("image")
def handle_image(data):
    base64_data = data["data"]
    img = preprocess_image(base64_data)
    if img is None:
        socketio.emit("error", {"message": "Failed to process image"})
        return
    with interpreter_lock:
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        detections = post_process(output_data)
        response = {"detections": detections}
        socketio.emit("detection", response)

if __name__ == "__main__":
    print("Starting server on port 5000...")
    ssl_context = ('192.168.15.14.pem', '192.168.15.14-key.pem')
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, ssl_context=ssl_context)
