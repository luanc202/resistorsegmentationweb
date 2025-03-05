import base64
import json
import numpy as np
import cv2
from flask import Flask
from flask_socketio import SocketIO
from tflite_runtime.interpreter import Interpreter
import threading

# Model metadata
LABELS = [
    "black_belt", "blue_belt", "brown_belt", "gold_belt", "gray_belt",
    "green_belt", "orange_belt", "purple_belt", "red_belt", "resistor",
    "white_belt", "yellow_belt"
]
IMG_SIZE = 640  # From metadata: 640x640

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load TF Lite model
interpreter = Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a lock for thread-safe access to the interpreter
interpreter_lock = threading.Lock()

def preprocess_image(base64_data):
    """Decode and preprocess the base64 image for TF Lite."""
    # Extract base64 string (skip "data:image/jpeg;base64," prefix)
    base64_string = base64_data.split(",")[1]
    img_data = base64.b64decode(base64_string)

    # Decode to OpenCV image
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to decode image")
        return None

    # Resize to 640x640
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert BGR to RGB and normalize to [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # Add batch dimension [1, 640, 640, 3]
    img = np.expand_dims(img, axis=0)
    return img

def post_process(output_data, threshold=0.5):
    """Process TF Lite output to extract detections."""
    detections = []
    # Assuming output is [1, num_detections, 6] (x, y, w, h, conf, class)
    output = output_data[0]  # First output tensor

    for detection in output:
        confidence = detection[4]
        if confidence > threshold:
            x, y, w, h = detection[0], detection[1], detection[2], detection[3]
            class_id = int(detection[5])

            # Convert center coordinates to top-left (x, y, width, height)
            box = [x - w / 2, y - h / 2, w, h]
            label = LABELS[class_id]

            detections.append({"box": box, "label": label})

    return detections

@socketio.on("connect")
def handle_connect():
    print("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

@socketio.on("frame")
def handle_frame(data):
    frame_id = data["id"]
    base64_data = data["data"]

    # Preprocess the image
    img = preprocess_image(base64_data)
    if img is None:
        socketio.emit("error", {"message": "Failed to process image"})
        return

    # Use lock to ensure thread-safe access to the interpreter
    with interpreter_lock:
        # Set input tensor
        interpreter.set_tensor(input_details[0]["index"], img)

        # Run inference
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]["index"])

        # Post-process detections
        detections = post_process(output_data)

        # Send response
        response = {"id": frame_id, "detections": detections}
        socketio.emit("detection", response)

if __name__ == "__main__":
    print("Starting server on port 5000...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
