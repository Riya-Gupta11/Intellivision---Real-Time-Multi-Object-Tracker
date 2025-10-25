from flask import Flask, render_template, Response, request, jsonify
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from threading import Lock
import time
import csv
import io
from datetime import datetime
import re

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.50
USE_GPU = torch.cuda.is_available()
IMG_SIZE = 320
HALF_PRECISION = USE_GPU
FRAME_SKIP = 2
RETRY_ATTEMPTS = 5
RETRY_DELAY = 1

app = Flask(__name__)

# --- Global Variables ---
camera_lock = Lock()
current_camera_source = "0"
current_target_class = None
current_confidence = CONFIDENCE_THRESHOLD
detection_mode = "auto"
detections_log = []
connection_error = None

# --- Load YOLOv8 Model ---
try:
    model = YOLO(MODEL_PATH)
    if USE_GPU:
        model.to('cuda')
        print("Using CUDA for acceleration")
    KNOWN_CLASSES = model.names
    print(f"YOLOv8 Model ({MODEL_PATH}) loaded successfully.")
    print(f"Available classes: {len(KNOWN_CLASSES)}")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()

def validate_url(url):
    pattern = r'^(http|https):\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}(\/.*)?$'
    return bool(re.match(pattern, url))

def get_camera_stream(source):
    global connection_error
    cap = None
    source = str(source).strip()
    if source in ("0", "local"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            connection_error = "Failed to open local webcam. Ensure it is connected and not in use."
            raise ConnectionError(connection_error)
    else:
        if not validate_url(source):
            connection_error = "Invalid URL. Use http://YOUR_IP:PORT/video."
            raise ConnectionError(connection_error)
        if '?' not in source:
            source += '?t=' + str(int(time.time()))
        for attempt in range(RETRY_ATTEMPTS):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            for _ in range(10):
                if cap and cap.isOpened():
                    connection_error = None
                    break
                time.sleep(0.15)
            if cap and cap.isOpened():
                break
            else:
                connection_error = f"Attempt {attempt+1}/{RETRY_ATTEMPTS}: Could not connect to {source}."
                print(connection_error)
                cap.release() if cap else None
                cap = None
                time.sleep(RETRY_DELAY * (2 ** attempt))
        if not cap or not cap.isOpened():
            connection_error = f"Failed to connect to {source} after {RETRY_ATTEMPTS} tries."
            raise ConnectionError(connection_error)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def generate_frames():
    global current_camera_source, detection_mode, current_target_class, current_confidence, detections_log, connection_error
    cap = None
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    frame_skip = 0

    while True:
        with camera_lock:
            if cap is None or not cap.isOpened():
                try:
                    cap = get_camera_stream(current_camera_source)
                except Exception as e:
                    print(f"Camera reconnect error: {e}")
                    connection_error = str(e)
                    time.sleep(1)
                    continue

        success, frame = cap.read()
        if not success:
            print("Failed to read frame, reconnecting...")
            connection_error = "Failed to read stream. Reconnecting..."
            if cap:
                cap.release()
            cap = None
            continue

        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        annotated_frame = frame.copy()
        if frame_skip % FRAME_SKIP == 0:
            try:
                predict_kwargs = {
                    'source': frame,
                    'conf': current_confidence,
                    'imgsz': IMG_SIZE,
                    'half': HALF_PRECISION and USE_GPU,
                    'verbose': False,
                    'device': 'cuda' if USE_GPU else 'cpu'
                }
                if detection_mode == "search" and current_target_class:
                    class_id = next((k for k, v in KNOWN_CLASSES.items()
                                   if v.lower() == current_target_class.lower()), None)
                    if class_id is not None:
                        predict_kwargs['classes'] = [class_id]
                results = model.predict(**predict_kwargs)
                annotated_frame = results[0].plot()
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        xyxy = box.xyxy.cpu().numpy()[0]
                        det = {
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'class': KNOWN_CLASSES[cls_id],
                            'confidence': round(conf, 3),
                            'bbox': xyxy.tolist()
                        }
                        detections_log.append(det)
                        if len(detections_log) > 400:
                            detections_log.pop(0)
            except Exception as e:
                print(f"Detection error: {e}")
                connection_error = f"Detection error: {e}"
        frame_skip += 1

        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    if cap:
        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global detection_mode, current_target_class
    data = request.json
    detection_mode = data.get('mode', 'auto')
    current_target_class = data.get('target', None)
    print(f"[i] Mode: {detection_mode}, Target: {current_target_class}")
    return jsonify({"status": "success", "mode": detection_mode, "target": current_target_class})

@app.route('/api/classes')
def get_classes():
    return jsonify({"classes": list(KNOWN_CLASSES.values())})

@app.route('/api/detections')
def get_detections():
    return jsonify(detections_log[-50:])

@app.route('/')
def index():
    return render_template('index.html', classes=list(KNOWN_CLASSES.values()))

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting YOLOv8 Detection Server")
    print("=" * 60)
    print(f"📊 Model: {MODEL_PATH}")
    print(f"🔢 Available Classes: {len(KNOWN_CLASSES)}")
    print(f"🖥️  CUDA Available: {USE_GPU}")
    print(f"🌐 Server: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
