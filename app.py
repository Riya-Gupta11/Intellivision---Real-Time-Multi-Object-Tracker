from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import os
import time
from datetime import datetime
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

# Config
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.45
SNAPSHOT_DIR = "snapshots"
CSV_LOG = "detections_log.csv"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Load model
print("[+] Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("[+] Model loaded successfully.")

known_classes = list(model.names.values())

# Global variables
camera = None
detection_mode = "auto"  # "auto" or "search"
target_object = None
log_data = []

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        time.sleep(1.0)
    return camera

def generate_frames():
    global detection_mode, target_object, log_data
    
    cap = get_camera()
    last_snapshot = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        
        # Resize frame
        base_height = int(frame.shape[0] * 640 / frame.shape[1])
        frame = cv2.resize(frame, (640, base_height))
        annotated = frame.copy()
        found_any = False
        
        # Auto-detect mode
        if detection_mode == "auto":
            results = model.predict(frame, conf=CONFIDENCE, verbose=False)
            counts = defaultdict(int)
            
            if results and hasattr(results[0], "boxes") and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if conf < CONFIDENCE:
                        continue
                    
                    label = model.names[cls_id]
                    counts[label] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw detection
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(annotated, f"{label} ({conf:.2f})", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display counts
                y0 = 30
                for cls, cnt in counts.items():
                    cv2.putText(annotated, f"{cls}: {cnt}", (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y0 += 30
        
        # Search specific mode
        elif detection_mode == "search" and target_object:
            candidates = [k for k, v in model.names.items() if target_object.lower() in v.lower()]
            
            if not candidates:
                cv2.putText(annotated, f"'{target_object}' not in YOLO classes", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                results = model.predict(frame, conf=CONFIDENCE, classes=candidates, verbose=False)
                
                if results and hasattr(results[0], "boxes") and results[0].boxes:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if conf < CONFIDENCE:
                            continue
                        
                        label = model.names[cls_id]
                        found_any = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw detection
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(annotated, f"{label} ({conf:.2f})", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Log
                        log_data.append({
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "object": label,
                            "confidence": round(conf, 2)
                        })
                        
                        # Snapshot every 7 seconds
                        now = time.time()
                        if now - last_snapshot > 7:
                            snap_name = os.path.join(SNAPSHOT_DIR, f"{target_object}_{int(now)}.jpg")
                            cv2.imwrite(snap_name, annotated)
                            print(f"[+] Snapshot saved: {snap_name}")
                            last_snapshot = now
                
                # Status overlay
                status = f"{target_object}: {'FOUND ✅' if found_any else 'SEARCHING 🔍'}"
                color = (0, 255, 0) if found_any else (0, 0, 255)
                cv2.putText(annotated, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Canvas display (aspect ratio preserved, centered)
        CANVAS_W, CANVAS_H = 900, 600
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        h, w = annotated.shape[:2]
        scale = min(CANVAS_W/w, CANVAS_H/h)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(annotated, (new_w, new_h))
        x_off = (CANVAS_W - new_w) // 2
        y_off = (CANVAS_H - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', canvas)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', classes=known_classes)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global detection_mode, target_object
    data = request.json
    detection_mode = data.get('mode', 'auto')
    target_object = data.get('target', None)
    print(f"[i] Mode: {detection_mode}, Target: {target_object}")
    return jsonify({"status": "success", "mode": detection_mode, "target": target_object})

@app.route('/get_logs')
def get_logs():
    if log_data:
        df = pd.DataFrame(log_data).tail(10)
        return df.to_html(index=False, classes='table table-striped')
    return "<p>No detection logs yet</p>"

@app.route('/save_logs')
def save_logs():
    if log_data:
        df = pd.DataFrame(log_data)
        df.to_csv(CSV_LOG, index=False)
        return jsonify({"status": "success", "message": f"Logs saved to {CSV_LOG}"})
    return jsonify({"status": "error", "message": "No logs to save"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)