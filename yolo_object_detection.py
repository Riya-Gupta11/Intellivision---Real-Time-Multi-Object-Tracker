from ultralytics import YOLO
import cv2
import os
import time
import pandas as pd
from datetime import datetime
from collections import defaultdict
import numpy as np

# ---------------- CONFIG ----------------
CONFIDENCE = 0.25
FRAME_WIDTH = 640
OUTPUT_VIDEO = "output.avi"
CSV_LOG = "detections_log.csv"
SNAPSHOT_DIR = "snapshots"
SNAPSHOT_INTERVAL = 7  # seconds

WINDOW_WIDTH = 900     # Display window width
WINDOW_HEIGHT = 600    # Display window height
# ----------------------------------------

print("[+] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("[+] Model loaded successfully.")

cap = cv2.VideoCapture(0)
time.sleep(1.0)

if not cap.isOpened():
    print("[!] Webcam not accessible.")
    exit()

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
log_data = []
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = None
last_snapshot = 0

print("[+] Starting real-time detection. Press 'q' to quit.")

cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Object Detection", WINDOW_WIDTH, WINDOW_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detection input size (keep aspect ratio)
    base_height = int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])
    frame = cv2.resize(frame, (FRAME_WIDTH, base_height))

    # YOLOv8 inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # Object counting and log
    counts = defaultdict(int)
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if conf < CONFIDENCE:
            continue
        label = model.names[cls_id]
        counts[label] += 1
        log_data.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object": label,
            "confidence": round(conf, 2)
        })

    # Display counts
    y0 = 25
    for cls, cnt in counts.items():
        cv2.putText(annotated_frame, f"{cls}: {cnt}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y0 += 25

    # Video writer
    if out is None:
        h, w = annotated_frame.shape[:2]
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (w, h))
    out.write(annotated_frame)

    # ------------- Aspect-ratio maintain, center paste! -------------
    # Create blank canvas
    canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    h, w = annotated_frame.shape[:2]
    scale = min(WINDOW_WIDTH/w, WINDOW_HEIGHT/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized_frame = cv2.resize(annotated_frame, (new_w, new_h))
    x_offset = (WINDOW_WIDTH - new_w) // 2
    y_offset = (WINDOW_HEIGHT - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
    cv2.imshow("YOLOv8 Object Detection", canvas)
    # ---------------------------------------------------------------

    # Save snapshots every interval (original size)
    if counts and (time.time() - last_snapshot > SNAPSHOT_INTERVAL):
        snap_name = os.path.join(SNAPSHOT_DIR, f"snap_{int(time.time())}.jpg")
        cv2.imwrite(snap_name, annotated_frame)
        print(f"[+] Snapshot saved: {snap_name}")
        last_snapshot = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[+] Exiting...")
cap.release()
out.release()
cv2.destroyAllWindows()

if log_data:
    pd.DataFrame(log_data).to_csv(CSV_LOG, index=False)
    print(f"[+] Log saved: {CSV_LOG}")

print(f"[+] Video saved: {OUTPUT_VIDEO}")
print(f"[+] Snapshots in: {SNAPSHOT_DIR}")
