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