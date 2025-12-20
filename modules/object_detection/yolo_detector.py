# modules/object_detection/yolo_detector.py

import os
import cv2
from ultralytics import YOLO
from core.logger import logger

MODEL_PATH = "models/yolov8/yolov8n.pt"
ROI_DIR = "data/processed/roi"
OUTPUT_DIR = "data/processed/detections"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

def run_object_detection():
    logger.info("Starting YOLOv8 object detection")

    for img_name in os.listdir(ROI_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(ROI_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model(image)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                if label in ["person", "car", "truck", "animal"] and conf > 0.5:
                    logger.info(f"Detected {label} ({conf:.2f}) in {img_name}")

                    save_path = os.path.join(
                        OUTPUT_DIR, f"{label}_{img_name}"
                    )
                    cv2.imwrite(save_path, image)
