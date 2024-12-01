# scripts/download_model.py
import os
from ultralytics import YOLO

# Check if the model already exists in the models directory
model_path = '../models/yolov8m.pt'
if not os.path.exists(model_path):
    print("Downloading YOLOv8 model...")
    model = YOLO('yolov8m.pt')
    # Save model to the specified path
    model.save(model_path)
    print("Model downloaded and saved in models/yolov8m.pt.")
else:
    print("YOLOv8 model already exists.")
