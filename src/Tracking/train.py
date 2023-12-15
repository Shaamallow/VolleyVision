from ultralytics import YOLO
import os

cwd = os.getcwd()

model = YOLO("models/yolov8/yolov8n.pt")
results = model.train("data/Tracking/dataset/data.yaml", epochs=100)
