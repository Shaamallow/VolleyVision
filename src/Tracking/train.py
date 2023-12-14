from ultralytics import YOLO
import os

cwd = os.getcwd()

model = YOLO(cwd + "/models/yolov8/yolov8n.pt")
results = model.train(data=cwd + "/data/Tracking/dataset/data.yaml", epochs=100)
