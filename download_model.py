from ultralytics import YOLO

# Download and save the YOLOv8n model
YOLO('yolov8n.pt').save('../model/yolov8n.pt')
