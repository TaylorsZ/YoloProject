import torch
from ultralytics import YOLO

print("MPS is available:", torch.backends.mps.is_available())
# Check if MPS is available and set the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the model
model = YOLO('yolov8n.pt').to(device)

# Training setup
model.train(task ="detect",data='datasets/bucket.yaml', epochs=5000, imgsz=640)