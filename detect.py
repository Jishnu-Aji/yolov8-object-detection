from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection on bus.jpg
results = model(
    source="bus.jpg",
    save=True,
    conf=0.25
)

print("Detection completed. Check runs/detect/predict/")
