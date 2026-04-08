from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("runs/detect/train4/weights/best.pt")

# Load image
image_path = "test.jpg"
image = cv2.imread(image_path)
h, w, _ = image.shape

# Run detection
results = model(image_path, conf=0.5, iou=0.4)

boxes = results[0].boxes

count = 0
total_weight = 0

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    conf = float(box.conf[0])

    # Convert to float
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    # Compute box area
    box_area = (x2 - x1) * (y2 - y1)

    # Normalize area (relative to image size)
    norm_area = box_area / (w * h)

    # Clamp extreme values (avoid distance issue)
    norm_area = max(0.0005, min(norm_area, 0.02))

    # Estimate weight using normalized area + confidence
    weight = (norm_area * 5000) * (0.5 + conf)

    # Clamp realistic tomato weight (grams)
    weight = max(40, min(weight, 150))

    total_weight += weight
    count += 1

# Final yield
yield_kg = total_weight / 1000

print("Tomato count:", count)
print("Estimated Yield:", round(yield_kg, 2), "kg")

# Clean visualization
annotated = results[0].plot(labels=False, line_width=1)
cv2.imwrite("output.jpg", annotated)