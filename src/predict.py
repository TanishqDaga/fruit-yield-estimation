from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train4/weights/best.pt")

# Load image
image_path = "test.jpg"   # put your test image here

results = model("test.jpg", conf=0.4, iou=0.4)

# Count tomatoes
count = len(results[0].boxes)

print("Tomato count:", count)

# Show result image
annotated = results[0].plot(
    labels=False,
    boxes=True,
    conf=False,
    line_width=1
)
cv2.imwrite("output.jpg", annotated)