from ultralytics import YOLO
import os

# Load model
model = YOLO("runs/detect/train4/weights/best.pt")

# Folder with test images
image_folder = "test_images"

# 🔴 MANUAL ACTUAL COUNTS (edit this)
actual_counts = {
    "img_1.jpg": 8,
    "img_2.jpg": 16,
    "img_3.jpg": 4
}

total_error = 0
num_images = len(actual_counts)

for img_name, actual in actual_counts.items():
    path = os.path.join(image_folder, img_name)

    results = model(path, conf=0.5, iou=0.4)
    predicted = len(results[0].boxes)

    error = abs(predicted - actual)
    total_error += error

    print(f"{img_name}: Predicted={predicted}, Actual={actual}, Error={error}")

# Calculate MAE
mae = total_error / num_images

print("\nFinal MAE:", mae)