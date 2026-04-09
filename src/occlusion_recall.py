from ultralytics import YOLO
import os
import numpy as np

# Load trained model
model = YOLO("runs/detect/train4/weights/best.pt")

image_folder = "test_images"

# MANUAL COUNTS
occlusion_data = {
    "img_1.jpg": 5,
    "img_2.jpg": 7,
    "img_3.jpg": 1
}

total_TP = 0
total_FN = 0

for img_name, actual_occluded in occlusion_data.items():
    path = os.path.join(image_folder, img_name)

    results = model(path, conf=0.5, iou=0.4)

    boxes = results[0].boxes

    if len(boxes) == 0:
        print(f"{img_name}: No detections")
        total_FN += actual_occluded
        continue

    areas = []
    confs = []

    # Collect stats
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        area = float((x2 - x1) * (y2 - y1))
        conf = float(box.conf[0])

        areas.append(area)
        confs.append(conf)

    avg_area = np.mean(areas)
    avg_conf = np.mean(confs)

    detected_occluded = 0

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        area = float((x2 - x1) * (y2 - y1))
        conf = float(box.conf[0])

        # OCCLUSION LOGIC
        if (
            area < 0.7 * avg_area and   # smaller than normal
            conf < avg_conf + 0.05      # not very confident
        ):
            detected_occluded += 1

    TP = min(detected_occluded, actual_occluded)
    FN = actual_occluded - TP

    total_TP += TP
    total_FN += FN

    print(f"{img_name}: Detected={detected_occluded}, Actual={actual_occluded}")

# Final recall
recall = total_TP / (total_TP + total_FN + 1e-6)

print("\n Occlusion Recall:", round(recall, 3))