from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load model
model = YOLO("runs/detect/train4/weights/best.pt")

image_path = "test2.jpg"

if not os.path.exists(image_path):
    print("Image not found!")
    exit()

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image")
    exit()

h, w, _ = image.shape



#CLARITY MEASURE


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Sharpness
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

# 2. Brightness
brightness = np.mean(gray)

# 3. Contrast
contrast = gray.std()

print("Sharpness:", laplacian_var)
print("Brightness:", brightness)
print("Contrast:", contrast)


# SMART DECISION


if laplacian_var < 50 or contrast < 40 or brightness < 90:
    conf = 0.18  # poor image
elif laplacian_var < 150 or contrast <150 or brightness <150:
    conf = 0.15  # medium
else:
    conf = 0.25   # not too strict

print("Adaptive confidence:", conf)

# STEP 3: YOLO DETECTION


results = model(image, conf=conf, iou=0.2)

boxes = results[0].boxes

count = len(boxes)

print("Tomato count:", count)


#YIELD ESTIMATION


total_weight = 0.0

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]

    # convert tensor → float
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    area = (x2 - x1) * (y2 - y1)

    # normalized area
    norm_area = area / (h * w)

    # weight estimation
    weight = norm_area * 5000

    # clamp realistic range
    weight = max(40, min(weight, 150))

    total_weight += weight

yield_kg = total_weight / 1000

print("Estimated Yield:", round(yield_kg, 2), "kg")


#OUTPUT IMAGE


annotated = results[0].plot(labels=False, line_width=1)
cv2.imwrite("output2.jpg", annotated)

print("Output saved as output2.jpg")