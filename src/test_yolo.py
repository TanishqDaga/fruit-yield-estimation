from ultralytics import YOLO

print("Script started...")

model = YOLO("yolov8n.pt")
print("Model loaded")

results = model("https://ultralytics.com/images/bus.jpg", save=True)

print("Prediction done")
print(results)