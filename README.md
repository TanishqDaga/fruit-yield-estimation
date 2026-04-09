# Fruit Yield Estimation using Computer Vision (YOLO + DIP)

## Overview

This project implements an automated system to detect, count, and estimate the yield of tomatoes from plant images using a combination of deep learning and digital image processing techniques. The system processes input images, identifies fruits, computes their count, and estimates total yield in kilograms.

---

## Objectives

* Detect tomatoes in plant images using an object detection model
* Count the number of fruits accurately
* Estimate total yield based on detected fruit size
* Handle partially visible (occluded) fruits
* Improve detection robustness using image processing techniques
* Evaluate system performance using quantitative metrics

---

## Methodology

### Pipeline

```
Image Input
→ Image Enhancement (DIP)
→ YOLO Detection
→ Filtering
→ Counting
→ Yield Estimation
→ Evaluation
```

---

## Technologies Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy

---

## Model Training

The model is trained using a labeled dataset of tomato images.

### Steps:

1. Images are annotated with bounding boxes around each tomato
2. Dataset is split into training, validation, and test sets
3. YOLOv8 is trained on this dataset for multiple epochs
4. The model learns:

   * Object localization (bounding boxes)
   * Classification (tomato detection)

### Training Outputs:

* Loss values (box loss, classification loss, DFL loss)
* Evaluation metrics (precision, recall, mAP)
* Final trained weights (`best.pt`)

---

## Digital Image Processing Enhancements

Before detection, image preprocessing is applied to improve accuracy:

* Contrast enhancement (CLAHE)
* Noise reduction (Gaussian blur)
* Color-based segmentation (HSV filtering for red regions)
* Morphological operations to remove noise

These steps improve detection performance in real-world conditions such as poor lighting or cluttered backgrounds.

---

## Detection and Counting

The trained YOLO model predicts bounding boxes around tomatoes.
Each valid bounding box corresponds to one detected fruit.

```
Count = Number of valid detections
```

Post-processing is applied to remove false detections using color filtering and confidence thresholds.

---

## Yield Estimation

Yield is estimated using the normalized area of each detected fruit.

```
Normalized Area = Bounding Box Area / Image Area
Weight ≈ Normalized Area × Scaling Factor
Total Yield = Sum of all estimated weights
```

The final yield is expressed in kilograms.

---

## Adaptive Confidence Threshold

Instead of using a fixed confidence threshold, the system adjusts it dynamically based on image quality.

Image quality is estimated using:

* Sharpness (Laplacian variance)
* Brightness
* Contrast

This improves detection performance across different lighting and clarity conditions.

---

## Evaluation Metrics

### Mean Absolute Error (MAE)

MAE measures the average difference between predicted and actual fruit counts.

```
MAE = (1/N) × Σ |Predicted − Actual|
```

* Lower MAE indicates better counting accuracy
* Example: If predicted = 18 and actual = 20, error = 2

---

### Occlusion Recall

Occlusion recall evaluates how well the model detects partially hidden fruits.

```
Recall = TP / (TP + FN)
```

Where:

* TP = correctly detected occluded fruits
* FN = occluded fruits that were missed

This metric is important because real-world farm images often contain overlapping or partially visible fruits.

---

## Project Structure

```
fruit-yield-estimation/
│
├── data/
│   ├── raw_images/
│   └── annotated/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── mae.py
│   └── occlusion_recall.py
│
├── test_images/
├── runs/
└── README.md
```

---

## Installation

```
pip install ultralytics opencv-python numpy matplotlib
```

---

## Usage

### Train the Model

```
python src/train.py
```

### Run Prediction

```
python src/predict.py
```

### Calculate MAE

```
python src/mae.py
```

### Compute Occlusion Recall

```
python src/occlusion_recall.py
```

---

## Sample Output

```
Tomato count: 12
Estimated Yield: 1.85 kg
MAE: 1.6
Occlusion Recall: 0.73
```

---

## Key Features

* Combines deep learning with image processing
* Adaptive detection based on image quality
* Handles partial occlusion
* Provides both count and yield estimation
* Includes evaluation metrics for performance analysis

---

## Limitations

* Performance depends on dataset size and quality
* Occlusion handling is approximate without explicit labeling
* Yield estimation is based on heuristics
* No depth information for true size estimation

---

## Future Improvements

* Depth estimation for accurate size measurement
* Multi-view image fusion
* Mobile or web deployment
* Integration with drone-based data collection
* Extension to other crops

---

## Applications

* Smart farming systems
* Precision agriculture
* Crop monitoring
* Yield forecasting

---

## Author

Tanishq Daga
VIT Vellore

---
