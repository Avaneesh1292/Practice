from ultralytics import YOLO
import cv2
import numpy as np
import random

# --- 1. Generate a test image on the fly ---
IMG_SIZE = 640
COLOR_YELLOW = (0, 255, 255)
COLORS_PALETTE = [(0, 0, 255), (0, 255, 0), COLOR_YELLOW]

test_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for _ in range(10):
    color = random.choice(COLORS_PALETTE)
    size = random.randint(50, 150)
    x1 = random.randint(0, IMG_SIZE - size - 1)
    y1 = random.randint(0, IMG_SIZE - size - 1)
    cv2.rectangle(test_img, (x1, y1), (x1+size, y1+size), color, -1)

cv2.imwrite("test_prediction.jpg", test_img)
print("Created test_prediction.jpg")

# --- 2. Load trained model and predict ---
# NOTE: Update this path if your training results are saved elsewhere
model_path = 'squares_training_results/yellow_detector/weights/best.pt'

try:
    model = YOLO(model_path)
except FileNotFoundError:
    print(f"Error: Could not find trained model at {model_path}")
    print("Please verify the path based on your training script output.")
    exit()

# Run prediction
results = model.predict("test_prediction.jpg", conf=0.5)

# --- 3. Show results ---
# results[0].plot() returns the image with bounding boxes drawn onto it
annotated_frame = results[0].plot()

# Display using OpenCV window
cv2.imshow("YOLO Detection Results", annotated_frame)
print("Press any key on the image window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()