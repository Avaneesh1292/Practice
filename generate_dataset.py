import cv2
import numpy as np
import os
import random
import shutil

# === Configuration ===
IMG_SIZE = 640
NUM_TRAIN = 200
NUM_VAL = 50
MAX_SQUARES_PER_IMG = 15
MIN_SQUARE_SIZE = 50
MAX_SQUARE_SIZE = 150

# Base directory for the dataset
DATASET_DIR = 'squares_dataset'

# Define colors in BGR format (OpenCV standard)
COLOR_YELLOW = (0, 255, 255)  # BGR for Yellow
COLORS_PALETTE = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    COLOR_YELLOW,   # Target color
    (255, 255, 255) # White
]

# YOLO Class ID for yellow square
YELLOW_CLASS_ID = 0

def setup_directories():
    """Creates the standard YOLO directory structure."""
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        print(f"Cleaning existing directory: {DATASET_DIR}")
    
    os.makedirs(os.path.join(DATASET_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels', 'val'), exist_ok=True)
    print("Directory structure created.")

def create_image_and_labels(filename_prefix, output_dir_imgs, output_dir_labels):
    """Generates a single image and its corresponding YOLO label file."""
    # Create a black background image
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    labels = []
    
    num_squares = random.randint(3, MAX_SQUARES_PER_IMG)
    
    for _ in range(num_squares):
        # Pick random properties
        color = random.choice(COLORS_PALETTE)
        size = random.randint(MIN_SQUARE_SIZE, MAX_SQUARE_SIZE)
        
        # Ensure square stays within image bounds
        x1 = random.randint(0, IMG_SIZE - size - 1)
        y1 = random.randint(0, IMG_SIZE - size - 1)
        x2 = x1 + size
        y2 = y1 + size
        
        # Draw filled rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # === YOLO Annotation Logic ===
        if color == COLOR_YELLOW:
            # YOLO format: class_id x_center y_center width height (normalized 0.0-1.0)
            x_center = (x1 + x2) / 2.0 / IMG_SIZE
            y_center = (y1 + y2) / 2.0 / IMG_SIZE
            width_norm = size / IMG_SIZE
            height_norm = size / IMG_SIZE
            
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width_norm = max(0.0, min(1.0, width_norm))
            height_norm = max(0.0, min(1.0, height_norm))

            labels.append(f"{YELLOW_CLASS_ID} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

    # Save Image
    img_path = os.path.join(output_dir_imgs, f"{filename_prefix}.jpg")
    cv2.imwrite(img_path, img)
    
    # Save Label Text File
    label_path = os.path.join(output_dir_labels, f"{filename_prefix}.txt")
    with open(label_path, "w") as f:
        f.write("\n".join(labels))

def create_yaml_file():
    """Creates the data.yaml file required by YOLO training."""
    yaml_content = f"""
path: {os.path.abspath(DATASET_DIR)} # dataset root dir
train: images/train  # train images (relative to 'path') 
val: images/val  # val images (relative to 'path') 

# Classes
names:
  0: yellow_square
"""
    with open("squares_data.yaml", "w") as f:
        f.write(yaml_content)
    print("Created squares_data.yaml configuration file.")

def main():
    setup_directories()
    
    print(f"Generating {NUM_TRAIN} training images...")
    for i in range(NUM_TRAIN):
        create_image_and_labels(f"train_{i}", 
                                os.path.join(DATASET_DIR, 'images', 'train'),
                                os.path.join(DATASET_DIR, 'labels', 'train'))
        
    print(f"Generating {NUM_VAL} validation images...")
    for i in range(NUM_VAL):
        create_image_and_labels(f"val_{i}", 
                                os.path.join(DATASET_DIR, 'images', 'val'),
                                os.path.join(DATASET_DIR, 'labels', 'val'))
                                
    create_yaml_file()
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()