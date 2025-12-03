from ultralytics import YOLO
import os

def train_model():
    if not os.path.exists("squares_data.yaml"):
        print("Error: squares_data.yaml not found. Run the generation script first.")
        return

    # 1. Load a model (YOLOv8 nano weights)
    model = YOLO('yolov8n.pt')  

    # 2. Train the model
    print("Starting training...")
    results = model.train(
        data='squares_data.yaml',
        epochs=10, 
        imgsz=640,
        plots=True,
        project='squares_training_results',
        name='yellow_detector'
    )
    
    print("Training complete.")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_model()