#!/usr/bin/env python3

from ultralytics import YOLO
import torch

def main():
    
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠️ GPU not available, training on CPU")
        device = 'cpu'
    
    # Note: 'yolo11n.pt' is not a standard model. Assuming 'yolov8n.pt'.
    print("📥 Loading YOLOv8n pretrained model...")
    model = YOLO('yolo12m.pt') 
    
    print("🎯 Starting training...")
    results = model.train(
        # Essential parameters
        data='data/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        project='runs/train',
        name='clean_training',
        
        # Core data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        
        # Optimization parameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Output control
        save=True,
        save_period=-1, # Saves only best.pt
        plots=False,    # Disables plot generation (e.g., results.png, confusion_matrix.png)
        
        # Other settings
        patience=50,
        verbose=True,
    )
    
    print("🎉 Training complete!")
    print(f"📁 Results saved to: {results.save_dir}")
    print(f"🏆 Best model at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()