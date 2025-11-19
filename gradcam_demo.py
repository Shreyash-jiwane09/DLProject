"""
Demo script to generate Grad-CAM visualizations for sample predictions.
This helps in understanding what regions of the image the model focuses on.
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from cnnClassifier.utils.gradcam import GradCAM

def generate_gradcam_samples():
    """Generate Grad-CAM visualizations for sample images"""
    
    # Check if model exists
    model_path = "artifacts/training/model.h5"
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the model first.")
        return
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    # Class names
    class_names = ['adenocarcinoma', 'normal']
    
    # Create output directory
    output_dir = Path("artifacts/gradcam_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample images directory
    test_data_dir = Path("artifacts/data_ingestion/test")
    
    if not test_data_dir.exists():
        print(f"❌ Test data not found at {test_data_dir}")
        return
    
    print(f"\n{'='*60}")
    print("Generating Grad-CAM Visualizations")
    print(f"{'='*60}\n")
    
    # Process samples from each class
    samples_per_class = 3
    
    for class_name in class_names:
        class_dir = test_data_dir / class_name
        if not class_dir.exists():
            print(f"⚠️  Class directory not found: {class_dir}")
            continue
        
        # Get sample images
        images = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        if not images:
            print(f"⚠️  No images found in {class_dir}")
            continue
        
        print(f"\nProcessing {class_name} samples...")
        
        for i, img_path in enumerate(images[:samples_per_class]):
            save_path = output_dir / f"{class_name}_sample_{i+1}_gradcam.png"
            
            try:
                result = gradcam.generate_visualization(
                    image_path=str(img_path),
                    save_path=str(save_path),
                    class_names=class_names
                )
                
                print(f"  ✅ {img_path.name}")
                print(f"     Prediction: {result['class_name']}")
                print(f"     Confidence: {result['confidence']:.4f}")
                print(f"     Saved to: {save_path}")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"✅ Grad-CAM visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_gradcam_samples()
