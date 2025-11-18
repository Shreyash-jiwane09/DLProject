import numpy as np


import os
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = os.path.join("artifacts", "training", "model.h5")
        self.class_names = ['adenocarcinoma', 'normal']
        
    def validate_image(self):
        """
        Validate input image format and size
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not os.path.exists(self.filename):
                logger.error(f"Image file not found: {self.filename}")
                return False
            
            # Check file size (limit to 10MB)
            file_size = os.path.getsize(self.filename) / (1024 * 1024)  # Convert to MB
            if file_size > 10:
                logger.error(f"Image file too large: {file_size:.2f}MB. Maximum allowed: 10MB")
                return False
            
            # Try to open image
            with Image.open(self.filename) as img:
                # Check if image format is valid
                valid_formats = ['JPEG', 'JPG', 'PNG', 'BMP']
                if img.format not in valid_formats:
                    logger.error(f"Invalid image format: {img.format}. Allowed: {valid_formats}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False
    
    def predict(self):
        """
        Predict the class of input image with confidence scores
        
        Returns:
            list: Prediction results with confidence scores
        """
        try:
            # Validate image first
            if not self.validate_image():
                return [{"error": "Invalid image file"}]
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return [{"error": "Model not found. Please train the model first."}]
            
            # Load model
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing import image
            model = load_model(self.model_path)
            
            # Load and preprocess image
            imagename = self.filename
            test_image = image.load_img(imagename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0) / 255.0
            
            # Get predictions
            predictions = model.predict(test_image, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class])
            
            # Get all class probabilities
            class_probabilities = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
            
            prediction_label = self.class_names[predicted_class]
            
            logger.info(f"Prediction: {prediction_label}, Confidence: {confidence:.4f}")
            
            return [{
                "prediction": prediction_label,
                "confidence": f"{confidence:.4f}",
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "all_probabilities": class_probabilities
            }]
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return [{"error": f"Prediction failed: {str(e)}"}]


