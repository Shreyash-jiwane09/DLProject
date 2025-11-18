import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    for visualizing which regions of an image the model focuses on for predictions.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM
        
        Args:
            model: Trained keras model
            layer_name: Name of the convolutional layer to visualize. 
                       If None, uses the last conv layer
        """
        self.model = model
        self.layer_name = layer_name
        
        if self.layer_name is None:
            # Find the last convolutional layer
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:
                    self.layer_name = layer.name
                    break
        
        # Create gradient model
        self.grad_model = keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            image: Input image (preprocessed)
            class_idx: Target class index. If None, uses predicted class
            eps: Small value to avoid division by zero
            
        Returns:
            heatmap: Numpy array representing the heatmap
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by their importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original image (numpy array)
            alpha: Transparency for overlay
            colormap: OpenCV colormap
            
        Returns:
            superimposed_img: Image with heatmap overlay
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert to RGB (from BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is in correct format
        if original_image.max() <= 1.0:
            original_image = np.uint8(255 * original_image)
        
        # Superimpose heatmap on image
        superimposed_img = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed_img
    
    def generate_visualization(self, image_path, save_path, class_names=None):
        """
        Generate and save complete Grad-CAM visualization
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
            class_names: List of class names for labeling
            
        Returns:
            dict: Prediction results with confidence
        """
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        original_img = img_array.copy()
        
        # Preprocess for model
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = predictions[0][pred_class]
        
        # Compute heatmap
        heatmap = self.compute_heatmap(img_array, class_idx=pred_class)
        
        # Create overlay
        superimposed_img = self.overlay_heatmap(heatmap, original_img.astype(np.uint8))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img.astype(np.uint8))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(superimposed_img)
        pred_label = class_names[pred_class] if class_names else f"Class {pred_class}"
        axes[2].set_title(f'Prediction: {pred_label}\nConfidence: {confidence:.2%}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'predicted_class': int(pred_class),
            'confidence': float(confidence),
            'class_name': pred_label,
            'all_predictions': predictions[0].tolist()
        }
