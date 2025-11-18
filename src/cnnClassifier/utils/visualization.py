import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import json


class MetricsVisualizer:
    """Utility class for creating training and evaluation visualizations"""
    
    def __init__(self, save_dir):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def plot_training_history(self, history, save_name='training_history.png'):
        """
        Plot training and validation metrics over epochs
        
        Args:
            history: Training history object from model.fit()
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s', linewidth=2)
        axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Train Loss', marker='o', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training history plot saved to: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(self, cm, class_names, save_name='confusion_matrix.png'):
        """
        Plot confusion matrix heatmap
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            save_name: Filename to save plot
        """
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix plot saved to: {save_path}")
        return str(save_path)
    
    def plot_roc_curve(self, y_true, y_pred_proba, class_names, save_name='roc_curve.png'):
        """
        Plot ROC curve for binary or multiclass classification
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            class_names: List of class names
            save_name: Filename to save plot
        """
        plt.figure(figsize=(8, 6))
        
        n_classes = len(class_names)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            
        else:
            # Multiclass - One-vs-Rest
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC curve plot saved to: {save_path}")
        return str(save_path)
    
    def plot_class_distribution(self, class_counts, class_names, save_name='class_distribution.png'):
        """
        Plot class distribution bar chart
        
        Args:
            class_counts: Dictionary or list of counts per class
            class_names: List of class names
            save_name: Filename to save plot
        """
        plt.figure(figsize=(10, 6))
        
        if isinstance(class_counts, dict):
            counts = [class_counts[name] for name in class_names]
        else:
            counts = class_counts
        
        bars = plt.bar(class_names, counts, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(class_names)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.title('Class Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Class distribution plot saved to: {save_path}")
        return str(save_path)
    
    def create_metrics_summary(self, metrics_dict, save_name='metrics_summary.png'):
        """
        Create a visual summary of key metrics
        
        Args:
            metrics_dict: Dictionary containing metrics
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Prepare text
        title_text = "Model Performance Summary"
        
        metrics_text = ""
        if 'test_accuracy' in metrics_dict:
            metrics_text += f"Test Accuracy: {metrics_dict['test_accuracy']:.4f} ({metrics_dict['test_accuracy']*100:.2f}%)\n"
        if 'test_loss' in metrics_dict:
            metrics_text += f"Test Loss: {metrics_dict['test_loss']:.4f}\n"
        if 'roc_auc' in metrics_dict:
            metrics_text += f"ROC-AUC Score: {metrics_dict['roc_auc']:.4f}\n"
        
        # Add class-wise metrics if available
        if 'classification_report' in metrics_dict:
            report = metrics_dict['classification_report']
            metrics_text += "\nPer-Class Performance:\n"
            for class_name, class_metrics in report.items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    metrics_text += f"\n{class_name.upper()}:\n"
                    metrics_text += f"  Precision: {class_metrics['precision']:.4f}\n"
                    metrics_text += f"  Recall: {class_metrics['recall']:.4f}\n"
                    metrics_text += f"  F1-Score: {class_metrics['f1-score']:.4f}\n"
        
        # Display text
        plt.text(0.5, 0.95, title_text, ha='center', va='top', 
                fontsize=18, fontweight='bold', transform=ax.transAxes)
        plt.text(0.1, 0.85, metrics_text, ha='left', va='top', 
                fontsize=12, family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Metrics summary saved to: {save_path}")
        return str(save_path)
