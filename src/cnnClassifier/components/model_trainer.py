import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from zipfile import ZipFile
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.visualization import MetricsVisualizer


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.visualizer = MetricsVisualizer(save_dir=self.config.root_dir)

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.summary()

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1. / 255)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Train generator
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.train_data,
            shuffle=True,
            **dataflow_kwargs
        )

        # Validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.val_data,
            shuffle=False,
            **dataflow_kwargs
        )

    def setup_test_generator(self):
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.test_data,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def save_training_logs(self):
        with open(self.config.training_log_path, "w") as f:
            f.write("Training History Log\n")
            f.write("="*80 + "\n\n")
            
            actual_epochs = len(self.history.history['loss'])
            for epoch in range(actual_epochs):
                f.write(
                    f"Epoch {epoch + 1}/{actual_epochs}:\n"
                    f"  Loss: {self.history.history['loss'][epoch]:.4f}, "
                    f"Accuracy: {self.history.history['accuracy'][epoch]:.4f}\n"
                    f"  Val Loss: {self.history.history['val_loss'][epoch]:.4f}, "
                    f"Val Accuracy: {self.history.history['val_accuracy'][epoch]:.4f}\n\n"
                )
            
            # Add summary
            f.write("\n" + "="*80 + "\n")
            f.write("Training Summary\n")
            f.write("="*80 + "\n")
            f.write(f"Total Epochs Trained: {actual_epochs}\n")
            f.write(f"Best Train Accuracy: {max(self.history.history['accuracy']):.4f}\n")
            f.write(f"Best Val Accuracy: {max(self.history.history['val_accuracy']):.4f}\n")
            f.write(f"Final Train Accuracy: {self.history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Final Val Accuracy: {self.history.history['val_accuracy'][-1]:.4f}\n")
        
        print(f"✅ Training logs saved to: {self.config.training_log_path}")

    def train(self):
        self.steps_per_epoch = max(1, self.train_generator.samples // self.train_generator.batch_size)
        self.validation_steps = max(1, self.valid_generator.samples // self.valid_generator.batch_size)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.best_model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print("\n" + "="*50)
        print("Starting Model Training")
        print("="*50)
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.valid_generator.samples}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Validation steps: {self.validation_steps}")
        print("="*50 + "\n")

        self.history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
        self.save_training_logs()

        # Plot training history
        self.visualizer.plot_training_history(self.history)

        final_train_acc = self.history.history["accuracy"][-1]
        final_val_acc = self.history.history["val_accuracy"][-1]
        final_train_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        
        print("\n" + "="*50)
        print("Training Completed Successfully!")
        print("="*50)
        print(f"Final Train Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"Final Train Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print("="*50 + "\n")

    def test(self):
        self.setup_test_generator()

        print("\n" + "="*50)
        print("Starting Model Evaluation on Test Set")
        print("="*50)
        print(f"Test samples: {self.test_generator.samples}")
        print("="*50 + "\n")

        test_loss, test_acc = self.model.evaluate(self.test_generator)
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        class_labels = list(self.test_generator.class_indices.keys())

        # Classification report and confusion matrix
        report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # Calculate ROC-AUC score
        if len(class_labels) == 2:
            roc_auc = roc_auc_score(y_true, predictions[:, 1])
        else:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
            roc_auc = roc_auc_score(y_true_bin, predictions, average='weighted', multi_class='ovr')

        results = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "roc_auc_score": float(roc_auc),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "test_samples": int(self.test_generator.samples)
        }

        # Save metrics
        with open(self.config.metrics_path, "w") as f:
            json.dump(results, f, indent=4)

        # Create visualizations
        self.visualizer.plot_confusion_matrix(cm, class_labels)
        self.visualizer.plot_roc_curve(y_true, predictions, class_labels)
        
        # Class distribution
        class_counts = {label: int(np.sum(y_true == idx)) for idx, label in enumerate(class_labels)}
        self.visualizer.plot_class_distribution(class_counts, class_labels, save_name='test_class_distribution.png')
        
        # Metrics summary
        self.visualizer.create_metrics_summary(results)

        print("\n" + "="*50)
        print("Test Evaluation Completed!")
        print("="*50)
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"\nPer-Class Metrics:")
        for class_name in class_labels:
            metrics = report_dict[class_name]
            print(f"\n{class_name.upper()}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
        print(f"\nMetrics saved to: {self.config.metrics_path}")
        print("="*50 + "\n")
