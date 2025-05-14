import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import json
from cnnClassifier.entity.config_entity import TrainingConfig




class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1./255)
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

    def test_generator(self):
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.test_data,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def save_training_logs(self, log_path="training_logs.txt"):
        with open(log_path, "w") as f:
            for epoch in range(self.config.params_epochs):
                f.write(
                    f"Epoch {epoch+1} - "
                    f"loss: {self.history.history['loss'][epoch]:.4f}, "
                    f"accuracy: {self.history.history['accuracy'][epoch]:.4f}, "
                    f"val_loss: {self.history.history['val_loss'][epoch]:.4f}, "
                    f"val_accuracy: {self.history.history['val_accuracy'][epoch]:.4f}\n"
                )


    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            verbose=1
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        # Logs
        self.save_training_logs()
        

        final_train_acc = self.history.history["accuracy"][-1]
        final_val_acc = self.history.history["val_accuracy"][-1]
        print(f"✅ Final Train Accuracy: {final_train_acc:.4f}")
        print(f"✅ Final Validation Accuracy: {final_val_acc:.4f}")

    
    def test(self):
        self.test_generator()

        # Evaluate test performance
        test_loss, test_acc = self.model.evaluate(self.test_generator)
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        class_labels = list(self.test_generator.class_indices.keys())

        # Classification report
        report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        # Combine all metrics
        results = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist()
        }

        # Save to JSON
        with open("metrics.json", "w") as f:
            json.dump(results, f, indent=4)

        print("\n✅ Metrics saved to 'metrics.json'")


    

