import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from zipfile import ZipFile
from sklearn.metrics import classification_report, confusion_matrix
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

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
            for epoch in range(self.config.params_epochs):
                f.write(
                    f"Epoch {epoch + 1} - "
                    f"loss: {self.history.history['loss'][epoch]:.4f}, "
                    f"accuracy: {self.history.history['accuracy'][epoch]:.4f}, "
                    f"val_loss: {self.history.history['val_loss'][epoch]:.4f}, "
                    f"val_accuracy: {self.history.history['val_accuracy'][epoch]:.4f}\n"
                )

    def train(self):
        self.steps_per_epoch = max(1, self.train_generator.samples // self.train_generator.batch_size)
        self.validation_steps = max(1, self.valid_generator.samples // self.valid_generator.batch_size)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.best_model_path,
                save_best_only=True
            )
        ]

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

        final_train_acc = self.history.history["accuracy"][-1]
        final_val_acc = self.history.history["val_accuracy"][-1]
        print(f"Final Train Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    def test(self):
        self.setup_test_generator()

        test_loss, test_acc = self.model.evaluate(self.test_generator)
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        class_labels = list(self.test_generator.class_indices.keys())

        report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        results = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist()
        }

        with open(self.config.metrics_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Metrics saved to: {self.config.metrics_path}")
