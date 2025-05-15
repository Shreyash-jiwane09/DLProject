import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


import dagshub
dagshub.init(repo_owner='shrey.jiwane09', repo_name='DLProject', mlflow=True)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

        # Predictions and ground truth
        predictions = self.model.predict(self.valid_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.valid_generator.classes
        class_labels = list(self.valid_generator.class_indices.keys())

        # Save to instance
        self.y_pred = y_pred
        self.y_true = y_true
        self.class_labels = class_labels

        # Classification report and confusion matrix
        self.report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True,zero_division=0)
        self.cm = confusion_matrix(y_true, y_pred)

    
    def save_score(self):
        all_metrics = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1]),
            "classification_report": self.report_dict,
            "confusion_matrix": self.cm.tolist()
        }

        save_json(path=Path("metrics.json"), data=all_metrics)
        print("✅ Saved evaluation metrics to metrics.json")

    

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            # Log model params
            mlflow.log_params(self.config.all_params)

            # Log metrics
            mlflow.log_metrics({
                "loss": float(self.score[0]),
                "accuracy": float(self.score[1])
            })

            # Save full classification report and confusion matrix
            mlflow.log_dict(self.report_dict, "classification_report.json")
            mlflow.log_dict({"confusion_matrix": self.cm.tolist()}, "confusion_matrix.json")

            # Log model to MLflow/DagsHub
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")

    