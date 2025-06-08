import os
from pathlib import Path
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        create_directories([config["root_dir"]])

        return DataIngestionConfig(
            root_dir=config["root_dir"],
            source_URL=config["source_URL"],
            local_data_file=config["local_data_file"],
            unzip_dir=config["unzip_dir"],
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config["prepare_base_model"]
        params = self.params

        create_directories([config["root_dir"]])

        return PrepareBaseModelConfig(
            root_dir=Path(config["root_dir"]),
            base_model_path=Path(config["base_model_path"]),
            updated_base_model_path=Path(config["updated_base_model_path"]),
            params_image_size=params["IMAGE_SIZE"],
            params_learning_rate=params["LEARNING_RATE"],
            params_include_top=params["INCLUDE_TOP"],
            params_weights=params["WEIGHTS"],
            params_classes=params["CLASSES"],
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config["training"]
        prepare = self.config["prepare_base_model"]
        params = self.params

        train_data = os.path.join(self.config["data_ingestion"]["unzip_dir"], "train")
        val_data = os.path.join(self.config["data_ingestion"]["unzip_dir"], "valid")
        test_data = os.path.join(self.config["data_ingestion"]["unzip_dir"], "test")

        create_directories([Path(training["root_dir"])])

        return TrainingConfig(
            updated_base_model_path=Path(prepare["updated_base_model_path"]),
            trained_model_path=Path(training["trained_model_path"]),
            best_model_path=Path(training["best_model_path"]),
            training_log_path=Path(training["training_log_path"]),
            metrics_path=Path(training["metrics_path"]),
            train_data=Path(train_data),
            val_data=Path(val_data),
            test_data=Path(test_data),
            params_epochs=params["EPOCHS"],
            params_batch_size=params["BATCH_SIZE"],
            params_image_size=params["IMAGE_SIZE"],
            params_is_augmentation=params["AUGMENTATION"],
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/train",
            mlflow_uri="https://dagshub.com/shrey.jiwane09/DLProject.mlflow",
            all_params=self.params,
            params_image_size=self.params["IMAGE_SIZE"],
            params_batch_size=self.params["BATCH_SIZE"],
        )
