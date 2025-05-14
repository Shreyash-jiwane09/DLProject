from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training
from cnnClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Load config
        config = ConfigurationManager()
        training_config = config.get_training_config()

        # Instantiate trainer
        training = Training(config=training_config)

        # Load model & data
        training.get_base_model()
        training.train_valid_generator()

        # Train the model
        training.train()

        # Evaluate on test data
        training.test()  # <-- NEW: Evaluate test accuracy


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
