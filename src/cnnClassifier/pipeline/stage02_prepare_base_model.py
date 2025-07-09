from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        """Initializes the PrepareBaseModelPipeline with the given configuration.
        """
        self.config = ConfigManager()
        self.prepare_base_model_config = self.config.get_prepare_base_model_config()
        self.prepare_base_model = PrepareBaseModel(config=self.prepare_base_model_config)

    def run(self):
        """Runs the prepare base model stage."""
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<<")
        self.prepare_base_model.get_base_model()
        logger.info("Base model created successfully. Now updating the base model...")
        self.prepare_base_model.update_base_model()
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} completed <<<<<<<<<<<") 


if __name__ == "__main__":
    try:
        logger.info("**************************")
        prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
        prepare_base_model_pipeline.run()
    except Exception as e:
        logger.exception(e)
        raise e