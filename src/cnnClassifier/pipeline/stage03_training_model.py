from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger

STAGE_NAME = "Training Stage"

class TrainingPipeline:
    def __init__(self):
        """Initializes the TrainingPipeline with the given configuration."""
        self.config = ConfigManager()
        
    def run(self):
        """Runs the training stage."""
        training_config = self.config.get_training_config()
        trainer = Training(training_config)
        trainer.train()
        
        


if __name__ == "__main__":
    try:
        logger.info("**************************")
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<<")
        training_pipeline = TrainingPipeline()
        training_pipeline.run()
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} completed <<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e