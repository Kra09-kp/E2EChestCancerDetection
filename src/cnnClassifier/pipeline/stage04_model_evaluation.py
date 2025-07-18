from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.model_evaluation import ModelEvaluation
from cnnClassifier import logger

STAGE_NAME = "Model Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        """Initializes the EvaluationPipeline with the given configuration."""
        self.config = ConfigManager()
        
    def run(self):
        """Runs the model evaluation stage."""
        evaluation_config = self.config.get_evaluation_config()
        evaluator = ModelEvaluation(evaluation_config)
        evaluator.evaluation()
        evaluator.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info("**************************")
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<<")
        evaluation_pipeline = EvaluationPipeline()
        evaluation_pipeline.run()
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} completed <<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e