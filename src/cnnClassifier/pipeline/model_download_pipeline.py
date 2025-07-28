from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.model_inference import ModelInference
from cnnClassifier import logger


class ModelInferencePipeline:
    def __init__(self):
        pass
    def run(self):
        
        config_manager = ConfigManager()
        model_inference_config = config_manager.get_model_inference_config()
        model_inference = ModelInference(config=model_inference_config)
        # Download the model
        logger.info("Starting model download...")
        model_inference.download_model()
        logger.info("Model download completed successfully.")

if __name__ == "__main__":
    pass