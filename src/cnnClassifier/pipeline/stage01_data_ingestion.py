from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger


STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        self.config = ConfigManager().get_data_ingestion_config()
        self.data_ingestion = DataIngestion(self.config)

    def run(self):
        
        config_manager = ConfigManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)

        data_ingestion.download_data()
        data_ingestion.unzip_data() 

        


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()
        logger.info(f">>>>>> Stage:  {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.error(f"Error in running data ingestion pipeline: {e}")
        raise e
    