from cnnClassifier import logger
from cnnClassifier.pipeline.stage01_data_ingestion import DataIngestionPipeline

STAGE_NAME = "Data Ingestion Stage" 

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.error(f"Error in running data ingestion pipeline: {e}")
    raise e