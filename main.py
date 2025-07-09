from cnnClassifier import logger
from cnnClassifier.pipeline.stage01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage" 

try:
    logger.info("**************************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.error(f"Error in running data ingestion pipeline: {e}")
    raise e


STAGE_NAME = "Prepare Base Model Stage"

try:
    logger.info("**************************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
    prepare_base_model_pipeline.run()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(f"Error in running prepare base model pipeline: {e}")
    raise e
