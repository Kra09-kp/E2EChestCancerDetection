import os

import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_file_size
from cnnClassifier.entity.config_entity import ModelInferenceConfig
from pathlib import Path


class ModelInference:
    def __init__(self,config: ModelInferenceConfig) -> None:
        self.config = config

    def download_model(self):
        
        '''
        Downloads model from the source URL to the local data file path.
        If the file already exists, it skips the download.
        '''

        try:
            model_url = self.config.source_url
            model_path = self.config.local_model_file
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info(f"Downloading model from {model_url}")

            file_id = model_url.split('/')[-2]
            prefix = f"https://drive.google.com/uc?export=download&id={file_id}"
            if os.path.exists(model_path):
                file_size = get_file_size(Path(model_path))
                if file_size > 0:
                    logger.info(f"File already exists at {model_path} with size {file_size} bytes. Skipping download.")
                    return
            
            gdown.download(prefix, model_path, quiet=False)

            logger.info(f"Downloaded data to {model_path}")

        except Exception as e:
            logger.info(f"Error downloading data: {e}")
            raise e
    