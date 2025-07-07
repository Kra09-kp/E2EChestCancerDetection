import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_file_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path 

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        '''
        Downloads data from the source URL to the local data file path.
        If the file already exists, it skips the download.
        '''

        try:
            dataset_url = self.config.source_url
            zip_download_path = self.config.local_data_file
            unzip_dir = self.config.unzip_dir
            os.makedirs(os.path.dirname(unzip_dir), exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} to {zip_download_path}")

            file_id = dataset_url.split('/')[-2]
            prefix = f"https://drive.google.com/uc?export=download&id={file_id}"
            if os.path.exists(zip_download_path):
                file_size = get_file_size(Path(zip_download_path))
                if file_size > 0:
                    logger.info(f"File already exists at {zip_download_path} with size {file_size} bytes. Skipping download.")
                    return
            gdown.download(prefix, zip_download_path, quiet=False)

            logger.info(f"Downloaded data to {zip_download_path}")

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise e  

    def unzip_data(self):
        '''
        Unzips the downloaded data file to the specified unzip directory.
        If the unzip directory already exists, it skips the unzipping.
        '''

        os.makedirs(self.config.unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info(f"Unzipped data to {self.config.unzip_dir}")
        