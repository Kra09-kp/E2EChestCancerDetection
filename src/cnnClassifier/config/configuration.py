from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig)

class ConfigManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        print(self.config.artifacts_root)
        create_directories([self.config.artifacts_root]) 
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        config = self.config.prepare_base_model

        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_save_path=Path(config.base_model_save_path),
            update_base_model_path=Path(config.update_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_classes=self.params.CLASSES,
            params_weights=self.params.WEIGHTS,
            params_learning_rate=self.params.LEARNING_RATE
        )

        return prepare_base_model_config