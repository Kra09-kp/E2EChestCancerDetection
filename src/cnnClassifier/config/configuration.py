from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, 
                                                PrepareBaseModelConfig, 
                                                TrainingConfig,
                                                EvaluationConfig,
                                                ModelInferenceConfig)

from pathlib import Path
import os

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
            params_learning_rate=self.params.LEARNING_RATE,
            params_freeze_all=self.params.FREEZE_ALL,
            params_freeze_till=self.params.FREEZE_TILL
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.train_model
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir,"Data")
        create_directories(
            [Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.model_save_path),
            updated_base_model_path=Path(prepare_base_model.update_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_learning_rate=params.LEARNING_RATE,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            model_path = Path("artifacts/train_model/best_model.pth"),
            testing_data = "artifacts/data_ingestion/Data",
            all_params=self.params,
            mlflow_uri="https://dagshub.com/Kra09-kp/E2EChestCancerDetection.mlflow", 
            params_image_size=self.params.IMAGE_SIZE[:2],
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config

    def get_model_inference_config(self) -> ModelInferenceConfig:
        config = self.config.model_inference
        create_directories([config.root_dir])
        return ModelInferenceConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_model_file=config.local_model_file,
            model_dir=config.model_dir
        )