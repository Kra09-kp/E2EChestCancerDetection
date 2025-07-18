from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_save_path: Path
    update_base_model_path: Path
    params_image_size: list
    params_batch_size: int
    params_epochs: int
    params_classes: int
    params_weights: str
    params_learning_rate: float
    params_freeze_all: bool
    params_freeze_till: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data : Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_learning_rate: float
    params_image_size: list


@dataclass(frozen=True)
class EvaluationConfig:
    model_path: Path 
    testing_data: str
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int