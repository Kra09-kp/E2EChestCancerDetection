import torch
import torch.nn as nn
import torchvision
import os
import mlflow
import mlflow.pytorch
import dagshub
from torchvision.models import VGG16_Weights
from torch.utils import data
from torchvision.transforms import v2 as T
from tqdm import tqdm 
from urllib.parse import urlparse
from pathlib import Path
from cnnClassifier.utils.common import save_json
from cnnClassifier.entity.config_entity import EvaluationConfig


class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_generator(self):
        self._transform = T.Compose([
            T.Resize(self.config.params_image_size),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

        self.test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.config.testing_data, "test"),
            transform=self._transform
        )

        print(f"Classes as per index (testing): {self.test_dataset.class_to_idx}")

        self.test_loader = data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )



    @staticmethod
    def load_model(config: EvaluationConfig):

        model_path = config.model_path
        state_dict = torch.load(model_path,map_location=torch.device('cpu'))
        if type(state_dict) is dict:
            classes = config.all_params['CLASSES']
            freeze_all = config.all_params['FREEZE_ALL']
            freeze_till = config.all_params['FREEZE_TILL']

            # return classes
            model = torchvision.models.vgg16(weights=None)

            if freeze_all:
                for param in model.parameters():
                    param.requires_grad = False

            elif freeze_till is not None:
                for param in list(model.parameters())[:freeze_till]:
                    param.requires_grad = False

            # Modify the classifier
            model.classifier = nn.Sequential(
                nn.Linear(25088, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, classes)  # for 4-class classification
            )
        
            # state_dict = torch.load(model_path,map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(type(model))
            return model
        else:
            return state_dict
        

    def evaluation(self):
        self.test_generator()
        self.model = self.load_model(self.config)
        len_test_dataset = len(self.test_dataset)
        self.model.to(self.device)
        self.model.eval()

        accuracy = 0.0
        loss = 0.0
        with torch.no_grad():
            loop = tqdm(self.test_loader, total=len(self.test_loader), desc="Evaluating Model")
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                criterion = nn.CrossEntropyLoss()
                # Calculate loss and accuracy
                loss += criterion(outputs, labels).item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                accuracy += (preds == labels).sum().item()
                loop.set_description("Evaluation")
                loop.set_postfix(loss=loss, acc=accuracy)
                # For testing purposes, we break after the first batch
                break 
        
        print(len_test_dataset)
        val_accuracy = accuracy / len_test_dataset * 100
        val_loss = loss / len_test_dataset

        self.score = {
            "model_accuracy": val_accuracy,
            "model_loss": val_loss
        }
        print(self.score)
        self.save_score()
        return self.score
    

    def save_score(self):
        save_json(
            path = Path("scores.json"),
            data = self.score
        )

        print(f"Score saved at {Path('scores.json').resolve()}")

    def log_into_mlflow(self):
        dagshub.init(repo_owner='Kra09-kp', repo_name='E2EChestCancerDetection', mlflow=True) # type: ignore
        mlflow.set_tracking_uri(self.config.mlflow_uri) #type: ignore
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme #type: ignore

        mlflow.set_experiment("Checking Model")  #type: ignore

        with mlflow.start_run(): #type: ignore
            mlflow.log_params(self.config.all_params) #type: ignore
            mlflow.log_metric("model_accuracy", self.score["model_accuracy"]) #type: ignore
            mlflow.log_metric("model_loss", self.score["model_loss"]) #type: ignore
            
            print(tracking_url_type_store)  # debugging
            
            if tracking_url_type_store != "file":
                # Local MLflow — use model registry
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="ChestCancerDetectionModel")
            else:
                # Remote MLflow (like DagsHub) — no registry
                mlflow.pytorch.log_model(self.model, "model")
            
            print("Model logged into MLflow")

