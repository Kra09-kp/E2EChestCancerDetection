import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision.models import VGG16_Weights
from cnnClassifier.entity.config_entity import (PrepareBaseModelConfig)



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """Initializes the PrepareBaseModel with the given configuration.
        
        Args:
            config (PrepareBaseModelConfig): Configuration for preparing the base model.
        """
        self.config = config

    def get_base_model(self):
        """ Return vgg16 model with imagenet weights without top layer."""
        base_model = torchvision.models.vgg16(
            weights=self.config.params_weights,
            progress=True,  
        )

        # Remove the top layer (classifier)
        base_model.classifier = nn.Sequential()

        #save the base model
        base_model_save_path = self.config.base_model_save_path
        os.makedirs(base_model_save_path.parent,exist_ok=True)
        self.save_model(base_model, base_model_save_path)
        print(f"Base model saved at: {base_model_save_path}")


    def update_base_model(self):
        """Updates the base model by modifying the classifier and setting up the optimizer.
        
        Args:
            freeze_all (bool): Whether to freeze all layers.
            freeze_till (int): Layer index till which to freeze.
        """
        model = self._prepare_full_model(
            model=self.config.base_model_save_path,
            classes=self.config.params_classes,
            freeze_all=self.config.params_freeze_all,
            freeze_till=self.config.params_freeze_till
        )
        # Save the updated model
        update_base_model_path = self.config.update_base_model_path
        os.makedirs(update_base_model_path.parent, exist_ok=True)
        torch.save(model, update_base_model_path)
        print(f"Updated base model saved at: {update_base_model_path}")

        # Print model summary
        print("Model Summary:")
        summary(model, input_size=(3,224,224), batch_size=self.config.params_batch_size)

    @staticmethod
    def _prepare_full_model(model,classes,freeze_all,freeze_till):
        """Prepares the full model by modifying the classifier and setting up the optimizer.
        
        Args:
            model (Path): Path to the base model.
            classes (int): Number of output classes.
            freeze_all (bool): Whether to freeze all layers.
            freeze_till (int): Layer index till which to freeze.
            learning_rate (float): Learning rate for the optimizer.
        
        Returns:
            torch.nn.Module: The modified model with a new classifier.
        """

        model = PrepareBaseModel.load_model(model_path=model)
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


        return model


    @staticmethod
    def save_model(model: torch.nn.Module, path: Path):
        """Saves the model to the specified path.
        
        Args:
            model (torch.nn.Module): The model to save.
            path (Path): The path where the model will be saved.
        """
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(model_path: Path):
        """Loads the model from the specified path.
        
        Args:
            model_path (Path): The path from which to load the model.
        Returns:
            torch.nn.Module: The loaded model.
        """
        model = torchvision.models.vgg16(weights=None)
        model.classifier = torch.nn.Sequential()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        return model
