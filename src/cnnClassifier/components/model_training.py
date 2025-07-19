import torch
import torch.nn as nn
import torchvision
import os
from torchvision.models import VGG16_Weights
from torch.utils import data
from torchvision.transforms import v2 as T
from tqdm import tqdm 
from cnnClassifier.entity.config_entity import TrainingConfig





class Training:
    def __init__(self, config: TrainingConfig):
        """Initializes the Training class with a configuration object.
        
        Args:
            config (TrainingConfig): Configuration object containing training parameters.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_base_model(self):
        """Loads the base model from the specified path."""
        if not self.config.updated_base_model_path.exists():
            raise FileNotFoundError(f"Base model not found at {self.config.updated_base_model_path}")
        
        self.model = torch.load(self.config.updated_base_model_path, map_location=self.device)
        self.model.to(self.device)

        
    def train_valid_generator(self):
        """Creates data loaders for training and validation datasets."""
        
        self.val_transform = T.Compose([
            T.Resize(self.config.params_image_size[:2]),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

        if self.config.params_is_augmentation:
            self.train_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=(-10, 10)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.Resize(self.config.params_image_size[:2]),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406],  # Same mean/std as ImageNet
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.train_transform = self.val_transform

        self.train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.config.training_data, "train"),
            transform=self.train_transform
        )

        self.valid_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.config.training_data, "valid"),
            transform=self.val_transform
        )

        print(f"Classes as per index (training): {self.train_dataset.class_to_idx}")
        print(f"Classes as per index (validation): {self.valid_dataset.class_to_idx}")

        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True
        )

        self.valid_loader = data.DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

        


    def train(self):
        """Trains the model using the training dataset."""

        print(f"Training on device: {self.device}")
        print("Loading the model....")
        self.load_base_model()
        print("Model loaded successfully.")
        print("Loading Data....")
        self.train_valid_generator()
        print("Data loaded successfully.Now training started...")

        criterion = self.get_loss_function()
        optimizer = self.get_optimizer(self.model, self.config.params_learning_rate)
        scheduler = self.get_scheduler(optimizer)
        best_val_acc = 0.0
        epochs = self.config.params_epochs 

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            train_accuracy = 0.0
            loop = tqdm(self.train_loader, total=len(self.train_loader))
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                train_accuracy += (preds == labels).sum().item()
                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=loss.item(), acc=train_accuracy)
                break # just to test the code, remove this line to train on all data , i used this for testing my pipeline
            epoch_loss = running_loss / len(self.train_dataset)
            epoch_accuracy = train_accuracy / len(self.train_dataset) * 100
            epoch_val_accuracy,epoch_val_loss = self.get_validation_metrics(self.model,
                                                                            self.valid_loader,
                                                                            self.device,
                                                                            len(self.valid_dataset))
            print(f"Epoch [{epoch+1}/{epochs}],\
                    Loss: {epoch_loss:.4f}, \
                    Train Accuracy: {epoch_accuracy:.3f},\
                    Validation Loss: {epoch_val_loss:.4f},\
                    Validation Accuracy: {epoch_val_accuracy:.3f}%")

            if epochs > 2: 
                scheduler.step(epoch_val_loss)
            if epoch_val_accuracy > best_val_acc:
                self.save_model(self.model, self.config.trained_model_path.parent / f"best_model.pth")
                best_val_acc= epoch_val_accuracy

        self.save_model(self.model, self.config.trained_model_path)
    
    @staticmethod
    def get_loss_function():
        """Returns the loss function for training."""
        return nn.CrossEntropyLoss()
    
    @staticmethod   
    def get_optimizer(model, learning_rate):
        """Returns the optimizer for training.
        
        Args:
            model: The model to optimize.
            learning_rate (float): Learning rate for the optimizer.
        """
        print(type(model))
        return torch.optim.Adam(model.parameters(), lr=learning_rate) # type: ignore
    
    @staticmethod
    def get_scheduler(optimizer):
        """Returns the learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule.
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5)    

    @staticmethod
    def get_validation_metrics(model, valid_loader, device,len_val_dataset):
        """Calculates the validation accuracy of the model.
        
        Args:
            model: The trained model.
            valid_loader: DataLoader for the validation dataset.
            device: Device to run the model on (CPU or GPU).
        """
        model.eval()

        accuracy = 0.0
        loss = 0.0
        with torch.no_grad():
            loop = tqdm(valid_loader, total=len(valid_loader))
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                criterion = Training.get_loss_function()
                loss += criterion(outputs, labels).item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                accuracy += (preds == labels).sum().item()
                loop.set_description("Validation")
                loop.set_postfix(loss=loss, acc=accuracy)
                # For testing purposes, we break after the first batch
                break 

        val_accuracy = accuracy / len_val_dataset * 100
        val_loss = loss / len_val_dataset

        return val_accuracy, val_loss
    
    
    
    @staticmethod
    def save_model(model, path):
        """Saves the trained model to the specified path.
        
        Args:
            model: The model to save.
            path (Path): Path to save the model.
        """
        torch.save(model, path)
        print(f"Model saved at {path}")



