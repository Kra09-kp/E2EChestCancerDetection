import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, List
import base64



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to the YAML file.
        
    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there is an error reading the YAML file.
        
    Returns:
        ConfigBox: Content of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("The YAML file is empty or not properly formatted.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {path_to_yaml} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the YAML file: {e}")  
    

@ensure_annotations
def create_directories(path_to_directories: list ,verbose: bool = True) -> type(None): # type: ignore
    """
    Creates directories if they do not exist.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        
    Returns:
        None
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose: 
            logger.info(f"Created directory: {path}")


@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """
    Saves a dictionary to a JSON file.
    
    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to save.
        
    Returns:
        None
    """
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"Saved data to {path}")

@ensure_annotations
def load_json(path: Path) -> dict:
    """
    Loads a dictionary from a JSON file.
    
    Args:
        path (Path): Path to the JSON file.
        
    Returns:
        dict: Loaded data.
        
    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If there is an error decoding the JSON file.
    """
    try:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {path} does not exist.")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from the file {path}.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the JSON file: {e}")
    
@ensure_annotations
def save_model(path: Path, model: Any) -> None:
    """
    Saves a model to a file using joblib.
    
    Args:
        path (Path): Path to save the model.
        model (Any): Model to save.
        
    Returns:
        None
    """
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

@ensure_annotations
def load_model(path: Path) -> Any:
    """
    Loads a model from a file using joblib.
    
    Args:
        path (Path): Path to the model file.
        
    Returns:
        Any: Loaded model.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        joblib.externals.loky.process_executor.TimeoutError: If there is an error loading the model.
    """
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The model file {path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the model: {e}")
    

@ensure_annotations
def encode_image(image_path: Path) -> str:
    """
    Encodes an image to a base64 string.
    
    Args:
        image_path (Path): Path to the image file.
        
    Returns:
        str: Base64 encoded string of the image.
        
    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If there is an error reading the image file.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        raise FileNotFoundError(f"The image file {image_path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while encoding the image: {e}")
    
@ensure_annotations
def decode_image(encoded_string: str, output_path: Path) -> None:
    """
    Decodes a base64 string to an image and saves it to a file.
    
    Args:
        encoded_string (str): Base64 encoded string of the image.
        output_path (Path): Path to save the decoded image.
        
    Returns:
        None
        
    Raises:
        Exception: If there is an error decoding the image or saving it.
    """
    try:
        with open(output_path, "wb") as image_file:
            image_file.write(base64.b64decode(encoded_string))
        logger.info(f"Image saved to {output_path}")
    except Exception as e:
        raise Exception(f"An error occurred while decoding the image: {e}")

@ensure_annotations
def get_file_size(path: Path) -> int:
    """
    Returns the size of a file in bytes.
    
    Args:
        path (Path): Path to the file.
        
    Returns:
        int: Size of the file in bytes.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")
    
    return path.stat().st_size

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """
    Saves data to a binary file.
    
    Args:
        data (Any): Data to save.
        path (Path): Path to the binary file.
        
    Returns:
        None
    """
    with open(path, 'wb') as bin_file:
        joblib.dump(data, bin_file)
    logger.info(f"Data saved into binary file at {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file.
    
    Args:
        path (Path): Path to the binary file.
        
    Returns:
        Any: Loaded data.
        
    Raises:
        FileNotFoundError: If the binary file does not exist.
        Exception: If there is an error loading the data.
    """
    try:
        with open(path, 'rb') as bin_file:
            return joblib.load(bin_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The binary file {path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the binary file: {e}")

