# src/utils.py

import os
import yaml
import pickle
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config/config.yaml"):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info("Configuration file loaded successfully.")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at path: {config_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        return None

def save_object(obj, file_path):
    """
    Saves a Python object to a file using pickle.

    Args:
        obj: The Python object to save.
        file_path (str): The path where the object will be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")

def load_object(file_path):
    """
    Loads a Python object from a pickle file.

    Args:
        file_path (str): The path of the pickle file.

    Returns:
        The loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            logging.info(f"Object loaded successfully from {file_path}")
            return obj
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")