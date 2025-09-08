# src/data_ingestion.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_config, logging

class DataIngestion:
    """
    Handles the ingestion of raw data, initial cleaning, and splitting into
    training and testing sets.
    """
    def __init__(self, config_path="config/config.yaml"):
        """Initializes the DataIngestion component."""
        try:
            self.config = load_config(config_path)
            if self.config is None:
                raise ValueError("Configuration could not be loaded.")
            
            self.ingestion_config = self.config['data_paths']
            self.model_params = self.config['model_params']
            
            # Ensure the processed data directory exists before we start
            processed_dir = os.path.dirname(self.ingestion_config['train_data'])
            os.makedirs(processed_dir, exist_ok=True)
            logging.info(f"Ensured '{processed_dir}' directory exists.")
            
        except Exception as e:
            logging.error(f"Error during DataIngestion initialization: {e}")
            raise e

    def initiate_data_ingestion(self):
        """
        Main method to perform data ingestion, cleaning, and splitting.
        """
        logging.info("Starting the data ingestion process.")
        raw_data_path = self.ingestion_config['raw_data']

        # --- Step 1: Read the CSV file with robust error handling ---
        try:
            logging.info(f"Attempting to read raw data from: {raw_data_path}")
            df = pd.read_csv(raw_data_path)
            logging.info(f"Successfully read raw data. Shape: {df.shape}")

        except FileNotFoundError:
            logging.error(f"CRITICAL ERROR: The raw data file was NOT FOUND at the path: {raw_data_path}")
            logging.error("Please double-check your file path in config.yaml and ensure the file exists.")
            # We must stop the program here.
            raise
        except Exception as e:
            logging.error(f"CRITICAL ERROR: An unexpected error occurred while reading the data: {e}")
            raise

        # --- Step 2: Perform cleaning and splitting on the loaded dataframe ---
        try:
            logging.info("Starting data cleaning...")
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df.dropna(subset=['TotalCharges'], inplace=True)
            logging.info("Data cleaning complete.")

            logging.info("Splitting data into training and testing sets.")
            train_set, test_set = train_test_split(
                df,
                test_size=self.model_params['test_size'],
                random_state=self.model_params['random_state'],
                stratify=df[self.model_params['target_column']]
            )

            train_path = self.ingestion_config['train_data']
            test_path = self.ingestion_config['test_data']
            
            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)

            logging.info(f"Training data saved to: {train_path}")
            logging.info(f"Testing data saved to: {test_path}")
            logging.info("Data ingestion process completed successfully.")

            return (train_path, test_path)

        except Exception as e:
            logging.error(f"An error occurred during data processing/splitting: {e}")
            raise