# src/pipeline.py

import pandas as pd
from src.utils import logging
from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluation

class TrainingPipeline:
    """
    This class orchestrates the entire model training and evaluation pipeline.
    """
    def run_pipeline(self):
        """
        Executes all stages of the pipeline in sequence.
        """
        try:
            # --- STAGE 01: Data Ingestion ---
            logging.info(">>>>> Starting Stage 01: Data Ingestion <<<<<")
            ingestor = DataIngestion()
            train_path, test_path = ingestor.initiate_data_ingestion()
            logging.info(">>>>> Stage 01: Data Ingestion Complete <<<<<")

            # --- STAGE 02: Feature Engineering ---
            logging.info(">>>>> Starting Stage 02: Feature Engineering <<<<<")
            # Apply to training data
            train_df = pd.read_csv(train_path)
            fe_train = FeatureEngineering(train_df)
            train_df_featured = fe_train.transform_data()
            train_df_featured.to_csv(train_path, index=False)
            logging.info("Feature engineering applied to training data.")

            # Apply to testing data
            test_df = pd.read_csv(test_path)
            fe_test = FeatureEngineering(test_df)
            test_df_featured = fe_test.transform_data()
            test_df_featured.to_csv(test_path, index=False)
            logging.info("Feature engineering applied to testing data.")
            logging.info(">>>>> Stage 02: Feature Engineering Complete <<<<<")

            # --- STAGE 03: Model Training ---
            logging.info(">>>>> Starting Stage 03: Model Training <<<<<")
            trainer = ModelTrainer()
            trainer.initiate_model_training()
            logging.info(">>>>> Stage 03: Model Training Complete <<<<<")

            # --- STAGE 04: Model Evaluation ---
            logging.info(">>>>> Starting Stage 04: Model Evaluation <<<<<")
            evaluator = ModelEvaluation()
            evaluator.log_into_mlflow()
            logging.info(">>>>> Stage 04: Model Evaluation Complete <<<<<")
            
            logging.info("🎉🎉🎉 Full Training Pipeline Completed Successfully! 🎉🎉🎉")

        except Exception as e:
            logging.error(f"Pipeline failed with error: {e}")
            raise e

# To run the entire pipeline
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()