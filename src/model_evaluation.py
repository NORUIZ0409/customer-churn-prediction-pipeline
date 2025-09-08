# src/model_evaluation.py

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse

from src.utils import load_config, load_object, logging
from src.feature_engineering import FeatureEngineering

class ModelEvaluation:
    """
    Handles the evaluation of the trained model on the test dataset,
    generates visualizations, and logs everything to MLflow.
    """
    def __init__(self, config_path="config/config.yaml"):
        """
        Initializes the ModelEvaluation component.

        Args:
            config_path (str): Path to the configuration file.
        """
        try:
            self.config = load_config(config_path)
            if self.config is None:
                logging.error("Failed to load configuration. Exiting.")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Error during ModelEvaluation initialization: {e}")
            raise e

    def _eval_metrics(self, actual, pred):
        """
        Calculates and returns a dictionary of classification metrics.

        Args:
            actual (np.array): True labels.
            pred (np.array): Predicted labels.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and f1-score.
        """
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def log_into_mlflow(self):
        """
        The main method to perform model evaluation and logging.
        It loads the test data, model, preprocessor, applies transformations,
        makes predictions, calculates metrics, and logs them to MLflow.
        """
        logging.info("--- Starting Model Evaluation and MLflow Logging ---")

        try:
            # --- Load Data and Artifacts ---
            test_data_path = self.config['data_paths']['test_data']
            model_path = os.path.join(self.config['model_paths']['model_output_dir'], "best_model.pkl")
            preprocessor_path = self.config['model_paths']['preprocessor_filename']
            reports_dir = self.config['reports_path']
            os.makedirs(reports_dir, exist_ok=True)

            test_df = pd.read_csv(test_data_path)
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            if model is None or preprocessor is None:
                logging.error("Model or preprocessor could not be loaded. Aborting evaluation.")
                return

            logging.info("Test data, model, and preprocessor loaded successfully.")

            # --- Prepare Test Data ---
            logging.info("Applying feature engineering and preprocessing to test data.")
            feature_engineer = FeatureEngineering(test_df)
            test_df_featured = feature_engineer.transform_data()
            
            target_column = self.config['model_params']['target_column']
            X_test = test_df_featured.drop(columns=[target_column, 'customerID'], errors='ignore')
            y_test = test_df_featured[target_column]
            # Convert target to binary if it's 'Yes'/'No'
            if y_test.dtype == 'object':
                y_test = y_test.apply(lambda x: 1 if x == 'Yes' else 0)


            X_test_processed = preprocessor.transform(X_test)

            # --- Set MLflow Tracking ---
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.config['mlflow']['experiment_name'])

            with mlflow.start_run():
                logging.info("MLflow run started for evaluation.")

                # --- Make Predictions ---
                predicted_labels = model.predict(X_test_processed)
                predicted_probs = model.predict_proba(X_test_processed)[:, 1]

                # --- Calculate Metrics ---
                metrics = self._eval_metrics(y_test, predicted_labels)
                roc_auc = roc_auc_score(y_test, predicted_probs)
                metrics['roc_auc'] = roc_auc
                
                logging.info(f"Evaluation Metrics on Test Data: {metrics}")
                
                # --- Log Metrics to MLflow ---
                mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

                # --- Generate and Save Visualizations ---
                # Confusion Matrix
                cm = confusion_matrix(y_test, predicted_labels)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(reports_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)
                logging.info(f"Confusion matrix saved to {cm_path} and logged to MLflow.")

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, predicted_probs)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                roc_path = os.path.join(reports_dir, "roc_curve.png")
                plt.savefig(roc_path)
                plt.close()
                mlflow.log_artifact(roc_path)
                logging.info(f"ROC curve saved to {roc_path} and logged to MLflow.")
                
                # Log model as well, for full traceability
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ChurnPredictionModel")
                else:
                    mlflow.sklearn.log_model(model, "model")

            logging.info("--- Model Evaluation and MLflow Logging Complete ---")

        except Exception as e:
            logging.error(f"An error occurred during model evaluation: {e}")
            raise e

# To run this script independently after training
if __name__ == '__main__':
    # You first need to run the training pipeline to generate the model
    # Then run this script to evaluate it.
    
    # You'll also need to add an mlflow section to your config.yaml
    # Example:
    # mlflow:
    #   tracking_uri: "mlruns"  # or a remote server like "http://127.0.0.1:5000"
    #   experiment_name: "ECommerce Churn Prediction"
    
    # And start the mlflow server with `mlflow ui` in your terminal
    
    evaluation = ModelEvaluation()
    evaluation.log_into_mlflow()