# src/model_training.py

import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.utils import load_config, save_object, logging
from src.feature_engineering import FeatureEngineering

class ModelTrainer:
    """
    This class handles the model training process, including preprocessing,
    training multiple models, evaluating them, and saving the best one.
    """
    def __init__(self, config_path="config/config.yaml"):
        """Initializes the ModelTrainer component."""
        try:
            self.config = load_config(config_path)
            if self.config is None:
                logging.error("Failed to load configuration. Exiting.")
                sys.exit(1)
            self.params = self.config['model_params']
            self.paths = self.config['data_paths']
            self.model_paths = self.config['model_paths']
        except Exception as e:
            logging.error(f"Error during ModelTrainer initialization: {e}")
            raise e

    def _get_preprocessor_object(self, df):
        """
        Creates and returns a scikit-learn preprocessor pipeline object
        based on the dataframe's column types.
        """
        logging.info("Creating data preprocessor object.")
        
        # Identify numerical and categorical columns from the dataframe
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        logging.info(f"Numerical columns: {list(numerical_cols)}")
        logging.info(f"Categorical columns: {list(categorical_cols)}")

        # Create preprocessing pipelines for each type
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine them using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )
        return preprocessor

    def initiate_model_training(self):
        """
        Main method to orchestrate the model training workflow.
        """
        logging.info("--- Starting Model Training Stage ---")
        try:
            # Load feature-engineered training data
            train_df = pd.read_csv(self.paths['train_data'])

            # Split data into features and target
            target_column = self.params['target_column']
            X = train_df.drop(columns=[target_column, 'customerID'], errors='ignore')
            y = train_df[target_column]
            # Convert target to binary if it's 'Yes'/'No'
            if y.dtype == 'object':
                y = y.apply(lambda x: 1 if x == 'Yes' else 0)


            # Split data for training and validation within the training set
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.params['random_state']
            )

            preprocessor = self._get_preprocessor_object(X_train)
            
            # --- Set up MLflow ---
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.config['mlflow']['experiment_name'])

            with mlflow.start_run():
                logging.info("MLflow run started for training.")
                mlflow.log_params(self.params) # Log general parameters

                # --- Transform data ---
                X_train_processed = preprocessor.fit_transform(X_train)
                X_val_processed = preprocessor.transform(X_val)

                # --- Save the preprocessor ---
                preprocessor_path = self.model_paths['preprocessor_filename']
                save_object(preprocessor, preprocessor_path)
                logging.info(f"Preprocessor object saved to {preprocessor_path}")

                # --- Train and evaluate multiple models ---
                models = {
                    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
                    "RandomForest": RandomForestClassifier(**self.params['random_forest']),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                }

                model_report = {}
                for name, model in models.items():
                    model.fit(X_train_processed, y_train)
                    y_pred_val = model.predict(X_val_processed)
                    score = f1_score(y_val, y_pred_val)
                    model_report[name] = score
                    logging.info(f"Model: {name}, Validation F1-Score: {score:.4f}")
                    mlflow.log_metric(f"val_f1_{name}", score)

                # --- Find and save the best model ---
                best_score = max(model_report.values())
                best_model_name = [name for name, score in model_report.items() if score == best_score][0]
                best_model = models[best_model_name]

                logging.info(f"Best model is {best_model_name} with F1-Score: {best_score:.4f}")
                mlflow.log_metric("best_model_f1_score", best_score)
                mlflow.set_tag("best_model", best_model_name)

                # Save the best model
                model_output_path = os.path.join(self.model_paths['model_output_dir'], "best_model.pkl")
                save_object(best_model, model_output_path)
                logging.info(f"Best model saved to {model_output_path}")

            logging.info("--- Model Training Stage Complete ---")
            return model_output_path
        
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise e

# To run this script independently for testing
if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.initiate_model_training()