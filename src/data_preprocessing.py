# src/data_preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import load_config, save_object, logging

class DataPreprocessor:
    def __init__(self, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        # Assuming your CSV has these columns
        self.numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_cols = ['Gender', 'Contract', 'PaymentMethod', 'DeviceProtection']
        self.target_col = self.config['model_params']['target_column']
        self.preprocessor_path = self.config['model_paths']['preprocessor_filename']

    def get_preprocessor_object(self):
        """
        Creates and returns a scikit-learn preprocessor pipeline object.
        This pipeline handles imputation, scaling, and encoding.
        """
        logging.info("Creating preprocessor object.")
        
        # Pipeline for numerical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine pipelines into a single ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='passthrough' # Keep other columns if any
        )
        
        return preprocessor
        
    def fit_and_transform_data(self, df):
        """
        Fits the preprocessor to the data and transforms it.
        Also saves the fitted preprocessor object.
        """
        logging.info("Fitting and transforming data.")
        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        preprocessor = self.get_preprocessor_object()
        X_transformed = preprocessor.fit_transform(X)
        
        # Save the preprocessor
        save_object(preprocessor, self.preprocessor_path)
        
        return X_transformed, y, preprocessor

    # Add many more functions for complex cleaning, handling outliers, etc.
    # to increase the line count with useful code.