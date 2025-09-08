# src/feature_engineering.py

import pandas as pd
import numpy as np
from src.utils import logging

class FeatureEngineering:
    """
    Performs feature engineering on the input dataframe to create new,
    potentially more predictive features.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the FeatureEngineering class.

        Args:
            data (pd.DataFrame): The input dataframe to transform.
        """
        self.df = data.copy() # Work on a copy to avoid side effects

    def _create_tenure_groups(self):
        """
        Bins the 'tenure' column into categorical groups.
        New customers are often different from long-term ones.
        """
        logging.info("Creating tenure groups feature.")
        bins = [0, 12, 24, 48, 60, 72]
        labels = ['0-1 Year', '1-2 Years', '2-4 Years', '4-5 Years', '5+ Years']
        self.df['TenureGroup'] = pd.cut(self.df['tenure'], bins=bins, labels=labels, right=False)
        return self

    def _create_service_count(self):
        """
        Calculates the total number of additional services a customer has.
        This can be an indicator of customer engagement.
        """
        logging.info("Creating total services count feature.")
        service_columns = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        # Count 'Yes' values across the service columns
        self.df['TotalServices'] = self.df[service_columns].apply(
            lambda row: sum(val == 'Yes' for val in row), axis=1
        )
        return self
    
    def _create_monthly_to_tenure_ratio(self):
        """
        Creates a ratio of MonthlyCharges to tenure.
        This might capture customers on expensive plans with low loyalty.
        """
        logging.info("Creating monthly charges to tenure ratio.")
        # Add 1 to tenure to avoid division by zero for new customers (tenure=0)
        self.df['MonthlyToTenureRatio'] = self.df['MonthlyCharges'] / (self.df['tenure'] + 1)
        return self

    def _simplify_contract_types(self):
        """
        Creates a binary feature for month-to-month contracts.
        This is often a very strong predictor of churn.
        """
        logging.info("Creating binary feature for month-to-month contract.")
        self.df['IsMonthToMonth'] = np.where(self.df['Contract'] == 'Month-to-month', 1, 0)
        return self

    def _has_no_dependents(self):
        """
        Creates a binary feature indicating if a customer has no partner and no dependents.
        Household structure can influence churn.
        """
        logging.info("Creating feature for customers with no dependents or partner.")
        self.df['HasNoDependents'] = np.where(
            (self.df['Partner'] == 'No') & (self.df['Dependents'] == 'No'), 1, 0
        )
        return self
    
    def transform_data(self):
        """
        Applies all feature engineering steps in a sequence.

        Returns:
            pd.DataFrame: The dataframe with new features.
        """
        logging.info("--- Starting Feature Engineering Transformation ---")
        (self._create_tenure_groups()
             ._create_service_count()
             ._create_monthly_to_tenure_ratio()
             ._simplify_contract_types()
             ._has_no_dependents())
        
        logging.info("Feature engineering complete. New columns added:")
        logging.info(self.df.columns)
        
        return self.df

# To run this script independently for testing
if __name__ == '__main__':
    # This assumes you have run data_ingestion.py first and have the train.csv file
    try:
        train_df_path = "data/processed/train.csv"
        train_df = pd.read_csv(train_df_path)
        
        logging.info("Original DataFrame columns:")
        logging.info(train_df.columns)
        print("\nOriginal DataFrame Head:\n", train_df.head())

        feature_engineer = FeatureEngineering(train_df)
        transformed_df = feature_engineer.transform_data()

        logging.info("Transformed DataFrame columns:")
        logging.info(transformed_df.columns)
        print("\nTransformed DataFrame Head with new features:\n", transformed_df.head())
        
        new_features = ['TenureGroup', 'TotalServices', 'MonthlyToTenureRatio', 'IsMonthToMonth', 'HasNoDependents']
        print("\nSample of new features:\n", transformed_df[new_features].head())

    except FileNotFoundError:
        print("\nERROR: `data/processed/train.csv` not found.")
        print("Please run the `src/data_ingestion.py` script first to generate it.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")