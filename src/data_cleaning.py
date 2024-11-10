# Data Cleaning Classes

# Importing necessary libraries for logging, abstract classes, and data processing
import logging
from abc import ABC, abstractmethod  # For defining abstract base class
from typing import Union  # For specifying return types of different data structures

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets

# Abstract class defining strategy for handling data
class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data.
    This is a base class that will be extended to define specific data handling strategies.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method for handling data. 
        The specific handling strategy (e.g., preprocessing, splitting) will be defined in subclasses.
        """
        pass


# Data Preprocessing Strategy
class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data.
    This class implements data preprocessing steps like handling missing values, 
    dropping unnecessary columns, and retaining only numerical features.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by cleaning and transforming it.
        
        Steps include:
        - Dropping unnecessary columns
        - Filling missing values for specific columns
        - Dropping non-numeric columns
        - Dropping specific irrelevant columns
        
        Args:
            data: The raw data to preprocess.
        
        Returns:
            pd.DataFrame: The preprocessed data.
        """
        try:
            # Dropping irrelevant columns
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1  # Drop along columns
            )

            # Filling missing values (NULL values) with the median for specific columns
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            # Retaining only numeric columns
            data = data.select_dtypes(include=[np.number])

            # Dropping unnecessary columns
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e  # Re-raise the error for further handling

        
# Strategy for Data Splitting (Train-Test Split)
class DataDivideStrategy(DataStrategy):
    """
    Strategy to divide data into training and testing sets.
    This class handles the splitting of the data into features and target variable and
    returns the data divided into training and testing datasets.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Splits the data into features (X) and target variable (y), and then performs 
        a train-test split on them.
        
        Args:
            data: The preprocessed data to split into training and testing sets.
        
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
            - X_train, X_test: Features for training and testing.
            - y_train, y_test: Target values for training and testing.
        """
        try:
            # Splitting the data into features (X) and target variable (y)
            X = data.drop(["review_score"], axis=1)  # Drop the target variable from features
            y = data["review_score"]  # Target variable is "review_score"

            # Perform train-test split (80% training, 20% testing)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test  # Return the split data
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e  # Re-raise the error for further handling
        

# Data Cleaning Class - uses the above strategies
class DataCleaning:
    """
    Class for cleaning data, which preprocesses the data and divides it into train and test.
    This class uses different strategies to handle the data depending on the passed strategy.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """
        Initializes the DataCleaning class with raw data and a strategy for handling data.
        
        Args:
            data: The raw data to clean.
            strategy: The strategy to apply for data cleaning (e.g., preprocessing, splitting).
        """
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles the data according to the given strategy. The strategy will define 
        the specific action to perform (e.g., preprocessing or splitting).
        
        Returns:
            Processed data (either preprocessed data or the split datasets).
        """
        try:
            return self.strategy.handle_data(self.data)  # Apply the strategy on the data
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e  # Re-raise the error for further handling
        
'''
# If you need to run this file

if __name__ == "__main__":
    # Load data from CSV
    data = pd.read_csv("D:/GitHub_Repos/MLOps-Project-DevOps-for-ML/data/olist_customers_dataset.csv")
    
    # Initialize the DataCleaning object with preprocessing strategy
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    
    # Handle data preprocessing
    data_cleaning.handle_data()
'''
