# Importing necessary libraries
import logging  # For logging messages (error, info, etc.)
import pandas as pd  # For handling data in DataFrame format
from zenml import step  # Decorator to define ZenML steps

# Importing custom classes for data cleaning and preprocessing strategies
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated  # For type annotations with metadata
from typing import Tuple  # For tuple typing

# Method for cleaning data
@step  # Decorator indicating this function is a ZenML step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],  # Training features
    Annotated[pd.DataFrame, "X_test"],   # Testing features
    Annotated[pd.Series, "y_train"],     # Training labels
    Annotated[pd.Series, "y_test"],      # Testing labels
]:
    """
    Cleans the data and divides it into train and test

    Args:
        df: Raw Data (input data in DataFrame format)
    Returns:
        X_train : Features for training
        X_test : Features for testing
        y_train : Labels for training
        y_test : Labels for testing
    """
    try:
        # Create a preprocessing strategy instance
        process_strategy = DataPreProcessStrategy()

        # Initialize DataCleaning class with the raw data and preprocessing strategy
        data_cleaning = DataCleaning(df, process_strategy)

        # Process the data (cleaning, missing values, encoding, etc.)
        processed_data = data_cleaning.handle_data()

        # Create a strategy for dividing the data into train and test sets
        divide_strategy = DataDivideStrategy()

        # Reinitialize DataCleaning with the processed data and data divide strategy
        data_cleaning = DataCleaning(processed_data, divide_strategy)

        # Perform the data splitting into X_train, X_test, y_train, and y_test
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        # Return the processed and split data
        return X_train, X_test, y_train, y_test
        
        # Log successful completion of the cleaning process
        logging.info("Data Cleaning Completed")
    
    except Exception as e:
        # If an error occurs during the cleaning process, log the error message
        logging.error("Error in cleaning data : {}".format(e))
        # Raise the error to ensure the pipeline fails gracefully
        raise e
