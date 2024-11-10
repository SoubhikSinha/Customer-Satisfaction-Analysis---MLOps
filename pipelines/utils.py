# Utility Functions

# Importing necessary libraries
import logging  # For logging errors and other messages
import pandas as pd  # For data manipulation and analysis
from src.data_cleaning import DataCleaning, DataPreProcessStrategy  # Custom modules for data cleaning

def get_data_for_test():
    """
    Function to load a sample dataset, clean the data, and return it in a JSON format.
    This is typically used for testing purposes in the pipeline.
    
    The steps performed in this function include:
        - Loading the dataset
        - Sampling a subset of the data
        - Applying data cleaning and preprocessing
        - Dropping an unwanted column
        - Converting the cleaned DataFrame to JSON
    
    Returns:
        result (str): A JSON string representation of the cleaned dataset.
    """
    try:
        # Step 1: Load the dataset from the specified path
        df = pd.read_csv("D:\GitHub_Repos\MLOps-DevOps-for-ML\data\olist_customers_dataset.csv")
        
        # Step 2: Take a random sample of 100 rows from the dataset
        df = df.sample(n=100)

        # Step 3: Initialize the preprocessing strategy and data cleaning objects
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        
        # Step 4: Clean the data using the custom `DataCleaning` class
        df = data_cleaning.handle_data()

        # Step 5: Drop the "review_score" column as it is not needed
        df.drop(["review_score"], axis=1, inplace=True)

        # Step 6: Convert the cleaned DataFrame to JSON (using 'split' orientation)
        result = df.to_json(orient="split")
        
        # Return the JSON result
        return result
    
    except Exception as e:
        # If an error occurs, log it and raise the exception
        logging.error(e)
        raise e
