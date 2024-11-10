# Ingest Data

# Importing necessary libraries for logging and data manipulation
import logging
import pandas as pd
from zenml import step  # ZenML decorator to define a pipeline step

# Class for Data Ingestion
class IngestData:
    """
    Ingesting the data from the specified data path.
    """
    def __init__(self, data_path: str):
        """
        Initializes the IngestData class with the path to the data.

        Args:
            data_path: Path to the dataset to be ingested.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Reads the data from the data_path and returns it as a pandas DataFrame.

        Logs an info message indicating data ingestion is happening from the specified path.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        # Using pandas to read the CSV file from the specified path
        return pd.read_csv(self.data_path)
    
# ZenML step function to ingest data
@step  # This decorator marks the function as a pipeline step in ZenML
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingests data from the provided path and returns it as a pandas DataFrame.

    Args:
        data_path: The path to the dataset (usually a CSV file).
    
    Returns:
        pd.DataFrame: The ingested data as a pandas DataFrame.
    """
    try:
        # Create an instance of the IngestData class with the provided data path
        ingest_data = IngestData(data_path)
        # Use the get_data method to load the data into a DataFrame
        df = ingest_data.get_data()
        # Return the ingested DataFrame
        return df
    except Exception as e:
        # If any error occurs, log the error and raise it to stop further execution
        logging.error(f"Error while ingesting data: {e}")
        raise e
