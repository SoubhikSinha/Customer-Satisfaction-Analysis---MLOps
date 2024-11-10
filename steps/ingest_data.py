'''
Ingest Data
'''

# Importing necessary libraries
import logging

import pandas as pd
from zenml import step

# Class of Data Ingestion
class IngestData:
    """
    Ingesting the data form the data_path
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path : path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data form the data_path
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path.

    Args:
        data_paths: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e