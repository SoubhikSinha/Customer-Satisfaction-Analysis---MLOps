# Model Training

# Importing necessary libraries for model training, logging, and experiment tracking
import logging
import pandas as pd
from zenml import step  # ZenML decorator for pipeline step definition
from src.model_dev import LinearRegressionModel  # Custom model for linear regression
from sklearn.base import RegressorMixin  # Base class for regression models
from .config import ModelNameConfig  # Configuration class for model selection

import mlflow  # For logging model metrics and configurations
from zenml.client import Client  # To interact with ZenML's client

# Set up experiment tracker from ZenML's active stack to track experiments
experiment_tracker = Client().active_stack.experiment_tracker

'''
👆 Enables you to systematically compare the outcomes of different 
models or experiments side by side, helping you identify which 
one performed best.
'''

# Method for training the model, decorated as a ZenML pipeline step
@step(experiment_tracker=experiment_tracker.name)  # Decorator to register as a pipeline step
def train_model(
    X_train: pd.DataFrame,  # Feature data for training
    X_test: pd.DataFrame,   # Feature data for testing (not used in training but for evaluation later)
    y_train: pd.Series,     # Target values for training
    y_test: pd.Series,      # Target values for testing (not used in training)
    config: ModelNameConfig = ModelNameConfig(),  # Model configuration, default to LinearRegression
) -> RegressorMixin:  # The function returns a trained regression model
    """
    Trains the model on the provided training data.

    Args:
        X_train: Features for training (input data).
        X_test: Features for testing (used for evaluation later).
        y_train: Actual target values for training.
        y_test: Actual target values for testing (used for evaluation later).
        config: Configuration object that specifies which model to use (default is LinearRegression).

    Returns:
        trained_model: The trained regression model.
    """
    try:
        model = None
        # Check if the model name in the configuration is "LinearRegression"
        if config.model_name == "LinearRegression":
            # Enable automatic logging of the model and metrics to MLflow
            mlflow.sklearn.autolog()  # This will log parameters, metrics, and the model itself
            # Instantiate and train the Linear Regression model
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)  # Train the model on the training data
            return trained_model  # Return the trained model
        else:
            # Raise an error if the model type is not supported
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        # Log the error message if any exception occurs during the training process
        logging.error(f"Error in training model: {e}")
        raise e  # Re-raise the exception after logging the error
