# Model Evaluation

# Importing necessary libraries for model evaluation, logging, and metrics calculation
import logging
import pandas as pd
from zenml import step  # Importing the ZenML step decorator to define pipeline steps
from zenml.client import Client  # Importing Client to interact with the ZenML stack
import mlflow  # For logging metrics to MLflow

# Importing required classes from scikit-learn and other modules for typing and custom metrics
from sklearn.base import RegressorMixin  # Base class for regression models
from typing import Tuple  # To define tuple return types
from typing_extensions import Annotated  # To annotate return values in a specific way

# Importing custom evaluation metrics (Mean Squared Error, Root Mean Squared Error, R-squared)
from src.evaluation import MSE, RMSE, R2

# Set up the experiment tracker from ZenML's active stack to track experiments
experiment_tracker = Client().active_stack.experiment_tracker

# Method for model evaluation - the decorated function is part of the ZenML pipeline
@step(experiment_tracker=experiment_tracker.name)  # ZenML decorator to register this function as a pipeline step
def evaluate_model(
    model: RegressorMixin,  # Input parameter for the regression model to evaluate
    X_test: pd.DataFrame,  # Input feature data for testing
    y_test: pd.DataFrame   # Ground truth labels for the test data
) -> Tuple[
    Annotated[float, "r2"],  # R-squared score, annotated as a float
    Annotated[float, "rmse"],  # Root Mean Squared Error score, annotated as a float
]:
    """
    Evaluates the model on the ingested data.
    
    Args:
        model: A trained regression model to evaluate
        X_test: The test features (input data)
        y_test: The actual labels (target values)

    Returns:
        r2: The R-squared score for the model
        rmse: The Root Mean Squared Error of the model's predictions
    """
    try:
        # Generate predictions using the model on the test set
        prediction = model.predict(X_test)

        # Initialize and calculate the Mean Squared Error (MSE) using the custom MSE class
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        # Log the MSE metric to MLflow
        mlflow.log_metric("mse", mse)

        # Initialize and calculate the R-squared (R2) score using the custom R2 class
        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        # Log the R-squared metric to MLflow
        mlflow.log_metric("r2", r2)

        # Initialize and calculate the Root Mean Squared Error (RMSE) using the custom RMSE class
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        # Log the RMSE metric to MLflow
        mlflow.log_metric("rmse", rmse)

        # Return the R2 and RMSE scores as a tuple
        return r2, rmse
    except Exception as e:
        # If any error occurs during evaluation, log the error and raise it
        logging.error("Error in Evaluating Model : {}".format(e))
        raise e
    