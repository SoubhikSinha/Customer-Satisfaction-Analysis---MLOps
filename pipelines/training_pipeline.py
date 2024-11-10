# Training Pipeline

# Importing necessary libraries from ZenML and custom steps
from zenml import pipeline
from steps.ingest_data import ingest_df  # Step for ingesting data
from steps.clean_data import clean_df  # Step for cleaning data
from steps.model_train import train_model  # Step for training the model
from steps.evaluation import evaluate_model  # Step for evaluating the trained model

# Method for Training Pipeline
# @pipeline(enable_cache=True) --> This decorator enables caching of the pipeline execution
# If enable_cache is set to False, the pipeline will not use previously computed outputs (e.g., fresh runs each time)
@pipeline
def train_pipeline(data_path: str):
    """
    Defines the pipeline for training the model. The pipeline steps involve ingesting the data,
    cleaning it, training a model, and evaluating its performance.

    Args:
        data_path (str): Path to the dataset used for training.
    """
    # Step 1: Ingest the data from the given path
    df = ingest_df(data_path)  # This step loads the data into a pandas DataFrame
    
    # Step 2: Clean the data, splitting it into training and testing sets
    X_train, X_test, y_train, y_test = clean_df(df)  # This step splits the data into training and test sets
    
    # Step 3: Train the model using the cleaned training data
    model = train_model(X_train, X_test, y_train, y_test)  # This step trains the model on the training data
    
    # Step 4: Evaluate the trained model using the test data
    r2, rmse = evaluate_model(model, X_test, y_test)  # This step evaluates the model's performance using R2 and RMSE metrics
