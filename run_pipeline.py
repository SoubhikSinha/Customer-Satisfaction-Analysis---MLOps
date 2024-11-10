# Importing the necessary libraries
from pipelines.training_pipeline import train_pipeline  # Import the training pipeline
from zenml.client import Client  # Import ZenML client to interact with the ZenML stack

# Running the pipeline
if __name__ == "__main__":
    # Get the tracking URI for experiment tracking from the active ZenML stack
    # This URI is essential for tracking the experiments and logging outputs (like metrics or models)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    # Run the training pipeline, providing the path to the dataset as an argument
    # Make sure to provide the correct path to the dataset for training the model
    train_pipeline(data_path = "D:/GitHub_Repos/MLOps-Project-DevOps-for-ML/data/olist_customers_dataset.csv")