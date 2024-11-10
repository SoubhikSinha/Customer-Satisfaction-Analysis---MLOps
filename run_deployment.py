# Importing necessary libraries
import click  # CLI framework for handling command-line interface
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri  # To retrieve MLflow tracking URI
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer  # For handling model deployment with MLflow
from zenml.integrations.mlflow.services import MLFlowDeploymentService  # To manage deployment services
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline  # The deployment and inference pipelines
from typing import cast  # For type casting to ensure correct service type

# Constants for configuration choices in the command-line interface
DEPLOY = "deploy"  # Option to run deployment only
PREDICT = "predict"  # Option to run prediction only
DEPLOY_AND_PREDICT = "deploy_and_predict"  # Option to run both deployment and prediction

# Define the CLI command using Click
@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),  # User can choose between deploy, predict, or both
    default=DEPLOY_AND_PREDICT,  # Default choice is to both deploy and predict
    help="Choose to run deployment pipeline ('deploy') or prediction ('predict'). Default is both ('deploy_and_predict')."
)
@click.option(
    "--min-accuracy",
    default=0.0,
    type=float,
    help="Minimum accuracy required to deploy the model"  # Minimum accuracy to deploy the model
)
def run_deployment(config: str, min_accuracy: float):
    # Initialize the MLFlow model deployer
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    # Determine whether to run deployment or prediction based on the config argument
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    # If deploy is selected, run the continuous deployment pipeline
    if deploy:
        continuous_deployment_pipeline(
            data_path="D:/GitHub_Repos/MLOps-Project-DevOps-for-ML/data/olist_customers_dataset.csv",  # Path to the dataset
            min_accuracy=min_accuracy,  # Minimum accuracy threshold for deployment
            workers=3,  # Number of workers to use during deployment
            timeout=60,  # Timeout for the pipeline execution
        )

    # If predict is selected, run the inference pipeline to make predictions
    if predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",  # Name of the pipeline
            pipeline_step_name="mlflow_model_deployer_step",  # The specific step within the pipeline
        )

    # Output MLflow UI command for the user to inspect the experiment runs
    print(
        "You can run:\n"
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n"
        "...to inspect your experiment runs within the MLflow UI.\n"
        "You can find your runs tracked within the 'mlflow_example_pipeline' experiment.\n"
    )

    # Find any existing model server for predictions
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"  # The model being deployed
    )

    # Check if the model server is running and display status
    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])  # Cast the service to the correct type
        if service.is_running:  # If the service is running, display its URL
            print(
                f"The MLflow prediction server is running at:\n   {service.prediction_url}\n"
                f"[italic green]`zenml model-deployer models delete {str(service.uuid)}`[/italic green]"  # Command to delete the model server
            )
        elif service.is_failed:  # If the service has failed, display the error details
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        # If no service is running, inform the user to run the deployment pipeline with the `--deploy` option
        print(
            "No MLflow prediction server is currently running. Run the deployment pipeline with the `--deploy` option."
        )

# Run the deployment function when the script is executed
if __name__ == "__main__":
    run_deployment()