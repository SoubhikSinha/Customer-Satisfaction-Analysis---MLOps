# Importing necessary libraries
import numpy as np
import pandas as pd
import mlflow
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel

# Importing custom pipeline steps
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from pipelines.utils import get_data_for_test
import json

# Docker Settings to integrate ZenML with Docker for containerized pipeline execution
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# Defining a class to trigger deployment based on a minimum accuracy requirement
class DeploymentTriggerConfig(BaseModel):
    """
    Deployment Trigger Config with minimum accuracy requirement
    """
    min_accuracy: float = 0  # Minimum accuracy required for deployment

    class Config:
        protected_namespaces = ()  # Disabling Pydantic warnings for attribute protection

@step(enable_cache=False)
def dynamic_importer() -> str:
    """
    Step to get data for testing
    Retrieves test data and returns it as a string
    """
    data = get_data_for_test()
    return data

@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    """
    Step to decide whether the model deployment should proceed based on accuracy.
    Returns True if the model accuracy meets or exceeds the minimum required accuracy.
    """
    return accuracy > config.min_accuracy

# Defining parameters for loading the MLFlow deployment
class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """
    Class defining the parameters needed to fetch the MLFlow deployment service.
    
    Attributes:
        pipeline_name : The name of the pipeline that deployed the model server
        step_name : The name of the step that deployed the model server
        running : If True, only running services are returned
        model_name : The name of the model deployed
    """
    pipeline_name : str
    step_name : str
    running : bool = True  # Default is to return running services
    model_name : str = "model"

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name : str,
    pipeline_step_name : str,
    running : bool = True,
    model_name : str = "model",
) -> MLFlowDeploymentService:
    """
    Step to load the prediction service started by a specific pipeline deployment.
    
    Args:
        pipeline_name : The pipeline that deployed the prediction service
        pipeline_step_name : The specific step that deployed the service
        running : If True, returns only running services
        model_name : The name of the deployed model
    
    Returns:
        An active MLFlow deployment service
    """
    # Fetching the active MLflow model deployer component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    # Finding existing services based on deployment details
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )

    # If no services are found, raise an error
    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}."
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    
    # Returning the first found deployment service
    return existing_services[0]

@step
def predictor(  # Prediction Service step
    service : MLFlowDeploymentService,
    data : str,
) -> np.array:
    """
    Step to use the loaded prediction service and make predictions based on the input data.
    
    Args:
        service : An active MLFlow deployment service
        data : JSON string containing the data to predict on
    
    Returns:
        Prediction results as a NumPy array
    """
    service.start(timeout=10)  # Starting the service with a timeout of 10 seconds
    data = json.loads(data)  # Parsing the input data from JSON
    data.pop("columns")  # Removing unnecessary "columns" key
    data.pop("index")  # Removing unnecessary "index" key
    
    # Defining the columns expected in the input data
    columns_for_df = [
        "payment_sequential", "payment_installments", "payment_value",
        "price", "freight_value", "product_name_lenght", "product_description_lenght",
        "product_photos_qty", "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm"
    ]
    
    # Converting the input data into a pandas DataFrame
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    
    # Converting the DataFrame to JSON and back to a NumPy array
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.ndarray(json_list)
    
    # Making the prediction using the service
    prediction = service.predict(data)
    return prediction

# Continuous Deployment Pipeline (CD Pipeline)
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str, 
    min_accuracy: float = 0, 
    workers: int = 1, 
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    """
    Pipeline to continuously deploy a model based on its accuracy.
    Only deploys if model accuracy meets or exceeds the minimum threshold.
    """
    # Setting up or creating the experiment for continuous deployment
    experiment_name = "continuous_deployment_pipeline"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)  # Create experiment if it doesn't exist
    mlflow.set_experiment(experiment_name)  # Set the experiment for logging

    # Pipeline steps
    df = ingest_df(data_path=data_path)  # Data ingestion step
    X_train, X_test, y_train, y_test = clean_df(df)  # Data cleaning step
    model = train_model(X_train, X_test, y_train, y_test)  # Model training step
    r2, rmse = evaluate_model(model, X_test, y_test)  # Model evaluation step

    # Deploy the model if it meets the accuracy threshold
    config = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    deployment_decision = deployment_trigger(accuracy=r2, config=config)

    # Conditionally deploy the model based on the evaluation
    if deployment_decision:
        mlflow_model_deployer_step(
            model=model,
            deploy_decision=deployment_decision,
            workers=workers,
            timeout=timeout,
        )

# Inference Pipeline for making predictions using the deployed model
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    pipeline_name : str,
    pipeline_step_name : str
):
    """
    Pipeline to use the deployed model for making predictions.
    Loads the prediction service and feeds it with test data.
    """
    data = dynamic_importer()  # Import test data
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,  # Only look for a non-running service
    )
    
    # Make predictions using the loaded service
    prediction = predictor(service=service, data=data)
    return prediction
