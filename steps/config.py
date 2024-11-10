'''
# Importing BaseParameters from zenml.steps to define a ZenML parameter class
from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model Configs: This class is used to define configuration parameters for the model
    using ZenML's parameter class.
    """
    # Defining a parameter to hold the name of the model, with a default value of "LinearRegression"
    model_name : str = "LinearRegression"
'''

# Importing BaseModel from pydantic to define a Pydantic model class for configuration
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    Model Configurations: This class is used to define configuration parameters for the model
    using Pydantic's BaseModel.
    """
    # Defining a parameter to hold the name of the model, with a default value of "LinearRegression"
    model_name: str = "LinearRegression"
