# Model Development Classes

# Importing necessary libraries
import logging  # For logging information and errors
from abc import ABC, abstractmethod  # For defining abstract base classes and abstract methods
from sklearn.linear_model import LinearRegression  # For the Linear Regression model from scikit-learn

# Abstract Model Class
class Model(ABC):
    """
    Abstract Base Class for all models.
    Any specific model (like Linear Regression, Decision Tree, etc.) should inherit from this class
    and implement the `train` method.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Abstract method to train the model.
        
        Args:
            X_train : pd.DataFrame or np.ndarray, training data (features)
            y_train : pd.Series or np.ndarray, training labels (target variable)
        
        Returns:
            model : The trained model (type depends on the concrete model implementation)
        """
        pass


# Linear Regression Model Class
class LinearRegressionModel(Model):
    """
    Concrete implementation of a Linear Regression Model.
    Inherits from the abstract `Model` class and implements the `train` method using Linear Regression.
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the Linear Regression model.
        
        Args:
            X_train : pd.DataFrame or np.ndarray, training data (features)
            y_train : pd.Series or np.ndarray, training labels (target variable)
            **kwargs : Additional arguments to be passed to the Linear Regression model (e.g., fit_intercept, normalize, etc.)
        
        Returns:
            reg : Trained Linear Regression model (scikit-learn LinearRegression object)
        """
        try:
            # Initialize the Linear Regression model
            reg = LinearRegression(**kwargs)
            # Train the model on the provided training data
            reg.fit(X_train, y_train)
            logging.info("Model training completed")  # Log success message
            return reg  # Return the trained model
        except Exception as e:
            logging.error("Error in Training Model: {}".format(e))  # Log any error that occurs during training
            raise e  # Re-raise the exception for further handling

