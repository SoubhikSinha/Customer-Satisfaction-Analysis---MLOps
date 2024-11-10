# Model Evaluation Classes

# Importing necessary libraries
import logging  # For logging errors and info during execution
from abc import ABC, abstractmethod  # For defining abstract base classes and abstract methods
import numpy as np  # For numerical operations
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error  # For model evaluation metrics

# Abstract Evaluation Class
class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation of models.
    This is a base class for different evaluation strategies (e.g., MSE, R2, RMSE).
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Abstract method to calculate model evaluation score.
        
        Args:
            y_true : True labels (ground truth)
            y_pred : Predicted labels (model predictions)
        
        Returns:
            None
        """
        pass


# Mean Squared Error (MSE) Evaluation Strategy
class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error (MSE) as the metric.
    MSE is a common metric for regression problems to measure the average squared difference 
    between the true labels and the predicted labels.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the Mean Squared Error (MSE) between true and predicted labels.
        
        Args:
            y_true : True labels (ground truth)
            y_pred : Predicted labels (model predictions)
        
        Returns:
            mse : Calculated Mean Squared Error (MSE) score
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)  # Calculate MSE using scikit-learn's function
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in Calculating MSE: {}".format(e))  # Log any errors that occur
            raise e  # Re-raise the exception for further handling


# R2-Score Evaluation Strategy
class R2(Evaluation):
    """
    Evaluation Strategy that uses R2-Score as the metric.
    R2-Score (Coefficient of Determination) measures the proportion of variance in the dependent 
    variable that is predictable from the independent variables. 
    It is used to evaluate regression models.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the R2-Score between true and predicted labels.
        
        Args:
            y_true : True labels (ground truth)
            y_pred : Predicted labels (model predictions)
        
        Returns:
            r2 : Calculated R2-Score
        """
        try:
            logging.info("Calculating R2-Score")
            r2 = r2_score(y_true, y_pred)  # Calculate R2-Score using scikit-learn's function
            logging.info("R2: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in Calculating R2-Score: {}".format(e))  # Log any errors that occur
            raise e  # Re-raise the exception for further handling


# Root Mean Squared Error (RMSE) Evaluation Strategy
class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error (RMSE) as the metric.
    RMSE is the square root of MSE and provides an error metric in the same units as the original data.
    It's widely used to evaluate regression models as it penalizes larger errors more significantly.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the Root Mean Squared Error (RMSE) between true and predicted labels.
        
        Args:
            y_true : True labels (ground truth)
            y_pred : Predicted labels (model predictions)
        
        Returns:
            rmse : Calculated Root Mean Squared Error (RMSE)
        """
        try:
            logging.info("Calculating RMSE")
            rmse = root_mean_squared_error(y_true, y_pred)  # Calculate RMSE using scikit-learn's function
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in Calculating RMSE: {}".format(e))  # Log any errors that occur
            raise e  # Re-raise the exception for further handling
