import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from ds_library.models.evaluator.evaluator import Evaluator


class RegressionEvaluator(Evaluator):
    """Class that contains operations necessary to evaluate a regression model."""
    R2 = 'r2'
    MSE = "mse"
    RMSE = 'rmse'
    MAE = 'mae'


    def calculate_r2(self,
                     y_test: pd.Series,
                     predictions: pd.Series) -> float:
        """Method that calculates the R-squared (R2) score.

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted values for the test set.

        Returns
        -------
        float
            Returns the R-squared score, a measure of how well the predictions approximate the actual values.
        """
        return r2_score(y_test, predictions)

    def calculate_mse(self,
                       y_test: pd.Series,
                       predictions: pd.Series) -> float:
        """Method that calculates the Mean Squared Error (MSE).

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted values for the test set.

        Returns
        -------
        float
            Returns the Mean Squared Error, a measure of the average squared difference between the predicted and actual values.
        """
        return mean_squared_error(y_test, predictions)

    def calculate_rmse(self,
                       y_test: pd.Series,
                       predictions: pd.Series) -> float:
        """Method that calculates the Root Mean Squared Error (RMSE).

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted values for the test set.

        Returns
        -------
        float
            Returns the Root Mean Squared Error, the square root of the Mean Squared Error, to measure prediction accuracy.
        """
        return np.sqrt(mean_squared_error(y_test, predictions))

    def calculate_mae(self,
                      y_test: pd.Series,
                      predictions: pd.Series) -> float:
        """Method that calculates the Mean Absolute Error (MAE).

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted values for the test set.

        Returns
        -------
        float
            Returns the Mean Absolute Error, the average of the absolute differences between the predicted and actual values.
        """
        return mean_absolute_error(y_test, predictions)

    def evaluate(self,
                 y_test: pd.Series,
                 predictions: pd.Series,
                 verbose: bool = False) -> pd.Series:
        """Evaluate the regression model by calculating R2, MSE, RMSE, and MAE.

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted values for the test set.
        verbose : bool, optional
            If True, prints the evaluation metrics, by default False

        Returns
        -------
        pd.Series
            Returns a Series containing the evaluation metrics (R2, MSE, RMSE, and MAE) calculated.
        """
        metrics = {
            self.R2: self.calculate_r2(y_test, predictions),
            self.MSE: self.calculate_mse(y_test, predictions),
            self.RMSE: self.calculate_rmse(y_test, predictions),
            self.MAE: self.calculate_mae(y_test, predictions)
        }
        metrics = pd.Series(metrics)
        if verbose:
            print(metrics)
        return metrics