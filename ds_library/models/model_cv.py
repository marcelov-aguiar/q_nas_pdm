from abc import ABC, abstractmethod
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import util
from ds_library.models.model_type import ModelType


class ModelCV(ABC):
    RANK_TEST_SCORE = "rank_test_score"
    MEAN_TEST_SCORE = 'mean_test_score'
    STD_TEST_SCORE = 'std_test_score'
    SCALER = "scaler"
    BASE_MODEL = "base_model"
    """Abstract class that contains operations of a Model Cross Validation."""

    def fit(self, X, y):
        """Method to fit the model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Method to do predictions."""
        check_is_fitted(self, 'model')
        return self.model.predict(X)

    def evaluate(self, y_test, predictions, verbose: bool = False):
        """Method that performs the evaluation of the model.

        Parameters
        ----------
        y_test : pd.Series
            True values of the target variable.
        predictions : pd.Series
            Predicted values from the model.
        verbose : bool, optional
            Whether to print the evaluation metrics, by default False
        """
        self.metrics = self.type().value.evaluate(y_test, predictions, verbose)

    def get_metrics(self) -> pd.Series:
        """Method that returns the evaluation metrics of the model.

        Returns
        -------
        pd.Series
            Evaluation metrics such as accuracy, precision, recall, and F1 score.
        """
        return self.metrics

    @abstractmethod
    def alias_model(self) -> str:
        """Abstract method to return a string alias for the model."""
        pass

    @abstractmethod
    def type(self) -> ModelType:
        """Method that returns type of a Model, ModelType."""
        pass

    @abstractmethod
    def get_best_mean_score(self) -> float:
        """Abstract method that returns the best mean score obtained during cross-validation."""
        pass

    @abstractmethod
    def get_best_mean_std_score(self) -> float:
        """Abstract method that returns the standard deviation of the best mean score obtained during cross-validation."""
        pass
