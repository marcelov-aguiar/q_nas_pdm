from abc import ABC, abstractmethod
import pandas as pd


class Evaluator(ABC):
    """Abstract class that contains operations of an Evaluator."""
    metrics: pd.Series = None

    @abstractmethod
    def evaluate(self,
                 y_test: pd.Series,
                 predictions: pd.Series,
                 verbose: bool = False) -> pd.Series:
        """Method that prints the result of the evaluation metrics."""
        pass
