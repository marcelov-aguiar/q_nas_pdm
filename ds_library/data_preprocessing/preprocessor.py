from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


class PreProcessor(ABC):
    """Abstract class that contains operations of a data preprocessor."""

    @abstractmethod
    def preprocess(self, dataset) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Method that returns the data treated and separated into x_train, x_test, y_train,y_test."""
        pass
