from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC):
    """Abstract class that contains operations of a Data Loader."""

    @abstractmethod
    def load_dataset(self) -> pd.DataFrame:
        """Method that returns a dataset."""
        pass
