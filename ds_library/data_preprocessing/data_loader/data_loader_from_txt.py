from typing import List
import numpy as np
from data_loader import DataLoader
import pandas as pd


class DataLoaderFromTXT(DataLoader):
    def __init__(self,
                 dataset_TXT: str,
                 features_name: List[str]):
        """Reads the CSV file of the dataset

        Parameters
        ----------
        dataset_URL : str
            Dataset URL to be read
        """
        data = np.loadtxt(dataset_TXT)

        self._dataset = pd.DataFrame(data, columns=features_name)

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset

        Returns
        -------
        pd.DataFrame
            Dataset in pandas DataFrame format
        """
        return self._dataset
