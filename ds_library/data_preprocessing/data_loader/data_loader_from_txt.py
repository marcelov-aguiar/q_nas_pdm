from typing import List
import numpy as np
import pandas as pd
from ds_library.data_preprocessing.data_loader.data_loader import DataLoader


class DataLoaderFromTXT(DataLoader):
    def __init__(self,
                 path_dataset_txt: str,
                 features_name: List[str]):
        """Reads the CSV file of the dataset

        Parameters
        ----------
        dataset_URL : str
            Dataset URL to be read
        """
        data = np.loadtxt(path_dataset_txt)

        self._dataset = pd.DataFrame(data, columns=features_name)

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset

        Returns
        -------
        pd.DataFrame
            Dataset in pandas DataFrame format
        """
        return self._dataset
