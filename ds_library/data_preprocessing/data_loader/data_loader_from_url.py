from data_loader import DataLoader
import pandas as pd


class DataLoaderFromURL(DataLoader):
    def __init__(self, dataset_URL: str):
        """Reads the CSV file of the dataset

        Parameters
        ----------
        dataset_URL : str
            Dataset URL to be read
        """
        self._dataset = pd.read_csv(dataset_URL)

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset

        Returns
        -------
        pd.DataFrame
            Dataset in pandas DataFrame format
        """
        return self._dataset
