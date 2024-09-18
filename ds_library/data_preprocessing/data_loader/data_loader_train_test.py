from typing import List
import pandas as pd
import util
from ds_library.data_preprocessing.data_loader.data_loader import DataLoader
from ds_library.data_preprocessing.data_ingestion.data_ingestion_txt import DataIngestionTxtToParquet
from ds_library.data_preprocessing.data_loader.data_loader_from_txt import DataLoaderFromTXT

class DataLoaderTrainTest(DataLoader):
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 features_name: List[str]):
        """Reads the CSV file of the dataset

        Parameters
        ----------
        dataset_URL : str
            Dataset URL to be read
        """
        self.train_path = train_path
        self.test_path = test_path
        self.features_name = features_name

    def load_dataset(self) -> DataLoader:
        """Load the dataset

        Returns
        -------
        pd.DataFrame
            Dataset in pandas DataFrame format
        """
        self.df_train = DataLoaderFromTXT(source_path=self.train_path,
                                          features_name=self.features_name).load_dataset()

        self.df_test = DataLoaderFromTXT(source_path=self.test_path,
                                         features_name=self.features_name).load_dataset()
        return self
    