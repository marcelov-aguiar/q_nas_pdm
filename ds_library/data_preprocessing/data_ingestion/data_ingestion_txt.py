from typing import List
import numpy as np
import pandas as pd
from ds_library.data_preprocessing.data_ingestion.data_ingestion import DataIngestion


class DataIngestionTxtToParquet(DataIngestion):
    def __init__(self,
                 source_path: str,
                 dest_path: str,
                 features_name: List[str]) -> None:
        self.source_path = source_path
        self.dest_path = dest_path
        self.features_name = features_name

    def process(self):
        df_data = DataIngestionTxtToParquet.txt_to_dataframe(self.source_path,
                                                             self.features_name)
        
        DataIngestionTxtToParquet.save_to_parquet(df_data, self.dest_path)

    @staticmethod
    def save_to_parquet(df_data: pd.DataFrame,
                        dest_path: str) -> None:
        df_data.to_parquet(dest_path, index=False)

    @staticmethod
    def txt_to_dataframe(source_path: str,
                         features_name: List[str]) -> pd.DataFrame:
        data = np.loadtxt(source_path)

        df_data = pd.DataFrame(data, columns=features_name)

        return df_data
