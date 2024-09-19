import pandas as pd
from turbofan_engine.data_preprocessing.rul_calculator.rul_config import RULConfig

class DataFrameMerger:
    """
    A class responsible for merging DataFrames to include Remaining Useful Life (RUL) 
    values and calculate the final target column.

    Methods
    -------
    merge_rul(df: pd.DataFrame, df_rul: pd.DataFrame, config: RULConfig) -> pd.DataFrame
        Merges the main DataFrame with the RUL DataFrame and calculates the RUL target for each unit.
    """
    def merge_rul(self, df: pd.DataFrame, df_rul: pd.DataFrame, config: RULConfig) -> pd.DataFrame:
        df = df.merge(df_rul, on=[config.feature_unit_number], how='left')
        df[config.target] = df[config.total_rul] - df[config.feature_time]
        return df.drop(config.total_rul, axis=1)