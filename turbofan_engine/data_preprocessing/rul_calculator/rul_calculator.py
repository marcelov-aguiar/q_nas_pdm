from abc import ABC, abstractmethod
import pandas as pd
from turbofan_engine.data_preprocessing.rul_calculator.rul_config import RULConfig

class RULCalculator(ABC):
    """
    An abstract base class for calculating Remaining Useful Life (RUL).
    
    Methods
    -------
    calculate_rul(df_rul: pd.DataFrame, df_max_time: pd.DataFrame, config: RULConfig) -> pd.DataFrame
        Abstract method to calculate the RUL for a given unit.
    """
    @abstractmethod
    def calculate_rul(self, df_rul: pd.DataFrame, df_max_time: pd.DataFrame, config: RULConfig) -> pd.DataFrame:
        pass

class DefaultRULCalculator(RULCalculator):
    """
    A concrete implementation of the RULCalculator that calculates the 
    Remaining Useful Life (RUL) for each unit based on the maximum time and RUL DataFrame.
    
    Methods
    -------
    calculate_rul(df_rul: pd.DataFrame, df_max_time: pd.DataFrame, config: RULConfig) -> pd.DataFrame
        Calculates and returns the updated RUL DataFrame with the total RUL for each unit.
    """
    def calculate_rul(self, df_rul: pd.DataFrame, df_max_time: pd.DataFrame, config: RULConfig) -> pd.DataFrame:
        df_rul[config.feature_unit_number] = df_rul.index.values + 1
        df_rul[config.total_rul] = df_max_time[config.default_max_name] + df_rul[config.default_max_name]
        return df_rul.drop(columns=[config.default_max_name])