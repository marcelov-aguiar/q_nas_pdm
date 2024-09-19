# rul_calculator/transformers.py

from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from turbofan_engine.data_preprocessing.rul_calculator.rul_config import RULConfig
from turbofan_engine.data_preprocessing.rul_calculator.rul_calculator import RULCalculator
from turbofan_engine.data_preprocessing.rul_calculator.rul_merge import DataFrameMerger


class CalcRULTest(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer class that calculates the Remaining Useful Life (RUL)
    and integrates it into the main DataFrame.

    This class implements the scikit-learn interface, allowing it to be used in pipelines.

    Attributes
    ----------
    rul_calculator : RULCalculator
        An object responsible for calculating the RUL for each unit.
    merger : DataFrameMerger
        A class responsible for merging DataFrames and calculating the target.
    df_rul : pd.DataFrame
        DataFrame containing the initial RUL data.
    config : RULConfig
        Configuration object holding the column names and constants.
    
    Methods
    -------
    fit(X: pd.DataFrame) -> 'CalcRULTest'
        Fits the transformer (dummy implementation as no fitting is needed).
    
    transform(X: pd.DataFrame) -> pd.DataFrame
        Transforms the input DataFrame by calculating RUL and merging the relevant columns.
    """
    def __init__(self, rul_calculator: RULCalculator, merger: DataFrameMerger, 
                 df_rul: pd.DataFrame, config: RULConfig) -> None:
        self.rul_calculator = rul_calculator
        self.merger = merger
        self.df_rul = df_rul
        self.config = config

    def fit(self, X: pd.DataFrame) -> 'CalcRULTest':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_max_time = pd.DataFrame(X.groupby(self.config.feature_unit_number)[self.config.feature_time].max()).reset_index()
        df_max_time.columns = [self.config.feature_unit_number, self.config.default_max_name]
        
        df_rul_calculated = self.rul_calculator.calculate_rul(self.df_rul, df_max_time, self.config)
        
        return self.merger.merge_rul(X, df_rul_calculated, self.config)
