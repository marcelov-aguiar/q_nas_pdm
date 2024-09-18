from typing import List
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import util
from ds_library.data_preprocessing.normalizer import Normalizer


class NormStandardScaler(Normalizer):
    """StandardScaler-based normalizer for specific columns in a DataFrame.

    Parameters
    ----------
    Normalizer: BaseEstimator, TransformerMixin
        Normalizer
    """
    def __init__(self,
                 columns_to_norm: List[str]):
        """Initialize the Normalizer with specific columns to be standardized.

        Parameters
        ----------
        columns_to_norm : List[str]
            List of column names to be standardized.
        """
        super().__init__()
        self.scaler = StandardScaler()
        self.columns_to_norm = columns_to_norm

    def fit(self, X:pd.DataFrame=None, y=None) -> object:
        """Fit the StandardScaler to the data in the specified columns.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Data to fit the scaler on, by default None
        y : _type_, optional
            There is no target column required for fitting., by default None

        Returns
        -------
        object
            Returns self, the instance of the transformer.
        """
        self.scaler.fit(X[self.columns_to_norm])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the specified columns using the fitted StandardScaler.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with specified columns standardized.
        """
        df_data = X.copy()
        data = self.scaler.transform(df_data[self.columns_to_norm])
        df_data = self._formats_normalized_dataframe(data, X)

        return df_data
    
    def _formats_normalized_dataframe(self,
                                      data: np.array,
                                      X: pd.DataFrame) -> pd.DataFrame:
        """Reconstructs the DataFrame after standardization.

        Parameters
        ----------
        data : np.array
            Transformed data array.
        X : pd.DataFrame
            Original DataFrame before transformation.

        Returns
        -------
        pd.DataFrame
            DataFrame with standardized columns merged with non-standardized columns.
        """
        df_data = pd.DataFrame(data, columns=self.columns_to_norm, index=X.index)
        df_data = pd.concat([df_data, X.drop(columns=self.columns_to_norm)], axis=1)

        return df_data