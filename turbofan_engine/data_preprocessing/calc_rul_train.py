from abc import abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import util
import ds_library.constants.constants_names as const_names
import turbofan_engine.constants as const_turbofan

class CalcRULTrain(BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> object:
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = \
            X.groupby(const_turbofan.FEATURE_UNIT_NUMBER).apply(self.__add_rul).reset_index()
        X = X.drop(columns=[const_names.LEVEL_1_NAME, const_names.DEFAULT_INDEX_NAME])

        return X

    def __add_rul(self,
                  df_unit: pd.DataFrame):
        """Responsible for creating the RUL (Remaining Useful Life) feature for each unit in the main DataFrame.

        Parameters
        ----------
        df_unit : pd.DataFrame
            DataFrame of a unit (part of the main DataFrame).
        
        Returns
        -------
        pd.DataFrame
            DataFrame of the unit with the RUL calculated.

        """
        df_unit[const_turbofan.TARGET] = [max(df_unit[const_turbofan.FEATURE_TIME])] * len(df_unit)
        df_unit[const_turbofan.TARGET] = df_unit[const_turbofan.TARGET] - df_unit[const_turbofan.FEATURE_TIME]
        del df_unit[const_turbofan.FEATURE_UNIT_NUMBER]
        return df_unit.reset_index()
