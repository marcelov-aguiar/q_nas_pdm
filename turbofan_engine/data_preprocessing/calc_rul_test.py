from abc import abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import util
import ds_library.constants.constants_names as const_names
import turbofan_engine.constants as const_turbofan

class CalcRULTest(BaseEstimator, TransformerMixin):
    def __init__(self,
                 df_rul: pd.DataFrame) -> None:
        self.df_rul = df_rul

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> object:
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_text_rul = pd.DataFrame(
            X.groupby(const_turbofan.FEATURE_UNIT_NUMBER)[const_turbofan.FEATURE_TIME].max()).reset_index()
        df_text_rul.columns = [const_names.DEFAULT_ID_NAME, const_turbofan.DEFAULT_MAX_NAME]

        self.df_rul[const_turbofan.TOTAL_RUL] = df_text_rul[const_turbofan.DEFAULT_MAX_NAME] + self.df_rul[const_turbofan.DEFAULT_MAX_NAME]
        self.df_rul = self.df_rul.drop(columns=[const_turbofan.DEFAULT_MAX_NAME])

        X.merge(self.df_rul, on=[const_turbofan.FEATURE_UNIT_NUMBER], how='left')
        X = X.merge(self.df_rul, on=[const_turbofan.FEATURE_UNIT_NUMBER], how='left')
        X[const_turbofan.TARGET] = X[const_turbofan.TOTAL_RUL] - X[const_turbofan.FEATURE_TIME]
        X.drop(const_turbofan.TOTAL_RUL, axis=1, inplace=True)

        return X
