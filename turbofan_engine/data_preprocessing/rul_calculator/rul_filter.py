from abc import abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import util
import turbofan_engine.constants as const_turbofan

class RULFilter(BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> object:
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[const_turbofan.TARGET][X[const_turbofan.TARGET] > const_turbofan.LENGHT_ROI] = \
            const_turbofan.LENGHT_ROI
        return X
    
class RULFilterTwo(BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> object:
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[X[const_turbofan.TARGET] <= const_turbofan.LENGHT_ROI]
        return X
