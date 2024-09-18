from abc import abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class Normalizer(BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> object:
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
