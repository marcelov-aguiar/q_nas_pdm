from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from ds_library.models.model_type import ModelType


class Model(BaseEstimator, ABC):
    """Abstract class that contains operations of a Model."""

    def fit(self, X, y):
        """Method to fit the model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Method to do predictions."""
        check_is_fitted(self, 'model')
        return self.model.predict(X)
    
    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def alias_model(self) -> str:
        return self._alias_model

    @abstractmethod
    def type(self) -> ModelType:
        """Method that returns type of a Model, ModelType."""
        pass
