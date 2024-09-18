from typing import Dict
from sklearn.ensemble import RandomForestRegressor
from ds_library.models.model import Model
from ds_library.models.model import ModelType
from ds_library.models.evaluator.evaluator import Evaluator

class RandomForestRegressionModel(Model):
    def __init__(self, **params: Dict):
        self.model = RandomForestRegressor(**params)
        self._alias_model = "RandomForest"

    def type(self) -> Evaluator:
        return ModelType.REGRESSION
